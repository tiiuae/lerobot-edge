# Trossen π0.5 Model Server

This is the fixed WebSocket deployment adapter for Trossen stationary
bimanual robots and compatible LeRobot 0.6.2 π0.5 checkpoints. The server
accepts the existing Trossen/OpenPI wire format, converts it to the model
interface, and converts the raw model output back to a 14-joint robot action.

The implementation is `deployment/model_server/server_policy.py`.

## Create the virtual environment

LeRobot 0.6.2 requires Python 3.12 or newer. From the repository root, create
and activate a local `.venv`, then install this LeRobot checkout and its
declared dependencies:

```bash
cd /home/saleha/lerobot-edge

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install .

uv pip install "websockets==17.0.1" "msgpack==1.2.1"
```

`websockets` and `msgpack` are installed explicitly because the deployment
transport uses them, but they are not dependencies of the LeRobot package.

Verify the environment:

```bash
python --version
python -c "import lerobot, torch, websockets, msgpack; print('lerobot', lerobot.__version__); print('torch', torch.__version__); print('websockets', websockets.__version__); print('msgpack', msgpack.__version__)"
```

The LeRobot version should be `0.6.2`. Create the environment once; for later
runs, only activate it with `source .venv/bin/activate`.

## Run the server

Use the LeRobot 0.6.2 environment inside this repository:

```bash
cd /home/saleha/lerobot-edge
source .venv/bin/activate

python -m deployment.model_server.server_policy \
  --ckpt_path /path/to/pi05/pretrained_model \
  --rtc \
  --debug_dir /home/saleha/lerobot-edge/logs/pi05_debug
```

`--rtc` enables Real-Time Chunking during inference; it does not retrain or
modify the checkpoint. Omit `--rtc` to retain ordinary chunk prediction. With
`--rtc`, every inference request should carry the RTC fields described in
[Real-Time Chunking](#real-time-chunking-protocol-v2); without them the server
returns unguided chunks.

Defaults:

| Setting | Value |
|---|---:|
| Device | `cuda` |
| Port | `8800` |
| Idle timeout | disabled (`-1`) |
| Action horizon | 50 steps |
| Dataset frequency | 30 Hz |

Optional CLI flags are `--device`, `--port`, `--idle_timeout`, `--debug_dir`,
`--two_views`, `--single_arm`, and the `--rtc_*` settings. Camera mapping,
stationary-base adaptation, and action mapping are fixed in the code for each
checkpoint layout; the two layout flags below only choose which layout to use.

## Checkpoint layout flags

Two independent flags select the checkpoint layout. They are not coupled, so
all four combinations are valid:

| Flag | Default (flag omitted) | With the flag |
|---|---|---|
| `--two_views` | 3 cameras: `primary`, `secondary`, `wrist` | 2 cameras: `primary`, `wrist` |
| `--single_arm` | 19-D state / 16-D action (bimanual + base) | 7-D right-arm state / action |

Known checkpoints exist for 3 views with `--single_arm` and for 2 views with
`--single_arm` (`pi05_aidrc_rightarm_2cam`). For example:

```bash
python -m deployment.model_server.server_policy \
  --ckpt_path /path/to/pi05_aidrc_rightarm_2cam/pretrained_model \
  --two_views --single_arm
```

The flags must match the checkpoint. The server validates the checkpoint at
startup and refuses to run with a mismatched layout. In every layout the
client interface stays the same: 14-D state in, `(50, 14)` actions out.

## Real-Time Chunking (protocol v2)

RTC conditions each new chunk on the part of the preceding chunk that the robot
is still going to execute, so consecutive chunks join without a jump. The
server keeps the preceding chunk in normalized model space (before the action
unnormalizer), which is the space the RTC guidance runs in.

### Server metadata

With `--rtc`, the metadata the server sends when a client connects includes:

```python
{
    "rtc_enabled": True,
    "rtc_protocol_version": 2,
    "rtc_requires_query_step": True,
    "rtc_requires_dynamic_inference_delay": True,
    "action_horizon": 50,
    ...
}
```

A client should refuse to run RTC against a server that does not report
`rtc_protocol_version: 2`.

### Request fields

Add these fields to every inference request, next to `images`, `state`, and
`prompt`:

```python
{
    "images": {...},
    "state": state,
    "prompt": prompt,
    "rtc_query_step": 120,      # control step at which this observation was captured
    "rtc_inference_delay": 6,   # forecast steps until this chunk is executed
    "rtc_reset": False,         # optional; True clears RTC state first
}
```

| Field | Type | Meaning |
|---|---|---|
| `rtc_query_step` | int ≥ 0 | The client's 30 Hz control-step counter at the moment the observation was captured. It should increase with every request; a repeated or lower step gets no guidance. |
| `rtc_inference_delay` | int ≥ 0 | How many control steps the robot will execute from the preceding chunk before this response is in use: the request round trip plus any time the observation waited to be sent, in 30 Hz steps, rounded up. A client that holds the robot still while it waits for the response sends `0`. |
| `rtc_reset` | bool, default `False` | Clears the policy and RTC state before this request is processed. Any other type is rejected with an error. |

Start the step counter again (for example from `0` at a new episode) only
together with `rtc_reset: True`, a `{"type": "reset"}` message, or a prompt
change. Otherwise the server compares the new steps with the preceding
episode's chunk: lower steps are skipped as stale, and once the counter passes
the old step, the old episode's chunk can be used for guidance.

The legacy field `rtc_steps_executed` from the first RTC protocol is ignored.
A request that sends it without `rtc_query_step` gets no guidance, and the
server logs a warning.

### How the server applies guidance

The preceding chunk was generated for the observation at the preceding
`rtc_query_step`. For a new request the server computes

```text
query_delta     = rtc_query_step - preceding rtc_query_step
prefix          = preceding_chunk[query_delta:]     # still to be executed
overlap_horizon = 50 - query_delta                  # length of that prefix
```

and calls `predict_action_chunk(inference_delay=rtc_inference_delay,
prev_chunk_left_over=prefix, execution_horizon=overlap_horizon)`. The first
`rtc_inference_delay` steps of the new chunk are guided toward the prefix with
full weight (the robot executes those steps of the preceding chunk before the
response arrives), the following steps up to `overlap_horizon` are guided with
a weight that decays according to `--rtc_attention_schedule`, and the rest of
the chunk is unconstrained. Guidance needs `rtc_inference_delay` to be smaller
than `overlap_horizon`.

The guidance always runs to the end of the remaining overlap, as in the RTC
paper. This differs from LeRobot's own RTC rollout, which stops at a fixed
`execution_horizon` (10 in the LeRobot docs). `--rtc_execution_horizon` is
therefore not used for live requests.

The server assumes the client executes the newest chunk exactly as returned,
starting at index `current_step - rtc_query_step` and replacing the previous
chunk entirely. Blending chunks (for example with a temporal ensemble) makes
the executed actions differ from the prefix the server guides with.

### Response fields

The response carries `actions` with shape `(50, 14)` plus:

| Field | Meaning |
|---|---|
| `rtc_query_step` | The request's `rtc_query_step`, or `None` if it was missing or invalid. Use it to match the response to its request. |
| `rtc_guidance_applied` | `True` if the chunk was guided by the preceding chunk. |
| `rtc_query_delta` | `query_delta` as above (negative for a stale request), or `None` if it could not be computed. |
| `rtc_overlap_horizon` | `overlap_horizon` as above, or `None` if it was not computed. |
| `rtc_inference_delay` | The request's `rtc_inference_delay`, or `None` if it was missing or invalid. |
| `rtc_skip_reason` | `None` when guidance was applied, otherwise one of the reasons below. |

### When guidance is skipped

A request that cannot be guided still returns a normal, unguided chunk. Whether
that chunk becomes the reference for the next request depends on the reason:

| `rtc_skip_reason` | Cause | Chunk kept for the next request |
|---|---|---|
| `no_previous_chunk` | First request, the first after a reset or prompt change, or the first after a request without a valid `rtc_query_step` | This chunk |
| `missing_rtc_query_step`, `invalid_rtc_query_step` | `rtc_query_step` absent, negative, or not an integer | None: the stored chunk is cleared, because the response cannot be placed in time |
| `non_increasing_query_step` | `rtc_query_step` is lower than the preceding one (stale request) | The preceding chunk is kept |
| `non_increasing_query_step` | `rtc_query_step` equals the preceding one | This chunk |
| `no_remaining_overlap` | `query_delta` ≥ 50: the preceding chunk has run out | This chunk |
| `missing_rtc_inference_delay`, `invalid_rtc_inference_delay` | `rtc_inference_delay` absent, negative, or not an integer | This chunk |
| `inference_delay_exhausts_overlap` | `rtc_inference_delay` ≥ `overlap_horizon`: the response would arrive after the preceding chunk ends | This chunk |

### Resets and prompt changes

The server clears its policy and RTC state on a `{"type": "reset"}` message,
on a request with `rtc_reset: True`, and, with `--rtc`, whenever the incoming
`prompt` differs from the preceding request's. The first chunk after any of
these is unguided. Requests that repeat the same prompt keep RTC continuity,
so prompt changes are safe during one long client episode (for example with
`--max_steps 10000`) without restarting either process.

### RTC flags

| Flag | Default | Effect |
|---|---|---|
| `--rtc` | off | Enables RTC. |
| `--rtc_max_guidance_weight` | `12.0` | Upper limit on the guidance strength. LeRobot recommends `10.0` for 10-step flow matching. |
| `--rtc_attention_schedule` | `EXP` | Guidance weight between `rtc_inference_delay` and `overlap_horizon`: `EXP` and `LINEAR` decay to zero, `ONES` keeps full weight, `ZEROS` guides only the first `rtc_inference_delay` steps. |
| `--rtc_inference_delay` | `4` | Not used for guidance. It is only logged and reported in the server metadata; each request supplies its own `rtc_inference_delay`. Must be between 0 and 50. |
| `--rtc_execution_horizon` | `50` | Not used for live requests; the server derives `overlap_horizon` per request. Must be between 1 and 50. |

## Compatible checkpoint contract

This server is not tied to one checkpoint path, but every checkpoint must use
the same Trossen feature layout. The server validates this contract at
startup:

| Checkpoint field | Required value |
|---|---|
| Policy type | `pi05` |
| Images | `primary`, `secondary`, `wrist`, each `(3,480,640)`; with `--two_views`: `primary`, `wrist` |
| State | 19-D: `left7 + right7 + base5`; with `--single_arm`: 7-D `right7` |
| Action | 16-D: `left7 + right7 + base_velocity2`; with `--single_arm`: 7-D `right7` |
| Action representation | absolute |
| Observation steps | 1 |
| Chunk/action steps | 50 |
| Model image resolution | `(224,224)` |

Checkpoints with different feature names, ordering, or state/action layouts
need a different adapter and are rejected instead of being silently remapped.

## Fixed client interface

### Client → server

```python
{
    "images": {
        "cam_high":        np.ndarray((3, 480, 640), dtype=np.uint8),
        "cam_left_wrist":  np.ndarray((3, 480, 640), dtype=np.uint8),
        "cam_right_wrist": np.ndarray((3, 480, 640), dtype=np.uint8),
    },
    "state":  np.ndarray((14,), dtype=np.float32),
    "prompt": "task instruction expected by the checkpoint",
}
```

The wire images are **RGB at native camera resolution**, untouched: CHW
`(3,480,640)` arrays, exactly as lerobot's camera produced them. The server
converts them to RGB float32 tensors in `[0,1]` without flipping channels. The
adapter also accepts HWC arrays and converts them to CHW.

Older Trossen clients squashed every frame to `(3,224,224)` and swapped it to
**BGR**. Those still work: launch with `--client_color_space bgr`, and square
frames are restored to the 4:3 letterbox (see below). The default is `rgb`.

Camera mapping is fixed:

| Client key | Checkpoint feature |
|---|---|
| `cam_high` | `observation.images.primary` |
| `cam_left_wrist` | `observation.images.secondary` |
| `cam_right_wrist` | `observation.images.wrist` |

All three cameras are required. With `--two_views`, only `cam_high`
(`primary`) and `cam_right_wrist` (`wrist`) are required; `cam_left_wrist` is
accepted and ignored if the client still sends it.

### Server → client

```python
{
    "actions": np.ndarray((50, 14), dtype=np.float32),
}
```

The WebSocket envelope also contains `status`, `ok`, `type`, and
`request_id`. Send `{"type": "reset"}` between episodes.

## Image resizing

π0.5 training uses aspect-ratio-preserving resize with centered padding. It
does not use center cropping and does not stretch the source image:

```text
Training image (RGB [0,1])       640×480
Content after bilinear resize    224×168
Centered vertical padding       28 top + 28 bottom
Model image                      224×224
```

LeRobot performs this in:

- `src/lerobot/policies/pi05/modeling_pi05.py::_preprocess_images`
- `src/lerobot/policies/common/vla_utils.py::resize_with_pad_torch`

The resize uses bilinear interpolation with `align_corners=False`, then pads
with `0.0`. The subsequent model normalization maps `[0,1]` to `[-1,1]`, so
the black padding becomes `-1.0` at the model input.

The current client sends the native `640×480` frame, so the server passes it
through and `_preprocess_images` letterboxes it exactly as in training. This is
pixel-identical to the training path.

Older clients resized each `640×480` frame to a square `224×224` wire image
before sending it. For those square frames only, the server restores the
intended geometry as follows:

```text
224×224 square wire image
    → resize content to 224×168
    → pad (left=0, right=0, top=28, bottom=28) with 0.0
    → 224×224 model image
```

This restores the training geometry and preserves the full field of view.
Because the client already performed a Lanczos square resize, it cannot be
pixel-identical to the single bilinear resize used in training. Exact pixel
parity would require the client to send the original `640×480` frame.

## State mapping

The robot sends 14 live arm values:

| Client dimensions | Meaning |
|---|---|
| `0:7` | left arm: 6 joints + gripper |
| `7:14` | right arm: 6 joints + gripper |

The checkpoint normalizer requires a 19-dimensional state. The server always
appends the five stationary mobile-base fields:

```text
client state14
    = left7 + right7

checkpoint state19
    = left7 + right7 + [odom_x, odom_y, odom_theta, linear_vel, angular_vel]
    = left7 + right7 + [0, 0, 0, 0, 0]
```

The five zeros are appended, never prepended.

With `--single_arm`, the checkpoint state is only the right arm, with no
base padding:

```text
checkpoint state7 = client state[7:14] = right7
```

## Action mapping

The checkpoint produces 16 absolute action values per timestep:

| Raw dimensions | Meaning | Deployment behavior |
|---|---|---|
| `0:7` | predicted left arm | ignored |
| `7:14` | predicted right arm | used |
| `14:16` | base velocity channels | ignored |

The server returns a homogeneous 14-dimensional action to the client:

```text
client action14 = measured left7 + predicted right7
```

With `--single_arm`, the checkpoint produces 7 values per timestep, which are
the predicted right arm (`right7`).

The measured left-arm pose from the request is repeated across the 50-step
chunk, keeping the parked arm stationary. This applies in both layouts, so the
client always receives 14 columns. Returning `(50,14)` also matches the
client's first-waypoint and action-ensemble code.

Do not change the checkpoint dimensions to 14. The client and checkpoint interfaces are
deliberately different:

| Interface | State dimension | Action dimension |
|---|---:|---:|
| Robot/OpenPI client | 14 | 14 |
| Compatible π0.5 checkpoint | 19 | 16 |
| Compatible π0.5 checkpoint, `--single_arm` | 7 | 7 |

## Language instructions

The server accepts the prompt from the client without an allowlist. This
field may be named `prompt`, `lang`, or `task`. Use an instruction from the
selected checkpoint's training vocabulary. Prompt validation remains on the
client side because different compatible checkpoints may use different tasks.

## Debug logs

Pass the repository-owned debug root shown in the run command:

```text
/home/saleha/lerobot-edge/logs/pi05_debug/
└── run_<UTC timestamp>/
    └── call_<N>/
        ├── instruction.txt          # present when the prompt is non-empty
        ├── image_primary.png
        ├── image_secondary.png
        ├── image_wrist.png
        ├── proprio.npy
        ├── input_state.npy
        ├── input_debug.json
        ├── output_actions.npy
        └── output_debug.json
```

The PNG files are the exact RGB `[0,1]` tensors presented to the LeRobot
preprocessor, converted to uint8 for viewing. With the current client each
image is the native `640×480` frame; the model letterboxes it afterwards. With a
legacy square client it is `224×224` with 28 black rows at the top and bottom. `proprio.npy` contains the raw 14-D state
received from the robot. `input_state.npy` contains the adapted `(1,19)`
checkpoint state, and `output_actions.npy` contains the final `(50,14)` client
action chunk. `input_debug.json` also records the proprio values in readable
JSON. With `--two_views` there is no `image_secondary.png`, and with
`--single_arm` `input_state.npy` is the `(1,7)` right-arm state.

Each server launch creates a new timestamped directory, so resets and restarts
do not overwrite earlier logs. The repository's existing `logs/` ignore rule
keeps these artifacts out of Git.

## Troubleshooting

| Error or symptom | Meaning / fix |
|---|---|
| Tensor size 14 versus 19 in the normalizer | An old server process is passing the client state directly. Restart the updated server so it appends the five base zeros. |
| `setting an array element with a sequence` after the first action chunk | An old server returned 7-D rows to a 14-D client waypoint. Restart the updated server and confirm the server prints `actions (50×14)`. |
| Permission denied under `/tmp/pi05_debug` | Use `--debug_dir /home/saleha/lerobot-edge/logs/pi05_debug`. |
| Images look stretched | Inspect the saved PNG. With the current client it should be an undistorted `640×480` frame. With a legacy square client, `224×168` content centered inside a `224×224` black canvas. |
| Colours look swapped (a wood table appears blue) | The client and `--client_color_space` disagree. The current client sends RGB (the default); use `bgr` only with clients built before it stopped swapping channels. |
| Unexpected behavior despite successful inference | Check `instruction.txt` and use an exact instruction from the selected checkpoint's training data. |
| `KeyboardInterrupt` after pressing Ctrl-C | Normal server shutdown, not an inference failure. |
