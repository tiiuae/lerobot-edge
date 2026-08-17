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
modify the checkpoint. The defaults match the LeRobot starting point:
`--rtc_inference_delay 4`, `--rtc_execution_horizon 10`,
`--rtc_max_guidance_weight 10.0`, and `--rtc_attention_schedule EXP`.
Omit `--rtc` to retain ordinary synchronous chunk prediction.

The server preserves the `(50, 14)` response contract. On each request it
keeps the preceding normalized model chunk and uses its unexecuted prefix for
RTC guidance. If the client consumed actions from the preceding response
before requesting another chunk, include the count in the inference payload:

```python
{
    "images": {...},
    "state": state,
    "prompt": prompt,
    "rtc_steps_executed": 12,
}
```

The value is relative to the preceding response and defaults to `0`. A client
may also override the configured latency estimate for an individual request
with `"rtc_inference_delay": 6`.

When the incoming `prompt` changes, the server automatically clears its policy
and RTC inference state before predicting the first chunk for the new prompt.
Requests that repeat the same prompt retain RTC continuity. This makes prompt
changes safe during one long client episode (for example with
`--max_steps 10000`) without restarting either process.

Defaults:

| Setting | Value |
|---|---:|
| Device | `cuda` |
| Port | `8800` |
| Idle timeout | disabled (`-1`) |
| Action horizon | 50 steps |
| Dataset frequency | 30 Hz |

Optional CLI flags are `--device`, `--port`, `--idle_timeout`, `--debug_dir`,
and the `--rtc_*` settings. Camera mapping, wire color space, stationary-base
adaptation, and action mapping are intentionally fixed in the code.

## Compatible checkpoint contract

This server is not tied to one checkpoint path, but every checkpoint must use
the same Trossen feature layout. The server validates this contract at
startup:

| Checkpoint field | Required value |
|---|---|
| Policy type | `pi05` |
| Images | `primary`, `secondary`, `wrist`, each `(3,480,640)` |
| State | 19-D: `left7 + right7 + base5` |
| Action | 16-D: `left7 + right7 + base_velocity2` |
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
        "cam_high":        np.ndarray((3, 224, 224), dtype=np.uint8),
        "cam_left_wrist":  np.ndarray((3, 224, 224), dtype=np.uint8),
        "cam_right_wrist": np.ndarray((3, 224, 224), dtype=np.uint8),
    },
    "state":  np.ndarray((14,), dtype=np.float32),
    "prompt": "task instruction expected by the checkpoint",
}
```

The wire images are BGR. The server converts them to RGB float32 tensors in
`[0,1]`. The current client sends CHW `(3,224,224)` arrays; the adapter also
accepts HWC `(224,224,3)` arrays and converts them to CHW.

Camera mapping is fixed:

| Client key | Checkpoint feature |
|---|---|
| `cam_high` | `observation.images.primary` |
| `cam_left_wrist` | `observation.images.secondary` |
| `cam_right_wrist` | `observation.images.wrist` |

All three cameras are required.

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

The existing OpenPI client has already resized each `640×480` frame to a
square `224×224` wire image. The server therefore restores the intended
geometry as follows:

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

The measured left-arm pose from the request is repeated across the 50-step
chunk, keeping the parked arm stationary. Returning `(50,14)` also matches the
client's first-waypoint and action-ensemble code.

Do not change the checkpoint dimensions to 14. The two interfaces are
deliberately different:

| Interface | State dimension | Action dimension |
|---|---:|---:|
| Robot/OpenPI client | 14 | 14 |
| Compatible π0.5 checkpoint | 19 | 16 |

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
preprocessor, converted to uint8 for viewing. Each image is `224×224` with
28 black rows at the top and bottom. `proprio.npy` contains the raw 14-D state
received from the robot. `input_state.npy` contains the adapted `(1,19)`
checkpoint state, and `output_actions.npy` contains the final `(50,14)` client
action chunk. `input_debug.json` also records the proprio values in readable
JSON.

Each server launch creates a new timestamped directory, so resets and restarts
do not overwrite earlier logs. The repository's existing `logs/` ignore rule
keeps these artifacts out of Git.

## Troubleshooting

| Error or symptom | Meaning / fix |
|---|---|
| Tensor size 14 versus 19 in the normalizer | An old server process is passing the client state directly. Restart the updated server so it appends the five base zeros. |
| `setting an array element with a sequence` after the first action chunk | An old server returned 7-D rows to a 14-D client waypoint. Restart the updated server and confirm the server prints `actions (50×14)`. |
| Permission denied under `/tmp/pi05_debug` | Use `--debug_dir /home/saleha/lerobot-edge/logs/pi05_debug`. |
| Images look stretched | Inspect the saved PNG. Correct output is `224×168` content centered inside a `224×224` black canvas. |
| Unexpected behavior despite successful inference | Check `instruction.txt` and use an exact instruction from the selected checkpoint's training data. |
| `KeyboardInterrupt` after pressing Ctrl-C | Normal server shutdown, not an inference failure. |
