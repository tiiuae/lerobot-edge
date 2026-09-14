# Trossen Model Server

This is the fixed WebSocket deployment adapter for Trossen stationary
bimanual robots and compatible LeRobot 0.6.2 policy checkpoints. The server
accepts the existing Trossen/OpenPI wire format, converts it to the model
interface, and converts the raw model output back to a 14-joint robot action.

The wire protocol (client camera keys, BGR images, the 14-dim arm state/action)
is fixed in code. Everything specific to one checkpoint -- camera mapping and
state/action layout -- is **inferred automatically from the checkpoint's own
`config.json`** at startup: no per-checkpoint file to write or maintain. This
is what makes the server generic across policy families (π0.5, mot_falcon, and
anything else LeRobot registers), not just π0.5 -- see "Automatic contract
inference" below.

The implementation is `deployment/model_server/server_policy.py` (the
inference rules live in `deployment/model_server/contract.py`).

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

To serve `pi05`/`mot0`/`mot_falcon` checkpoints, also install:

```bash
uv pip install "transformers>=5.4.0,<5.6.0"   # resolves to 5.5.4
uv pip install scipy                          # resolves to 1.18.1 -- Falcon3-VL's dynamic module loading needs it

# mot_falcon only: the vendored Falcon3-VL transformers fork, as its own
# distinctly-named package so it doesn't collide with real transformers above
# (see "mot_falcon / mot0 prerequisites" below for why this is needed).
uv pip install /lustre1/tier2/projects/falcon-vla/mot/distill_codebase_20260909/teammate_env/wheels/transformers_falcon-4.48.0.dev0-py3-none-any.whl
```

Verify the environment:

```bash
python --version
python -c "import lerobot, torch, websockets, msgpack; print('lerobot', lerobot.__version__); print('torch', torch.__version__); print('websockets', websockets.__version__); print('msgpack', msgpack.__version__)"
```

The LeRobot version should be `0.6.2`. Create the environment once; for later
runs, only activate it with `source .venv/bin/activate`.

Exact versions this has been verified against (`uv pip list`):

| Package | Version |
|---|---|
| `lerobot` | `0.6.2` (editable, this repo) |
| `torch` | `2.11.0+cu128` |
| `transformers` | `5.5.4` |
| `transformers-falcon` | `4.48.0.dev0` |
| `scipy` | `1.18.1` |
| `websockets` | `17.0.1` |
| `msgpack` | `1.2.1` |

If you hit an import error for a class real `transformers` should have (e.g.
`cannot import name 'Qwen3VLConfig' from transformers`), diff your `uv pip list`
against this table before assuming it's a code bug -- see the troubleshooting
table below.

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

Camera mapping and state/action layout are read straight off the checkpoint's
own `config.json` -- there is no profile flag to pass. Before deploying a new
checkpoint, validate it without a GPU or loaded weights:

```bash
python -m deployment.model_server.server_policy \
  --ckpt_path /path/to/new/pretrained_model \
  --dry_run
```

This loads only the checkpoint's `config.json`, infers its contract, and
checks it against this deployment's fixed expectations, then exits -- useful
in CI, and for confirming a new checkpoint is servable before it's ever
loaded onto a GPU. A `--dry_run` failure with "Policy type '...' is not
registered" means that checkpoint's policy family isn't ported into this
LeRobot checkout at all; a `draccus.utils.DecodingError` (unexpected/missing
fields) means the checkpoint's saved `config.json` doesn't match the
currently-registered config dataclass for its type (e.g. the checkpoint
predates a field being added/removed); an error naming an "unrecognized"
camera name or state/action dim means `deployment/model_server/contract.py`'s
rule tables don't cover this checkpoint's shape yet -- see "Automatic
contract inference" below for how to extend them.

`--rtc` enables Real-Time Chunking during inference; it does not retrain or
modify the checkpoint. The defaults match the LeRobot starting point:
`--rtc_inference_delay 4`, `--rtc_execution_horizon 10`,
`--rtc_max_guidance_weight 10.0`, and `--rtc_attention_schedule EXP`.
Omit `--rtc` to retain ordinary synchronous chunk prediction.

The server preserves the `(chunk_size, 14)` response contract (`chunk_size` is
read from the checkpoint itself -- 50 for the original pi05 checkpoints, 25 for
the newer real-world pi05/mot_falcon ones). On each request it keeps the
preceding normalized model chunk and uses its unexecuted prefix for RTC
guidance. If the client consumed actions from the preceding response before
requesting another chunk, include the count in the inference payload:

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
| Action horizon | read from the checkpoint (50 or 25 seen so far) |
| Dataset frequency | 30 Hz |

Optional CLI flags are `--device`, `--port`, `--idle_timeout`, `--debug_dir`,
`--dry_run`, and the `--rtc_*` settings. Camera mapping, wire color space, and
action mapping are inferred automatically (see below), not passed as flags.

## Automatic contract inference

Every checkpoint's `config.json` already declares its policy type, its image
feature names/shapes, and its state/action dimensions. What it does **not**
declare is how those relate to the *fixed* Trossen wire protocol (client
camera keys `cam_high`/`cam_left_wrist`/`cam_right_wrist`, a 14-dim arm
state/action) -- that mapping is inferred by
`deployment/model_server/contract.py`'s `infer_contract(config)`, using a
small, fixed rule set rather than a hand-written file per checkpoint. Every
checkpoint contract seen so far -- π0.5 and mot_falcon, 2 or 3 cameras, three
distinct state/action widths, two distinct camera-naming conventions -- is
fully determined by two things alone:

**Camera names**: for each image feature the checkpoint declares (e.g.
`observation.images.cam_high`), the bare name (`cam_high`) is matched to a
client camera key either directly (identity -- used by every `mot_falcon`
checkpoint and the newer "real-world" pi05 checkpoints) or via a small fixed
legacy table (the original pi05 deployment checkpoints name their features
`primary`/`secondary`/`wrist` instead):

| Checkpoint feature name | Client camera key |
|---|---|
| `cam_high` / `cam_left_wrist` / `cam_right_wrist` | itself (identity) |
| `primary` | `cam_high` |
| `secondary` | `cam_left_wrist` |
| `wrist` | `cam_right_wrist` |

Any client camera key the checkpoint doesn't need is tolerated as an ignored
extra if the client still sends it (e.g. a 2-camera checkpoint still receiving
`cam_left_wrist`). A checkpoint image feature name matching neither raises a
clear `ValueError` naming exactly what's unrecognized -- add a rule to
`_LEGACY_CAMERA_NAMES` in `contract.py` if this is a new, legitimate
convention, rather than guessing.

**State/action width**: the checkpoint's own declared state and action
dimensions select a fixed segment layout (each describes how to build the
checkpoint's vector from the client's 14-dim one, or vice versa):

| Checkpoint state dim | Layout |
|---:|---|
| 7 | client's `[7:14]` slice (right arm only; left arm unmodelled) |
| 14 | client's full `[0:14]` (nothing padded) |
| 19 | client's full `[0:14]` + 5 zeros (bimanual + stationary mobile base) |

| Checkpoint action dim | Layout |
|---:|---|
| 7 | held client `[0:7]` (measured left arm, broadcast) + model's `[0:7]` |
| 14 | model's full `[0:14]` (nothing held, both arms predicted) |
| 16 | held client `[0:7]` + model's `[7:14]` (base-velocity columns discarded) |

A checkpoint whose state or action width isn't one of these raises a clear
`ValueError` naming the unrecognized dimension -- add an entry to
`_STATE_LAYOUT_BY_DIM` / `_ACTION_LAYOUT_BY_DIM` in `contract.py` for a
genuinely new layout, rather than guessing one.

Separately, `validate_universal_expectations(config)` checks a small set of
fixed values every checkpoint compatible with this deployment shares
regardless of policy type or training run (dtype, normalization scheme, image
resolution, `max_state_dim`/`max_action_dim`, etc.) -- this is *not*
per-checkpoint-family data, just a sanity net that catches a genuinely
incompatible checkpoint early.

### mot_falcon / mot0 prerequisites

`mot_falcon`/`mot0` policy/config classes are ported and registered
(`src/lerobot/policies/mot0/`, `mot_falcon/`, sourced from lerobot-internal's
`kebin/realworld-dataset-training` branch; `mot_falcon_pi05expert`, the
distilled falcon-mot student, isn't on that branch yet). **Confirmed working
end to end** (config load, `--dry_run`, weight loading, and a real
`predict_action` call) against several real checkpoints. Three environment
prerequisites, beyond the usual `lerobot[pi]`-equivalent install:

1. **The vendored Falcon3-VL `transformers` fork** -- Falcon3-VL isn't a model
   type upstream `transformers` recognizes natively. Install it as its own
   distinctly-named package (`transformers_falcon`, so it doesn't collide with
   the real `transformers` pi05/mot0 need):
   ```bash
   uv pip install /lustre1/tier2/projects/falcon-vla/mot/distill_codebase_20260909/teammate_env/wheels/transformers_falcon-4.48.0.dev0-py3-none-any.whl
   ```
   `server_policy.py` handles the rest automatically via
   `lerobot.policies.mot_falcon.transformers_falcon_shim.masquerade_transformers_falcon()`,
   which temporarily swaps `sys.modules['transformers']` only while
   constructing a `mot_falcon` policy.
2. **`scipy`** (`uv pip install scipy`, or `lerobot[scipy-dep]`) -- needed by
   the Falcon3-VL processor's dynamic module loading.
3. **Readable checkpoint files** -- MOT-Falcon checkpoints have been seen with
   `.safetensors` files (model weights, processor state) saved at mode
   `0600`/`0660` owned by the training user's personal group, unreadable by
   anyone else even with matching project-group membership. `ls -la` the
   checkpoint directory before deploying; ask its owner to widen permissions
   (`chmod o+r` or a shared group) if needed.

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

Camera mapping is inferred per-checkpoint (see "Automatic contract inference"
above), not fixed in code. A 3-camera, original-convention pi05 checkpoint's
inferred mapping looks like:

| Client key | Checkpoint feature |
|---|---|
| `cam_high` | `observation.images.primary` |
| `cam_left_wrist` | `observation.images.secondary` |
| `cam_right_wrist` | `observation.images.wrist` |

with all three cameras required; a 2-camera checkpoint drops
`cam_left_wrist`'s requirement (tolerating it as an ignored extra if the
client still sends it). Every `mot_falcon` checkpoint and the newer
"real-world" pi05 checkpoints instead declare their image features under the
client's own camera key names directly (`cam_high` → `observation.images.cam_high`,
etc.) -- both conventions are recognized automatically.

### Server → client

```python
{
    "actions": np.ndarray((chunk_size, 14), dtype=np.float32),
}
```

`chunk_size` is read from the checkpoint (50 for the original pi05 checkpoints,
25 for the newer ones). The WebSocket envelope also contains `status`, `ok`,
`type`, and `request_id`. Send `{"type": "reset"}` between episodes.

## Image resizing

π0.5 training uses aspect-ratio-preserving resize with centered padding. It
does not use center cropping and does not stretch the source image:

```text
Training image (RGB [0,1])       640×480
Content after bilinear resize    224×168
Centered vertical padding       28 top + 28 bottom
Model image                      224×224
```

This describes π0.5's own internal preprocessing, which the adapter's
`_restore_squashed_aspect` reconstructs the geometry for; other policy
families may implement their native resize differently. LeRobot performs
π0.5's version in:

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

## State and action mapping

The robot always sends/receives 14 values (client's fixed wire contract):

| Client dimensions | Meaning |
|---|---|
| `0:7` | left arm: 6 joints + gripper |
| `7:14` | right arm: 6 joints + gripper |

How these relate to a given checkpoint's own state/action vectors is inferred
automatically from the checkpoint's declared dimensions (see "Automatic
contract inference" above), not fixed in code or hand-written per checkpoint.
The 19-dim-state / 16-dim-action (bimanual + base) case is the running example
below; other checkpoints use the same segment mechanism with different
widths.

**State**: for a 19-dim-state checkpoint, the server appends the five
stationary mobile-base fields:

```text
client state14
    = left7 + right7

checkpoint state19
    = left7 + right7 + [odom_x, odom_y, odom_theta, linear_vel, angular_vel]
    = left7 + right7 + [0, 0, 0, 0, 0]
```

The five zeros are appended, never prepended.

**Action**: a 16-dim-action checkpoint produces 16 absolute action values per
timestep:

| Raw dimensions | Meaning | Deployment behavior |
|---|---|---|
| `0:7` | predicted left arm | ignored |
| `7:14` | predicted right arm | used |
| `14:16` | base velocity channels | ignored |

The server returns a homogeneous 14-dimensional action to the client:

```text
client action14 = measured left7 + predicted right7
```

The measured left-arm pose from the request is repeated across the chunk,
keeping the parked arm stationary. Returning `(chunk_size, 14)` also matches
the client's first-waypoint and action-ensemble code.

Do not change the checkpoint dimensions to 14. The client and checkpoint
interfaces are deliberately different, and vary per checkpoint, e.g.:

| Interface | State dimension | Action dimension |
|---|---:|---:|
| Robot/OpenPI client | 14 | 14 |
| bimanual+base checkpoint | 19 | 16 |
| right-arm-only checkpoint | 7 | 7 |
| full-bimanual, no padding checkpoint | 14 | 14 |

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
        ├── image_<camera>.png       # one per checkpoint camera
        ├── proprio.npy
        ├── input_state.npy
        ├── input_debug.json
        ├── output_actions.npy
        └── output_debug.json
```

The PNG files are the exact RGB `[0,1]` tensors presented to the LeRobot
preprocessor, converted to uint8 for viewing. Each image is `224×224` with
28 black rows at the top and bottom. `proprio.npy` contains the raw 14-D state
received from the robot. `input_state.npy` contains the adapted checkpoint
state, and `output_actions.npy` contains the final client action chunk.
`input_debug.json` also records the proprio values in readable JSON.

Each server launch creates a new timestamped directory, so resets and restarts
do not overwrite earlier logs. The repository's existing `logs/` ignore rule
keeps these artifacts out of Git.

## Troubleshooting

| Error or symptom | Meaning / fix |
|---|---|
| Tensor size 14 versus 19 in the normalizer | An old server process is passing the client state directly. Restart the updated server so it appends the five base zeros. |
| `setting an array element with a sequence` after the first action chunk | An old server returned 7-D rows to a 14-D client waypoint. Restart the updated server and confirm the server prints `actions (N×14)`. |
| Permission denied under `/tmp/pi05_debug` | Use `--debug_dir /home/saleha/lerobot-edge/logs/pi05_debug`. |
| Images look stretched | Inspect the saved PNG. Correct output is `224×168` content centered inside a `224×224` black canvas. |
| Unexpected behavior despite successful inference | Check `instruction.txt` and use an exact instruction from the selected checkpoint's training data. |
| `KeyboardInterrupt` after pressing Ctrl-C | Normal server shutdown, not an inference failure. |
| `Checkpoint does not match this deployment's fixed expectations` | The checkpoint's `config.json` doesn't match the universal sanity checks in `contract.py`'s `validate_universal_expectations` (dtype, normalization, image resolution, etc.); the error lists every mismatched field. Not fixable via a flag -- the checkpoint genuinely differs from every known-good one this deployment serves. |
| `Unrecognized checkpoint image feature '...'` / `... state/action dim N doesn't match a known layout` | The checkpoint's camera names or state/action width don't fit any rule in `deployment/model_server/contract.py`. Add a rule there (not a new file) if this is a legitimate new convention -- see "Automatic contract inference" above. |
| `Policy type '...' is not registered` | The checkpoint's policy type has no config/policy class in this LeRobot checkout (e.g. `mot_falcon_pi05expert`, the distilled falcon-mot student, isn't ported yet -- `mot0`/`mot_falcon` are). Blocks both `--dry_run` and normal serving -- port the policy class into this repo first, from lerobot-internal's `kebin/realworld-dataset-training` branch specifically (see that branch for anything newer). |
| `draccus.utils.DecodingError: The fields ... are not valid for ...Config` | The checkpoint's saved `config.json` has a field the currently-registered config dataclass for its type doesn't declare (or is missing one it requires) -- usually because the checkpoint was saved by an older/newer version of that policy's config class than what's in this checkout. Needs the config class or the checkpoint reconciled. |
| `ModuleNotFoundError: No module named 'transformers_falcon'` | The vendored Falcon3-VL `transformers` fork isn't installed in this environment. `uv pip install` the wheel referenced above ("mot_falcon / mot0 prerequisites"), then retry. |
| `cannot import name 'X' from transformers` (for a class real `transformers` should have, e.g. `Qwen3VLConfig`) -- for a *different* policy than the one you're deploying, or outside any `mot_falcon` construction | Not a `mot_falcon`-masquerade leak -- `masquerade_transformers_falcon()` is verified to correctly scope and restore `sys.modules['transformers']` around `mot_falcon` construction only. This means your environment's real `transformers` install itself is stale/mismatched. Compare `uv pip list` against the version table above ("Create the virtual environment") and reinstall `transformers` if it differs. |
| `ImportError: This modeling file requires the following packages ...: scipy` | Install `scipy` (`uv pip install scipy`) -- needed by the Falcon3-VL processor's dynamic module loading, independent of the transformers fork. |
| `KeyError: "Processor step 'delta_actions_processor' not found in registry"` | lerobot-internal serializes `RelativeActionsProcessorStep` under the name `delta_actions_processor`; this repo's own convention is `relative_actions_processor` for the same class. `mot_falcon`'s `processor_mot_falcon.py` registers a read-only alias for the former on import -- if this still fires, confirm `mot_falcon` was actually imported before loading the checkpoint's processors (e.g. via `lerobot.policies`, not a narrower import). |
| `FileNotFoundError: ... model.safetensors` (or another `.safetensors` file) despite the file existing | Permissions, not a missing file -- `ls -la` the checkpoint directory; MOT-Falcon checkpoints have been seen with `.safetensors` files at mode `0600`/`0660` owned by the training user's personal group, unreadable by anyone else even if group membership looks right. Ask the checkpoint's owner to widen read permissions. |
| `--profile` / `--2_views` / `-7AD` flags no longer exist | Camera mapping and state/action layout are inferred automatically from the checkpoint's own `config.json` -- see "Automatic contract inference" above. Just pass `--ckpt_path`. |
