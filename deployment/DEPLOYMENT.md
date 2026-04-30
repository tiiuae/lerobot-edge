# LeRobot Policy Server — Deployment

Websocket front-end for any HuggingFace LeRobot policy. Mirrors layout of
`starVLA-internal-new/deployment/model_server/server_policy.py`. Client side
stays identical (msgpack-numpy over websocket, `predict_action` route).

## Layout

```
lerobot/deployment/
├── DEPLOYMENT.md                       # this file
└── model_server/
    ├── server_policy.py                # entry point
    └── tools/
        ├── websocket_policy_server.py  # ws server (copied from starVLA)
        └── msgpack_numpy.py            # msgpack codec
```

## 1. Environment

One-time setup. Venv lives inside `lerobot/`.

```bash
cd /home/ibrahim/openpi/lerobot
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e .
uv pip install websockets msgpack
```

Activate before every run:

```bash
source /home/ibrahim/openpi/lerobot/.venv/bin/activate
```

## 2. CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--ckpt_path` | required | Local dir with `config.json` + weights, or HF Hub repo id. |
| `--policy_type` | auto | `act`, `diffusion`, `pi0`, `pi05`, `smolvla`, `tdmpc`, `vqbet`, `groot`, `xvla`, `wall_x`. Auto-detected from local `config.json`. Pass explicitly for HF repos. |
| `--camera_keys` | `cam_high,cam_left_wrist,cam_right_wrist` | CSV of client cam keys. Must equal lerobot keys without `observation.images.` prefix. |
| `--actions_per_chunk` | full chunk | Slice action chunk to this length (e.g. 25 of 50 for `pi05`). |
| `--device` | `cuda` | `cuda`, `cuda:0`, `cpu`. |
| `--use_bf16` | off | Cast policy to bfloat16. |
| `--port` | `8800` | Websocket port. |
| `--idle_timeout` | `-1` | Seconds before auto-shutdown on no traffic. `-1` = never. |
| `--debug_dir` | none | Save per-call inputs/outputs under `<dir>/call_<N>/`. |

## 3. Wire format

### Client → server

```python
{
    "type": "infer",                 # or omit (default)
    "request_id": "anything",
    "images": {
        "cam_high":        np.ndarray(C, H, W) uint8,   # any HxW; resized to policy's image_features
        "cam_left_wrist":  np.ndarray(C, H, W) uint8,
        "cam_right_wrist": np.ndarray(C, H, W) uint8,
    },
    "state":  np.ndarray(D,) float32,   # optional; padded inside policy if needed
    "prompt": "pick the red cup",
}
```

### Server → client

```python
{
    "status": "ok",
    "ok": True,
    "type": "inference_result",
    "request_id": "...",
    "actions": np.ndarray(T, D),     # un-normalised, ready to execute
}
```

Reset between episodes: send `{"type": "reset"}`.

## 4. Pipeline (per inference call)

```
client dict
  └─ _adapt_example      -> {observation.images.<cam>: (1,C,H,W) f32 [0,1],
                              observation.state:        (1,D)     f32,
                              task:                     str}
  └─ preprocessor        (tokenize prompt, normalize state, move to device)
  └─ policy.predict_action_chunk -> (B, T, D) tensor
  └─ postprocessor       (per-step unnormalize, move to cpu)
  └─ {"actions": (T, D) ndarray}
```

Pre/post processors loaded via `make_pre_post_processors(policy.config, pretrained_path=ckpt)`,
so dataset stats and normalisation come straight from the saved checkpoint — no
manual `dataset_statistics.json` parsing.

## 5. Example — pi0.5

`pi05` config: chunk_size=50, image 224×224, max_state_dim=32, max_action_dim=32.
Pick a HF Hub checkpoint (e.g. `lerobot/pi05_base`, `lerobot/pi05_libero`) or a
local fine-tune dir.

### Local checkpoint

```bash
source /home/ibrahim/openpi/lerobot/.venv/bin/activate
cd /home/ibrahim/openpi/lerobot

python -m deployment.model_server.server_policy \
    --ckpt_path /home/ibrahim/storage/VLA_MODELS/pi05/my_finetune \
    --camera_keys cam_high,cam_left_wrist,cam_right_wrist \
    --actions_per_chunk 25 \
    --use_bf16 \
    --device cuda:0 \
    --port 8800 \
    --debug_dir /tmp/pi05_debug
```

### HF Hub checkpoint

```bash
python -m deployment.model_server.server_policy \
    --ckpt_path lerobot/pi05_base \
    --policy_type pi05 \
    --camera_keys cam_high,cam_left_wrist,cam_right_wrist \
    --use_bf16 \
    --port 8800
```

`--policy_type` required for Hub paths because `_detect_policy_type` only reads
local `config.json`.

### Camera key mapping

If trained checkpoint uses different cam names (check `image_features` in its
`config.json`, e.g. `observation.images.front`, `observation.images.wrist`),
match them in `--camera_keys`:

```bash
--camera_keys front,wrist
```

Client must then send `images: {"front": ..., "wrist": ...}`.

## 6. Client snippet

Reuse the existing openpi-client `WebsocketClientPolicy` — protocol is identical.

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="<server-ip>", port=8800)
out = policy.infer({
    "images": {"cam_high": img_chw, "cam_left_wrist": img_l, "cam_right_wrist": img_r},
    "state":  state_vec,
    "prompt": "pick the red cup",
})
actions = out["actions"]   # (T, D) ndarray
```

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `'type' field missing in config.json` | Old / hand-edited config. Pass `--policy_type` explicitly. |
| `KeyError: observation.images.<cam>` inside policy | `--camera_keys` mismatch with policy's `image_features`. |
| `RuntimeError: shape mismatch` on state | Client `state` dim > `max_state_dim`. Pad or trim client side, or check that policy was trained on this robot. |
| OOM on load | Drop `--use_bf16`, or use `--device cpu` for sanity, or pick a smaller variant. |
| Hanging on first call | Tokenizer / weight download from HF Hub. Check stderr; first call slow. |

## 8. Notes

- `predict_action` runs under `torch.no_grad()`.
- Postprocessor invoked per-timestep (it expects `(B, action_dim)`); already
  matches behaviour of `lerobot/async_inference/policy_server.py`.
- `reset()` clears policy internal caches via `policy.reset()` if defined,
  plus zeroes the call counter for debug folder naming.
