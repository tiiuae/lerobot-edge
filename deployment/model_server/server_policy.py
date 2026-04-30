# Copyright 2026. Licensed under the MIT License.
# LeRobot policy server — websocket front-end mirroring the starVLA server_policy.py
# layout, but powered by a HuggingFace LeRobot policy + processor pipeline.

import argparse
import dataclasses
import json
import logging
import os
import socket
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image as _PIL_Image

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


# ANSI colour helpers
_G = "\033[32m"
_Y = "\033[33m"
_C = "\033[36m"
_R = "\033[0m"


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------

class PolicyServer:
    """
    Wrapper between the WebSocket server and a HuggingFace LeRobot policy.

    Responsibilities:
      1. Observation adaptation  — translates the openpi-client format
                                   (`{"images": {...}, "state": ndarray, "prompt": str}`)
                                   to the LeRobot policy format
                                   (`observation.images.<cam>`, `observation.state`, `task`).
      2. Pre/Post processing     — runs the policy's preprocessor pipeline
                                   (tokenisation, normalisation, device move),
                                   then `policy.predict_action_chunk`,
                                   then the postprocessor pipeline (unnormalisation).
      3. Action post-processing  — returns `{"actions": (T, D)}` numpy.
      4. Coloured action logging — prints every predicted chunk to stdout.
      5. Debug I/O saving        — optional (pass debug_dir=None to skip).
                                   Saved per call under <debug_dir>/call_<N>/:
                                     instruction.txt
                                     image_<k>_<i>.png
                                     output_actions.npy
                                     output_debug.json

    Client observation format (each example):
        {
            "images": {"cam_high": ndarray(C,H,W) uint8, ...},
            "state":  ndarray(D,) float,        # optional
            "prompt": str,
        }
    """

    def __init__(
        self,
        policy,
        preprocessor,
        postprocessor,
        camera_map: dict[str, str],             # client cam key → lerobot cam key (without `observation.images.`)
        actions_per_chunk: int | None = None,
        debug_dir: str | None = None,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._camera_map = camera_map
        self._actions_per_chunk = actions_per_chunk
        self._device = next(policy.parameters()).device
        self._debug = debug_dir is not None
        self._call_idx = 0

        # Resolve target image (H, W) per lerobot cam from policy config
        self._image_targets: dict[str, tuple[int, int]] = {}
        img_feats = getattr(policy.config, "image_features", {}) or {}
        for lerobot_cam in camera_map.values():
            key = f"{OBS_IMAGES}.{lerobot_cam}"
            if key in img_feats:
                _, h, w = img_feats[key].shape
                self._image_targets[lerobot_cam] = (h, w)

        expected = sorted(k for k in img_feats if k.startswith(f"{OBS_IMAGES}."))
        provided = sorted(f"{OBS_IMAGES}.{v}" for v in camera_map.values())
        if expected and set(expected) - set(provided):
            logging.warning(
                "Policy expects image features %s but camera_map only provides %s — "
                "missing cams will be absent from the batch.", expected, provided,
            )
        logging.info("PolicyServer: device=%s, camera_map=%s, image_targets=%s",
                     self._device, camera_map, self._image_targets)

        if self._debug:
            self._debug_dir = Path(debug_dir)
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            logging.info("PolicyServer: debug I/O → %s", self._debug_dir)

    # ------------------------------------------------------------------
    # 1. Observation adaptation
    # ------------------------------------------------------------------

    @staticmethod
    def _to_chw_uint8(arr: np.ndarray) -> np.ndarray:
        """Accept (C,H,W) or (H,W,C); return contiguous (C,H,W) uint8."""
        a = np.asarray(arr)
        if a.ndim == 3 and a.shape[-1] in (1, 3, 4) and a.shape[0] not in (1, 3, 4):
            a = a.transpose(2, 0, 1)                # HWC → CHW
        if a.dtype != np.uint8:
            a = (a * 255 if a.max() <= 1.0 else a).clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(a)

    def _image_to_tensor(self, lerobot_cam: str, arr: np.ndarray) -> torch.Tensor:
        """(C,H,W) uint8 ndarray → (1,C,H,W) float32 tensor in [0,1], resized if needed."""
        chw = self._to_chw_uint8(arr)
        t = torch.from_numpy(np.array(chw, copy=True)).to(torch.float32) / 255.0
        target = self._image_targets.get(lerobot_cam)
        if target is not None and (t.shape[1], t.shape[2]) != target:
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=target, mode="bilinear", align_corners=False
            ).squeeze(0)
        return t.unsqueeze(0).contiguous()

    def _adapt_example(self, ex: dict) -> dict:
        """openpi client format → lerobot Observation dict."""
        observation: dict = {}

        images_dict = ex.get("images", {}) or {}
        matched = []
        for client_cam, lerobot_cam in self._camera_map.items():
            if client_cam not in images_dict:
                continue
            observation[f"{OBS_IMAGES}.{lerobot_cam}"] = self._image_to_tensor(
                lerobot_cam, images_dict[client_cam]
            )
            matched.append(client_cam)
        if images_dict and not matched:
            logging.warning(
                "No client image keys matched camera_map. Client sent %s, expected one of %s.",
                sorted(images_dict.keys()), sorted(self._camera_map.keys()),
            )

        state = ex.get("state")
        if state is not None:
            s = torch.as_tensor(np.array(state, copy=True), dtype=torch.float32)
            if s.ndim == 1:
                s = s.unsqueeze(0)                  # (D,) → (1, D)
            observation[OBS_STATE] = s

        lang = ex.get("prompt") or ex.get("lang") or ex.get("task") or ""
        if lang == "None":
            lang = ""
        observation["task"] = lang
        return observation

    # ------------------------------------------------------------------
    # 2. Debug helpers
    # ------------------------------------------------------------------

    def _save_inputs(self, call_dir: Path, ex: dict) -> None:
        lang = ex.get("prompt") or ex.get("lang") or ex.get("task")
        if lang:
            (call_dir / "instruction.txt").write_text(str(lang))
        images_dict = ex.get("images", {}) or {}
        for cam, arr in images_dict.items():
            chw = self._to_chw_uint8(arr)
            hwc = chw.transpose(1, 2, 0)
            if hwc.shape[-1] == 1:
                hwc = hwc.squeeze(-1)
            _PIL_Image.fromarray(hwc).save(call_dir / f"image_{cam}.png")

    def _save_outputs(self, call_dir: Path, actions: np.ndarray) -> None:
        np.save(call_dir / "output_actions.npy", actions)
        debug = {"meta": {"call_idx": self._call_idx}, "actions": actions.tolist()}
        (call_dir / "output_debug.json").write_text(json.dumps(debug, indent=2))

    # ------------------------------------------------------------------
    # 3. Inference pipeline
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_pipeline(self, observation: dict) -> np.ndarray:
        """preprocessor → policy.predict_action_chunk → postprocessor → (T, D) ndarray."""
        observation = self._preprocessor(observation)

        chunk = self._policy.predict_action_chunk(observation)   # (B, T, D) or (B, D)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)                           # → (B, 1, D)
        if self._actions_per_chunk is not None:
            chunk = chunk[:, : self._actions_per_chunk, :]

        # postprocessor expects (B, action_dim) per call
        _, T, _ = chunk.shape
        processed = [self._postprocessor(chunk[:, i, :]) for i in range(T)]
        actions = torch.stack(processed, dim=1).squeeze(0)       # (T, D)
        return actions.detach().cpu().to(torch.float32).numpy()

    # ------------------------------------------------------------------
    # 4. Coloured action print
    # ------------------------------------------------------------------

    @staticmethod
    def _print_actions(actions: np.ndarray, call_idx: int) -> None:
        T, D = actions.shape
        print(f"{_C}[call {call_idx:05d}]{_R} {_Y}actions ({T}×{D}){_R}")
        for t, row in enumerate(actions):
            vals = "  ".join(f"{_G}{v:+.4f}{_R}" for v in row)
            print(f"  t={t:02d}  [ {vals} ]")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict_action(self, examples=None, **kwargs) -> dict:
        if examples is None:
            examples = []
        if not isinstance(examples, list):
            examples = [examples]
        if not examples:
            raise ValueError("predict_action: empty examples list")

        ex = examples[0]                                         # realtime — single obs

        # --- 1. Adapt observation ---
        observation = self._adapt_example(ex)

        # --- 2. (optional) save inputs ---
        if self._debug:
            call_dir = self._debug_dir / f"call_{self._call_idx:05d}"
            call_dir.mkdir(parents=True, exist_ok=True)
            self._save_inputs(call_dir, ex)

        # --- 3. Run pipeline ---
        actions = self._run_pipeline(observation)
        self._print_actions(actions, self._call_idx)

        # --- 4. (optional) save outputs ---
        if self._debug:
            self._save_outputs(call_dir, actions)

        self._call_idx += 1
        return {"actions": actions}

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()
        self._call_idx = 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _detect_policy_type(pretrained_path: str) -> str:
    """Read config.json to recover the policy type registered under draccus."""
    cfg_file = Path(pretrained_path) / "config.json"
    if not cfg_file.is_file():
        raise FileNotFoundError(
            f"config.json not found in {pretrained_path}. "
            f"Pass --policy_type explicitly for HF Hub repos."
        )
    with open(cfg_file) as f:
        cfg = json.load(f)
    if "type" not in cfg:
        raise ValueError(f"'type' field missing in {cfg_file}")
    return cfg["type"]


def _config_class_for(policy_type: str):
    """Lookup the dataclass registered under draccus for a given policy type."""
    return PreTrainedConfig.get_choice_class(policy_type)


def _parse_camera_map(spec: str) -> dict[str, str]:
    """Parse `--camera_keys` CSV.

    Accepts either bare names (`primary`) or `client_key:lerobot_key` mappings
    (`cam_high:primary`). Bare names map identically.
    """
    out: dict[str, str] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            client, lerobot = entry.split(":", 1)
            out[client.strip()] = lerobot.strip()
        else:
            out[entry] = entry
    return out


def _staging_dir_with_clean_config(pretrained_path: str, policy_type: str) -> str:
    """Return a directory whose `config.json` only contains fields known to the
    target dataclass. All other files (weights, processors) are symlinked from
    the original path, so loading is zero-copy.

    Drops keys from `config.json` that the registered config dataclass does not
    declare. Useful for checkpoints saved with a newer/customised training fork
    that introduced extra hyperparameters.
    """
    src = Path(pretrained_path).resolve()
    cfg_path = src / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    cfg_cls = _config_class_for(policy_type)
    valid = {f.name for f in dataclasses.fields(cfg_cls)} | {"type"}
    dropped = sorted(set(cfg) - valid)
    if not dropped:
        return str(src)                                              # nothing to strip

    cleaned = {k: v for k, v in cfg.items() if k in valid}
    logging.warning("Stripping unknown config.json fields: %s", dropped)

    staging = Path(tempfile.mkdtemp(prefix="lerobot_clean_cfg_"))
    for entry in src.iterdir():
        if entry.name == "config.json":
            continue
        os.symlink(entry, staging / entry.name)
    with open(staging / "config.json", "w") as f:
        json.dump(cleaned, f, indent=2)
    logging.info("Loading from staged dir: %s", staging)
    return str(staging)


def main(args) -> None:
    policy_type = args.policy_type or _detect_policy_type(args.ckpt_path)
    logging.info("Loading policy type=%s from %s", policy_type, args.ckpt_path)

    load_path = args.ckpt_path
    if Path(args.ckpt_path).is_dir() and not args.no_strip_unknown:
        load_path = _staging_dir_with_clean_config(args.ckpt_path, policy_type)

    policy_class = get_policy_class(policy_type)
    policy = policy_class.from_pretrained(load_path)
    if args.use_bf16:
        policy = policy.to(torch.bfloat16)
    policy = policy.to(args.device).eval()

    preprocessor_overrides = {"device_processor": {"device": args.device}}
    postprocessor_overrides = {"device_processor": {"device": args.device}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=load_path,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )

    camera_map = _parse_camera_map(args.camera_keys)
    policy_server = PolicyServer(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        camera_map=camera_map,
        actions_per_chunk=args.actions_per_chunk,
        debug_dir=args.debug_dir,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy_server,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "lerobot", "policy_type": policy_type},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="Local path or HF Hub repo id of the pretrained LeRobot policy.")
    p.add_argument("--policy_type", type=str, default=None,
                   help="One of: act, smolvla, diffusion, tdmpc, vqbet, pi0, pi05, groot, xvla, wall_x. "
                        "Auto-detected from config.json if local path.")
    p.add_argument("--camera_keys", type=str, default="cam_high,cam_left_wrist,cam_right_wrist",
                   help="CSV of camera keys. Each entry is either `name` (client and lerobot share name) "
                        "or `client_key:lerobot_key` (remap). Lerobot full key = `observation.images.<lerobot_key>`. "
                        "Example: `cam_high:primary,cam_left_wrist:secondary,cam_right_wrist:wrist`.")
    p.add_argument("--actions_per_chunk", type=int, default=None,
                   help="Slice the action chunk to this length. Default = full chunk.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--idle_timeout", type=int, default=-1,
                   help="Idle timeout in seconds, -1 means never close.")
    p.add_argument("--debug_dir", type=str, default=None,
                   help="Directory to save per-call debug artefacts. None disables.")
    p.add_argument("--no_strip_unknown", action="store_true",
                   help="Disable auto-stripping of unknown fields in config.json. "
                        "Default: drop fields not declared on the policy's config dataclass.")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
