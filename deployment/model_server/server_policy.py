# Copyright 2026. Licensed under the MIT License.
# AIDRC pi0.5 policy server — websocket front-end for the fixed deployment
# contract documented with the August 2026 checkpoint.

import argparse
import json
import logging
import socket
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


_CAMERA_MAP = {
    "cam_high": "primary",
    "cam_left_wrist": "secondary",
    "cam_right_wrist": "wrist",
}
_IMAGE_FEATURES = {
    "observation.images.primary": (3, 480, 640),
    "observation.images.secondary": (3, 480, 640),
    "observation.images.wrist": (3, 480, 640),
}
_POLICY_STATE_DIM = 19
_POLICY_ACTION_DIM = 16
_RIGHT_ARM_ACTION_SLICE = slice(7, 14)


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------


class PolicyServer:
    """
    Fixed adapter between the Trossen OpenPI client and the AIDRC pi0.5 policy.

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
                                     image_<camera>.png
                                     input_debug.json
                                     output_actions.npy
                                     output_debug.json

    Client observation format (each example):
        {
            "images": {"cam_high": ndarray(C,H,W) uint8, ...},
            "state":  ndarray(D,) float,
            "prompt": str,
        }
    """

    def __init__(
        self,
        policy,
        preprocessor,
        postprocessor,
        debug_dir: str | None = None,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._camera_map = dict(_CAMERA_MAP)
        self._device = next(policy.parameters()).device
        self._debug = debug_dir is not None
        self._call_idx = 0

        self._image_shapes = dict(_IMAGE_FEATURES)
        self._model_image_size = (224, 224)
        logging.info(
            "PolicyServer: device=%s, camera_map=%s, client_color_space=bgr",
            self._device,
            self._camera_map,
        )
        logging.info(
            "Image path: Trossen BGR -> RGB CHW float32, with the square resize corrected "
            "to LeRobot pi0.5's native 4:3 letterboxed geometry and padding value."
        )

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
            a = a.transpose(2, 0, 1)  # HWC → CHW
        if a.ndim != 3 or a.shape[0] not in (1, 3, 4):
            raise ValueError(f"Expected CHW or HWC image, got shape {a.shape}")
        if a.dtype != np.uint8:
            if not np.isfinite(a).all():
                raise ValueError("Image contains NaN or infinite values")
            a = (a * 255 if a.max() <= 1.0 else a).clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(a)

    def _restore_squashed_aspect(self, tensor: torch.Tensor, feature_key: str) -> torch.Tensor:
        """Undo the Trossen client's 4:3-to-square resize for the AIDRC checkpoint.

        The client has already stretched a 640x480 frame to 224x224. Resizing
        that square content to 224x168 and padding it recreates the geometry
        that pi0.5's native ``resize_with_pad_torch`` would have produced from
        the original frame. LeRobot pads float images with -1.0 before its
        model normalization, so the same value is used here. This is only
        applied to square wire images whose saved feature shape is non-square.
        """
        feature_shape = self._image_shapes[feature_key]
        _, expected_h, expected_w = feature_shape
        _, wire_h, wire_w = tensor.shape
        if wire_h != wire_w or expected_h == expected_w:
            return tensor

        output_h, output_w = self._model_image_size
        ratio = max(expected_w / output_w, expected_h / output_h)
        resized_h = int(expected_h / ratio)
        resized_w = int(expected_w / ratio)
        resized = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        pad_h0, remainder_h = divmod(output_h - resized_h, 2)
        pad_h1 = pad_h0 + remainder_h
        pad_w0, remainder_w = divmod(output_w - resized_w, 2)
        pad_w1 = pad_w0 + remainder_w
        restored = torch.nn.functional.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1), value=-1.0)
        logging.debug(
            "Restored %s aspect ratio: wire=%sx%s, training=%sx%s, model=%sx%s",
            feature_key,
            wire_h,
            wire_w,
            expected_h,
            expected_w,
            output_h,
            output_w,
        )
        return restored

    def _image_to_tensor(self, feature_key: str, arr: np.ndarray) -> torch.Tensor:
        """Convert a wire image to the RGB tensor convention used during training.

        LeRobot training uses RGB ``(C,H,W)`` float32 tensors in ``[0,1]``.
        The Trossen bridge sends BGR after squashing 640x480 to 224x224, so this
        method reverses the channels and reconstructs the final 4:3 letterbox.
        """
        chw = self._to_chw_uint8(arr)
        if chw.shape != (3, 224, 224):
            raise ValueError(
                "Trossen client images must have shape (3, 224, 224) before aspect correction; "
                f"got {chw.shape} for {feature_key}"
            )
        chw = np.ascontiguousarray(chw[::-1])  # fixed Trossen BGR -> training RGB
        t = torch.from_numpy(np.array(chw, copy=True)).to(torch.float32) / 255.0
        t = self._restore_squashed_aspect(t, feature_key)
        return t.unsqueeze(0).contiguous()

    def _adapt_example(self, ex: dict) -> dict:
        """openpi client format → lerobot Observation dict."""
        observation: dict = {}

        images_dict = ex.get("images", {}) or {}
        missing_cameras = set(self._camera_map) - set(images_dict)
        extra_cameras = set(images_dict) - set(self._camera_map)
        if missing_cameras:
            raise ValueError(
                f"AIDRC checkpoint requires all three cameras; missing {sorted(missing_cameras)}"
            )
        if extra_cameras:
            raise ValueError(f"AIDRC checkpoint does not accept extra cameras: {sorted(extra_cameras)}")
        for client_cam, lerobot_cam in self._camera_map.items():
            feature_key = f"{OBS_IMAGES}.{lerobot_cam}"
            observation[feature_key] = self._image_to_tensor(feature_key, images_dict[client_cam])

        state = ex.get("state")
        if state is None:
            raise ValueError("AIDRC checkpoint requires a state vector")
        s = torch.as_tensor(np.array(state, copy=True), dtype=torch.float32)
        if s.ndim == 1:
            s = s.unsqueeze(0)  # (D,) → (1, D)
        if s.ndim != 2 or s.shape[0] != 1:
            raise ValueError(f"AIDRC server requires one state vector, got {tuple(s.shape)}")
        if not torch.isfinite(s).all():
            raise ValueError("AIDRC arm state contains NaN or infinite values")
        observation[OBS_STATE] = s

        lang = ex.get("prompt") or ex.get("lang") or ex.get("task") or ""
        observation["task"] = lang
        return observation

    # ------------------------------------------------------------------
    # 2. Debug helpers
    # ------------------------------------------------------------------

    def _save_inputs(self, call_dir: Path, ex: dict, observation: dict) -> None:
        """Save the exact RGB images passed to the LeRobot preprocessor."""
        lang = ex.get("prompt") or ex.get("lang") or ex.get("task")
        if lang:
            (call_dir / "instruction.txt").write_text(str(lang))

        manifest = {
            "call_idx": self._call_idx,
            "image_convention": (
                "RGB CHW float32; content is [0,1] and padding is -1, matching "
                "LeRobot pi0.5's native resize_with_pad_torch before model normalization"
            ),
            "png_visualization": "Negative padding is clamped to black in the saved PNG files.",
            "client_color_space": "bgr",
            "images": {},
        }
        for feature_key, value in observation.items():
            if not feature_key.startswith(f"{OBS_IMAGES}."):
                continue
            tensor = value.detach().cpu().to(torch.float32)
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.ndim != 3 or tensor.shape[0] != 3:
                raise ValueError(f"Cannot log image {feature_key}: unexpected shape {tuple(tensor.shape)}")
            rgb = tensor.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            camera_name = feature_key.removeprefix(f"{OBS_IMAGES}.").replace("/", "_")
            filename = f"image_{camera_name}.png"
            _PIL_Image.fromarray(rgb, mode="RGB").save(call_dir / filename)
            manifest["images"][feature_key] = {
                "file": filename,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(value.min().item()),
                "max": float(value.max().item()),
            }
        (call_dir / "input_debug.json").write_text(json.dumps(manifest, indent=2))

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

        chunk = self._policy.predict_action_chunk(observation)  # (B, T, D) or (B, D)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # → (B, 1, D)
        if chunk.shape != (1, 50, _POLICY_ACTION_DIM):
            raise ValueError(f"AIDRC checkpoint must return shape (1, 50, 16); got {tuple(chunk.shape)}")

        # postprocessor expects (B, action_dim) per call
        _, horizon, _ = chunk.shape
        processed = [self._postprocessor(chunk[:, i, :]) for i in range(horizon)]
        actions = torch.stack(processed, dim=1).squeeze(0)  # (T, D)
        actions = actions.detach().cpu().to(torch.float32).numpy()
        if not np.isfinite(actions).all():
            raise ValueError("AIDRC policy returned NaN or infinite actions")
        # Only right-arm joints + gripper (documented dimensions 7:14) are
        # actionable. The client receives exactly seven values and fills the
        # parked left arm from its frozen pose.
        return actions[:, _RIGHT_ARM_ACTION_SLICE]

    # ------------------------------------------------------------------
    # 4. Coloured action print
    # ------------------------------------------------------------------

    @staticmethod
    def _print_actions(actions: np.ndarray, call_idx: int) -> None:
        horizon, action_dim = actions.shape
        print(f"{_C}[call {call_idx:05d}]{_R} {_Y}actions ({horizon}×{action_dim}){_R}")
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
        if len(examples) != 1 or not isinstance(examples[0], dict):
            raise ValueError("AIDRC server requires exactly one observation dictionary per request")

        ex = examples[0]  # realtime — single obs

        # --- 1. Adapt observation ---
        observation = self._adapt_example(ex)

        # --- 2. (optional) save inputs ---
        if self._debug:
            call_dir = self._debug_dir / f"call_{self._call_idx:05d}"
            call_dir.mkdir(parents=True, exist_ok=True)
            self._save_inputs(call_dir, ex, observation)

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
# Checkpoint validation and entry point
# ---------------------------------------------------------------------------


def _enum_value(value):
    return getattr(value, "value", value)


def _validate_checkpoint_config(config) -> None:
    """Reject anything outside the deployment guide's fixed interface."""
    errors = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    image_features = getattr(config, "image_features", {}) or {}
    actual_images = {key: tuple(feature.shape) for key, feature in image_features.items()}
    state_feature = getattr(config, "robot_state_feature", None)
    action_feature = getattr(config, "action_feature", None)
    normalization = {
        key: _enum_value(value) for key, value in (getattr(config, "normalization_mapping", {}) or {}).items()
    }

    require(getattr(config, "type", None) == "pi05", "policy type must be pi05")
    require(
        actual_images == _IMAGE_FEATURES and tuple(actual_images) == tuple(_IMAGE_FEATURES),
        f"ordered image features must be {_IMAGE_FEATURES}, got {actual_images}",
    )
    require(
        state_feature is not None and tuple(state_feature.shape) == (_POLICY_STATE_DIM,),
        "state feature must have shape (19,)",
    )
    require(
        action_feature is not None and tuple(action_feature.shape) == (_POLICY_ACTION_DIM,),
        "action feature must have shape (16,)",
    )
    require(getattr(config, "n_obs_steps", None) == 1, "n_obs_steps must be 1")
    require(getattr(config, "chunk_size", None) == 50, "chunk_size must be 50")
    require(getattr(config, "n_action_steps", None) == 50, "n_action_steps must be 50")
    require(getattr(config, "num_inference_steps", None) == 10, "num_inference_steps must be 10")
    require(getattr(config, "max_state_dim", None) == 32, "max_state_dim must be 32")
    require(getattr(config, "max_action_dim", None) == 32, "max_action_dim must be 32")
    require(tuple(getattr(config, "image_resolution", ())) == (224, 224), "image_resolution must be 224x224")
    require(getattr(config, "empty_cameras", None) == 0, "empty_cameras must be 0")
    require(getattr(config, "tokenizer_max_length", None) == 200, "tokenizer_max_length must be 200")
    require(getattr(config, "dtype", None) == "bfloat16", "dtype must be bfloat16")
    require(getattr(config, "use_relative_actions", None) is False, "actions must be absolute")
    require(
        normalization == {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"},
        "normalization must be VISUAL=IDENTITY and STATE/ACTION=QUANTILES",
    )

    if errors:
        raise ValueError("Checkpoint does not match deployment_guide.md:\n- " + "\n- ".join(errors))


def _validate_checkpoint_files(checkpoint: Path) -> None:
    required = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    }
    missing = sorted(name for name in required if not (checkpoint / name).is_file())
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing required deployment files: {missing}")


def main(args) -> None:
    checkpoint = Path(args.ckpt_path).expanduser().resolve(strict=True)
    if not checkpoint.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a directory: {checkpoint}")
    _validate_checkpoint_files(checkpoint)

    logging.info("Loading fixed AIDRC pi0.5 checkpoint from %s", checkpoint)
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    config.device = args.device
    _validate_checkpoint_config(config)

    policy_class = get_policy_class("pi05")
    policy = policy_class.from_pretrained(str(checkpoint), config=config).to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    policy_server = PolicyServer(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
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
        metadata={
            "env": "aidrc_pi05",
            "policy_type": "pi05",
            "camera_map": _CAMERA_MAP,
            "client_color_space": "bgr",
            "action_dim": 7,
            "action_horizon": 50,
            "fps": 30,
        },
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    p = argparse.ArgumentParser(description="Serve the fixed AIDRC pi0.5 deployment checkpoint.")
    p.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Local pretrained_model directory described by deployment_guide.md.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument(
        "--idle_timeout", type=int, default=-1, help="Idle timeout in seconds, -1 means never close."
    )
    p.add_argument(
        "--debug_dir",
        type=str,
        default=None,
        help="Directory to save per-call debug artefacts. None disables.",
    )
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
