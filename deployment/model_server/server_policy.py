

import argparse
import contextlib
import json
import logging
import queue
import socket
import threading
import time
from datetime import UTC, datetime
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from PIL import Image as _PIL_Image

from deployment.model_server.contract import (
    CheckpointContract,
    infer_contract,
    validate_universal_expectations,
)
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from lerobot.configs import RTCAttentionSchedule
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.mot_falcon.transformers_falcon_shim import (
    TYPES_NEEDING_TRANSFORMERS_FALCON,
    masquerade_transformers_falcon,
)
from lerobot.policies.rtc import RTCConfig
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

# ANSI colour helpers
_G = "\033[32m"
_Y = "\033[33m"
_C = "\033[36m"
_R = "\033[0m"

_RTC_PROTOCOL_VERSION = 2
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_DEBUG_ROOT = _REPOSITORY_ROOT / "logs" / "pi05_debug"


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------


class PolicyServer:
    """
    Generic adapter between the Trossen OpenPI client and any LeRobot policy checkpoint, driven
    by a :class:`~deployment.model_server.contract.CheckpointContract` inferred from the loaded
    checkpoint's own config (see ``deployment/model_server/contract.py``).

    Responsibilities:
      1. Observation adaptation  — translates the openpi-client format
                                   (`{"images": {...}, "state": ndarray, "prompt": str}`)
                                   to the LeRobot policy format
                                   (`observation.images.<cam>`, `observation.state`, `task`),
                                   per the contract's camera map and state layout.
      2. Pre/Post processing     — runs the policy's preprocessor pipeline
                                   (tokenisation, normalisation, device move),
                                   then `policy.predict_action_chunk`,
                                   then the postprocessor pipeline (unnormalisation).
      3. Action post-processing  — reassembles the client's fixed 14-dim action from the
                                   model's output and the held raw client state, per the
                                   contract's action layout, and returns `{"actions": (T, D)}`.
      4. Coloured action logging — prints every predicted chunk to stdout.
      5. Debug I/O saving        — optional (pass debug_dir=None to skip).
                                   Saved per call under
                                   <debug_dir>/run_<timestamp>/call_<N>/:
                                     instruction.txt
                                     image_<camera>.png
                                     proprio.npy
                                     input_state.npy
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
        rtc_inference_delay: int = 0,
        *,
        contract: CheckpointContract,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._contract = contract
        self._camera_map = contract.camera_map
        self._image_shapes = contract.image_shapes
        self._tolerated_extra_cameras = set(contract.tolerated_extra_client_cameras)
        self._client_state_dim = contract.client_state_dim
        self._client_action_dim = contract.client_action_dim
        self._checkpoint_state_dim = contract.checkpoint_state_dim
        self._checkpoint_action_dim = contract.checkpoint_action_dim
        self._checkpoint_action_horizon = len(policy.config.action_delta_indices)
        self._device = next(policy.parameters()).device
        self._debug = debug_dir is not None
        self._call_idx = 0
        self._debug_idx = 0
        self._rtc_config = getattr(policy.config, "rtc_config", None)
        self._previous_action_chunk: torch.Tensor | None = None
        self._previous_query_step: int | None = None
        self._last_rtc_metadata: dict = {}
        self._last_prompt: str | None = None

        if self._rtc_config is not None and self._rtc_config.enabled:
            if rtc_inference_delay < 0:
                raise ValueError("rtc_inference_delay must be non-negative")
            logging.info(
                "RTC enabled: protocol=v%d, dynamic query-step alignment, "
                "max_guidance_weight=%.2f, prefix_attention_schedule=%s",
                _RTC_PROTOCOL_VERSION,
                self._rtc_config.max_guidance_weight,
                self._rtc_config.prefix_attention_schedule.value,
            )
            if rtc_inference_delay:
                logging.info(
                    "Configured rtc_inference_delay=%d is a legacy compatibility hint only; "
                    "protocol v%d requires rtc_inference_delay on every request that can use guidance",
                    rtc_inference_delay,
                    _RTC_PROTOCOL_VERSION,
                )

        self._model_image_size = (224, 224)
        logging.info(
            "PolicyServer: device=%s, policy_type=%s, camera_map=%s, client_color_space=bgr, "
            "checkpoint_state_dim=%d, checkpoint_action_dim=%d, action_horizon=%d",
            self._device,
            contract.policy_type,
            self._camera_map,
            self._checkpoint_state_dim,
            self._checkpoint_action_dim,
            self._checkpoint_action_horizon,
        )
        logging.info(
            "Image path: Trossen BGR -> RGB CHW float32 [0, 1], with the square resize "
            "corrected to the checkpoint's 4:3 letterboxed geometry."
        )

        if self._debug:
            debug_root = Path(debug_dir).expanduser()
            run_name = datetime.now(UTC).strftime("run_%Y%m%dT%H%M%S_%fZ")
            self._debug_dir = debug_root / run_name
            try:
                self._debug_dir.mkdir(parents=True, exist_ok=False)
            except OSError as exc:
                raise OSError(
                    f"Could not create a debug run directory under {debug_root}: {exc}. "
                    f"Use --debug_dir {_REPOSITORY_DEBUG_ROOT}"
                ) from exc
            # Disk writes (PNGs, JSON, .npy) are queued to a background thread
            # instead of running inline in predict_action(). With RTC enabled,
            # the client times the whole request/response round trip to
            # forecast rtc_inference_delay; synchronous debug I/O on this path
            # inflates and destabilizes that measurement and, in turn, how
            # often the server can apply RTC guidance at all.
            self._debug_queue: queue.Queue = queue.Queue()
            self._debug_writer_thread = threading.Thread(
                target=self._debug_writer_loop, daemon=True, name="debug-writer"
            )
            self._debug_writer_thread.start()
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
        """Undo the Trossen client's 4:3-to-square resize for the checkpoint's native geometry.

        The client has already stretched a 640x480 frame to 224x224. Resizing
        that square content to 224x168 and padding it recreates the geometry
        that the checkpoint's native resize-with-padding would have produced from
        the original frame. This is only applied to square wire images whose saved
        feature shape is non-square.
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
        restored = torch.nn.functional.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1), value=0.0)
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

    def _adapt_example(self, ex: dict) -> tuple[dict, np.ndarray]:
        """openpi client format → (lerobot Observation dict, raw 14-dim client state).

        The raw client state is returned in full (not pre-sliced) so that any
        ``held_client_slice`` segment in the contract's ``checkpoint_action_layout``
        can pull whatever sub-range it declares when reconstructing the client
        action -- e.g. in right-arm-only contracts ``observation[OBS_STATE]`` is
        only the 7-dim right arm and has no left-arm slot to read back out of.
        """
        observation: dict = {}

        images_dict = ex.get("images", {}) or {}
        missing_cameras = set(self._camera_map) - set(images_dict)
        extra_cameras = set(images_dict) - set(self._camera_map) - self._tolerated_extra_cameras
        if missing_cameras:
            raise ValueError(
                f"checkpoint requires camera(s) {sorted(self._camera_map)}; "
                f"missing {sorted(missing_cameras)}"
            )
        if extra_cameras:
            raise ValueError(f"checkpoint does not accept extra cameras: {sorted(extra_cameras)}")
        for client_cam, checkpoint_cam in self._camera_map.items():
            feature_key = f"{OBS_IMAGES}.{checkpoint_cam}"
            observation[feature_key] = self._image_to_tensor(feature_key, images_dict[client_cam])

        state = ex.get("state")
        if state is None:
            raise ValueError("checkpoint requires a state vector")
        s = torch.as_tensor(np.array(state, copy=True), dtype=torch.float32)
        if s.ndim == 1:
            s = s.unsqueeze(0)  # (D,) → (1, D)
        if s.ndim != 2 or s.shape[0] != 1:
            raise ValueError(f"server requires one state vector, got {tuple(s.shape)}")
        if not torch.isfinite(s).all():
            raise ValueError("arm state contains NaN or infinite values")
        if s.shape[1] != self._client_state_dim:
            raise ValueError(f"OpenPI must send {self._client_state_dim} arm-state values; got {s.shape[1]}")

        # Preserve the currently measured full client state before any slicing/padding below.
        # This is what any `held_client_slice` action-layout segment reads from later, not
        # `observation[OBS_STATE]` -- which may already be sliced/padded to a different width.
        raw_client_state = s[0].detach().cpu().to(torch.float32).numpy().copy()

        segments = []
        for seg in self._contract.checkpoint_state_layout:
            if seg.source == "client_slice":
                segments.append(s[:, seg.start : seg.start + seg.length])
            else:  # "zeros"
                segments.append(s.new_zeros((s.shape[0], seg.length)))
        model_state = torch.cat(segments, dim=1)
        if model_state.shape[1] != self._checkpoint_state_dim:
            raise RuntimeError(f"Internal state mapping produced {tuple(model_state.shape)}")
        observation[OBS_STATE] = model_state

        lang = ex.get("prompt") or ex.get("lang") or ex.get("task") or ""
        observation["task"] = lang
        return observation, raw_client_state

    # ------------------------------------------------------------------
    # 2. Debug helpers
    # ------------------------------------------------------------------

    def _debug_writer_loop(self) -> None:
        """Consume queued (path, write_fn) jobs off the request path.

        Every job here operates on plain numpy/str data already detached
        from any live tensor, so it is safe to run whenever this thread gets
        to it — nothing else mutates that data afterward.
        """
        while True:
            job = self._debug_queue.get()
            if job is None:
                return
            try:
                job()
            except Exception:
                logging.exception("PolicyServer: debug write failed")

    def _save_inputs(self, call_dir: Path, ex: dict, observation: dict) -> None:
        """Compute the exact RGB images/state passed to the LeRobot preprocessor,
        then queue the disk writes so this stays off the synchronous inference path."""
        lang = ex.get("prompt") or ex.get("lang") or ex.get("task")
        if lang:
            text = str(lang)
            self._debug_queue.put(lambda p=call_dir / "instruction.txt", t=text: p.write_text(t))

        manifest = {
            "call_idx": self._call_idx,
            "policy_type": self._contract.policy_type,
            "image_convention": (
                "RGB CHW float32 [0,1], with geometry equivalent to the checkpoint's "
                "native resize-with-padding"
            ),
            "client_color_space": "bgr",
            "images": {},
        }
        proprio = np.asarray(ex["state"]).copy()
        self._debug_queue.put(lambda p=call_dir / "proprio.npy", a=proprio: np.save(p, a))
        manifest["proprio"] = {
            "file": "proprio.npy",
            "shape": list(proprio.shape),
            "dtype": str(proprio.dtype),
            "values": proprio.tolist(),
        }
        model_state = observation[OBS_STATE].detach().cpu().to(torch.float32).numpy()
        self._debug_queue.put(lambda p=call_dir / "input_state.npy", a=model_state: np.save(p, a))
        manifest["model_state"] = {
            "file": "input_state.npy",
            "shape": list(model_state.shape),
            "dtype": str(model_state.dtype),
        }
        offset = 0
        zero_segments = []
        for seg in self._contract.checkpoint_state_layout:
            if seg.source == "zeros":
                zero_segments.append(
                    {
                        "start": offset,
                        "length": seg.length,
                        "values": model_state[0, offset : offset + seg.length].tolist(),
                    }
                )
            offset += seg.length
        if zero_segments:
            manifest["model_state"]["zero_padded_segments"] = zero_segments
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
            self._debug_queue.put(
                lambda p=call_dir / filename, a=rgb: _PIL_Image.fromarray(a, mode="RGB").save(p)
            )
            manifest["images"][feature_key] = {
                "file": filename,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(value.min().item()),
                "max": float(value.max().item()),
            }
        manifest_text = json.dumps(manifest, indent=2)
        self._debug_queue.put(lambda p=call_dir / "input_debug.json", t=manifest_text: p.write_text(t))

    def _save_outputs(self, call_dir: Path, actions: np.ndarray, rtc_metadata: dict | None = None) -> None:
        debug = {"meta": {"call_idx": self._call_idx}, "actions": actions.tolist()}
        if rtc_metadata:
            debug["meta"].update(rtc_metadata)
        debug_text = json.dumps(debug, indent=2)
        self._debug_queue.put(lambda p=call_dir / "output_actions.npy", a=actions: np.save(p, a))
        self._debug_queue.put(lambda p=call_dir / "output_debug.json", t=debug_text: p.write_text(t))

    # ------------------------------------------------------------------
    # 3. Inference pipeline
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_pipeline(
        self,
        observation: dict,
        raw_client_state: np.ndarray,
        *,
        rtc_query_step=None,
        rtc_inference_delay=None,
    ) -> np.ndarray:
        """preprocessor → policy.predict_action_chunk → postprocessor → (T, D) ndarray.

        ``raw_client_state`` is the raw 14-dim client request state (see
        ``_adapt_example``), not derived from ``observation`` here -- depending on
        the contract, ``observation[OBS_STATE]`` may already be sliced/padded and
        have no slot to read a held segment back out of.
        """
        observation = self._preprocessor(observation)

        predict_kwargs = {}
        if self._rtc_config is not None and self._rtc_config.enabled:
            query_step, query_step_error = self._parse_rtc_step(rtc_query_step, "rtc_query_step")
            inference_delay, inference_delay_error = self._parse_rtc_step(
                rtc_inference_delay, "rtc_inference_delay"
            )
            rtc_metadata = {
                "rtc_query_step": query_step,
                "rtc_guidance_applied": False,
                "rtc_query_delta": None,
                "rtc_overlap_horizon": None,
                "rtc_inference_delay": inference_delay,
                "rtc_skip_reason": None,
            }

            if query_step_error is not None:
                rtc_metadata["rtc_skip_reason"] = query_step_error
            elif self._previous_action_chunk is None or self._previous_query_step is None:
                rtc_metadata["rtc_skip_reason"] = "no_previous_chunk"
            else:
                query_delta = query_step - self._previous_query_step
                rtc_metadata["rtc_query_delta"] = query_delta
                chunk_horizon = self._previous_action_chunk.shape[1]

                if query_delta <= 0:
                    rtc_metadata["rtc_skip_reason"] = "non_increasing_query_step"
                elif query_delta >= chunk_horizon:
                    rtc_metadata["rtc_overlap_horizon"] = max(chunk_horizon - query_delta, 0)
                    rtc_metadata["rtc_skip_reason"] = "no_remaining_overlap"
                else:
                    overlap_horizon = chunk_horizon - query_delta
                    rtc_metadata["rtc_overlap_horizon"] = overlap_horizon
                    if inference_delay_error is not None:
                        rtc_metadata["rtc_skip_reason"] = inference_delay_error
                    elif inference_delay >= overlap_horizon:
                        rtc_metadata["rtc_skip_reason"] = "inference_delay_exhausts_overlap"
                    else:
                        # The preceding chunk was generated for _previous_query_step.
                        # Align it to this observation's query step before guiding the
                        # new chunk. The prefix endpoint is the remaining overlap H-s,
                        # not a static execution-horizon truncation.
                        predict_kwargs = {
                            "inference_delay": inference_delay,
                            "prev_chunk_left_over": self._previous_action_chunk[:, query_delta:],
                            "execution_horizon": overlap_horizon,
                        }
                        rtc_metadata["rtc_guidance_applied"] = True

            self._last_rtc_metadata = rtc_metadata

        chunk = self._policy.predict_action_chunk(observation, **predict_kwargs)  # (B, T, D) or (B, D)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # → (B, 1, D)
        if chunk.shape != (1, self._checkpoint_action_horizon, self._checkpoint_action_dim):
            raise ValueError(
                f"checkpoint must return shape (1, {self._checkpoint_action_horizon}, "
                f"{self._checkpoint_action_dim}); got {tuple(chunk.shape)}"
            )
        if self._rtc_config is not None and self._rtc_config.enabled:
            # Keep normalized model actions: RTC guidance runs before the
            # action postprocessor/unnormalizer. An out-of-order response is
            # not part of the committed client stream, so it must not replace
            # the established baseline used by the next valid request.
            query_step = self._last_rtc_metadata["rtc_query_step"]
            skip_reason = self._last_rtc_metadata["rtc_skip_reason"]
            if query_step is None:
                # We cannot time-align anything across an untagged request.
                self._previous_action_chunk = None
                self._previous_query_step = None
            elif (
                skip_reason != "non_increasing_query_step"
                or self._last_rtc_metadata["rtc_query_delta"] == 0
            ):
                self._previous_action_chunk = chunk.detach().clone()
                self._previous_query_step = query_step

        # postprocessor expects (B, action_dim) per call
        _, horizon, _ = chunk.shape
        processed = [self._postprocessor(chunk[:, i, :]) for i in range(horizon)]
        actions = torch.stack(processed, dim=1).squeeze(0)  # (T, D)
        actions = actions.detach().cpu().to(torch.float32).numpy()
        if not np.isfinite(actions).all():
            raise ValueError("policy returned NaN or infinite actions")

        # Reassemble the client's fixed-width action from the contract's inferred layout: each
        # segment is either a slice of the model's own output, or a client-state value held
        # constant (broadcast) across the horizon -- e.g. an unmodelled arm parked at its
        # measured pose.
        segments = []
        for seg in self._contract.checkpoint_action_layout:
            if seg.source == "model_slice":
                segments.append(actions[:, seg.start : seg.start + seg.length])
            else:  # "held_client_slice"
                held = raw_client_state[seg.start : seg.start + seg.length]
                segments.append(np.broadcast_to(held, (horizon, seg.length)).copy())
        client_actions = np.concatenate(segments, axis=1)
        if client_actions.shape != (horizon, self._client_action_dim):
            raise RuntimeError(f"Internal action mapping produced {client_actions.shape}")
        return client_actions

    @staticmethod
    def _parse_rtc_step(value, field_name: str) -> tuple[int | None, str | None]:
        """Parse non-negative integral RTC timing, returning a safe skip reason on failure."""
        if value is None:
            return None, f"missing_{field_name}"
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            return None, f"invalid_{field_name}"
        return int(value), None

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
            raise ValueError("server requires exactly one observation dictionary per request")

        ex = examples[0]  # realtime — single obs

        rtc_enabled = self._rtc_config is not None and self._rtc_config.enabled
        if rtc_enabled:
            rtc_reset = kwargs.get("rtc_reset", ex.get("rtc_reset", False))
            if not isinstance(rtc_reset, (bool, np.bool_)):
                raise ValueError("rtc_reset must be a boolean")
            if rtc_reset:
                logging.info("RTC reset requested by client; clearing policy and session state")
                self.reset()

        # --- 1. Adapt observation ---
        observation, raw_client_state = self._adapt_example(ex)

        if rtc_enabled:
            prompt = observation["task"]
            if self._last_prompt is not None and prompt != self._last_prompt:
                logging.info(
                    "Prompt changed from %r to %r; clearing policy and RTC inference state",
                    self._last_prompt,
                    prompt,
                )
                self.reset()
            self._last_prompt = prompt

        # --- 2. (optional) save inputs ---
        if self._debug:
            debug_idx = self._debug_idx
            self._debug_idx += 1
            call_dir = self._debug_dir / f"call_{debug_idx:05d}"
            call_dir.mkdir(parents=True, exist_ok=False)
            self._save_inputs(call_dir, ex, observation)

        # --- 3. Run pipeline ---
        rtc_query_step = kwargs.get("rtc_query_step", ex.get("rtc_query_step"))
        rtc_inference_delay = kwargs.get("rtc_inference_delay", ex.get("rtc_inference_delay"))
        if (
            self._rtc_config is not None
            and self._rtc_config.enabled
            and rtc_query_step is None
            and ("rtc_steps_executed" in kwargs or "rtc_steps_executed" in ex)
        ):
            logging.warning(
                "Ignoring unsafe legacy rtc_steps_executed metadata; protocol v%d requires "
                "an absolute rtc_query_step and a dynamic rtc_inference_delay",
                _RTC_PROTOCOL_VERSION,
            )
        actions = self._run_pipeline(
            observation,
            raw_client_state,
            rtc_query_step=rtc_query_step,
            rtc_inference_delay=rtc_inference_delay,
        )
        self._print_actions(actions, self._call_idx)

        # --- 4. (optional) save outputs ---
        if self._debug:
            self._save_outputs(call_dir, actions, self._last_rtc_metadata)

        self._call_idx += 1
        return {"actions": actions, **self._last_rtc_metadata}

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()
        self._previous_action_chunk = None
        self._previous_query_step = None
        self._last_rtc_metadata = {}
        self._last_prompt = None
        self._call_idx = 0


# ---------------------------------------------------------------------------
# Checkpoint validation and entry point
# ---------------------------------------------------------------------------


def _validate_checkpoint_files(checkpoint: Path) -> None:
    required = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }
    missing = sorted(name for name in required if not (checkpoint / name).is_file())
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing required deployment files: {missing}")

    # Processor state filenames encode a step index (e.g. "..._step_3_normalizer_processor...")
    # that shifts with the pipeline's own step count/order -- a checkpoint with no relative/delta
    # actions step, for instance, has its normalizer at a lower index than one that has it. Derive
    # exactly which state files this checkpoint needs from its own saved pipeline configs instead
    # of hardcoding one checkpoint's specific layout.
    for pipeline_file in ("policy_preprocessor.json", "policy_postprocessor.json"):
        pipeline_config = json.loads((checkpoint / pipeline_file).read_text())
        for step in pipeline_config.get("steps", []):
            state_file = step.get("state_file")
            if state_file and not (checkpoint / state_file).is_file():
                raise FileNotFoundError(
                    f"Checkpoint is missing a deployment file referenced by {pipeline_file}: {state_file}"
                )


def _configure_rtc(config, args) -> None:
    """Make the CLI flag the sole RTC activation switch for a loaded checkpoint."""
    if not args.rtc:
        # Do not inherit an enabled RTC block serialized by a checkpoint under
        # test; absence of --rtc must preserve ordinary inference.
        config.rtc_config = None
        return

    if not 0 <= args.rtc_inference_delay <= config.chunk_size:
        raise ValueError(f"rtc_inference_delay must be between 0 and {config.chunk_size}")
    if not 1 <= args.rtc_execution_horizon <= config.chunk_size:
        raise ValueError(f"rtc_execution_horizon must be between 1 and {config.chunk_size}")
    config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=args.rtc_execution_horizon,
        max_guidance_weight=args.rtc_max_guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule(args.rtc_attention_schedule),
    )


def _warmup_compiled_policy(policy_server: PolicyServer, *, contract: CheckpointContract) -> None:
    """Run one synthetic request through the full pipeline before serving real traffic.

    With --compile, torch.compile only traces/compiles on the first call to
    sample_actions; without this, that multi-second (or longer, under
    max-autotune) cost would land on whichever robot happens to send the
    first real request instead of at startup.
    """
    images = {cam.client_key: np.zeros((3, 224, 224), dtype=np.uint8) for cam in contract.cameras}
    example = {
        "images": images,
        "state": np.zeros(contract.client_state_dim, dtype=np.float32),
        "prompt": "warmup",
    }

    was_debug = policy_server._debug  # noqa: SLF001
    policy_server._debug = False  # noqa: SLF001
    try:
        started = time.monotonic()
        policy_server.predict_action(examples=example)
        elapsed = time.monotonic() - started
        logging.info(
            "Warmup inference done in %.2fs; subsequent requests reuse the compiled graph", elapsed
        )
    finally:
        policy_server._debug = was_debug  # noqa: SLF001
        # Drop the synthetic prompt/chunk/RTC state the warmup call left behind.
        policy_server.reset()


def main(args) -> None:
    checkpoint = Path(args.ckpt_path).expanduser().resolve(strict=True)
    if not checkpoint.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a directory: {checkpoint}")
    _validate_checkpoint_files(checkpoint)

    logging.info("Loading checkpoint from %s", checkpoint)
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    config.device = args.device

    # Everything about how to serve this checkpoint -- camera mapping, state/action layout -- is
    # read straight off its own declared config, not a hand-authored profile. See
    # deployment/model_server/contract.py for the (small, fixed) rule set this applies.
    contract = infer_contract(config)
    validate_universal_expectations(config)
    logging.info(
        "Inferred contract: policy_type=%s, camera_map=%s, checkpoint_state_dim=%d, "
        "checkpoint_action_dim=%d",
        contract.policy_type,
        contract.camera_map,
        contract.checkpoint_state_dim,
        contract.checkpoint_action_dim,
    )

    if args.dry_run:
        logging.info(
            "Checkpoint at %s matches this deployment's contract; --dry_run set, exiting.", checkpoint
        )
        return

    _configure_rtc(config, args)

    # torch.compile is a construction-time setting, unrelated to how the
    # checkpoint was trained: a policy's __init__ reads config.compile_model
    # when it builds the model and wraps its inference path accordingly,
    # before from_pretrained loads any weights into it.
    config.compile_model = args.compile
    if args.compile:
        config.compile_mode = args.compile_mode

    policy_class = get_policy_class(config.type)
    # mot_falcon needs the vendored Falcon3-VL transformers fork (see
    # lerobot.policies.mot_falcon.transformers_falcon_shim); everything else constructs against
    # the real, installed transformers package as usual.
    falcon_ctx = (
        masquerade_transformers_falcon()
        if config.type in TYPES_NEEDING_TRANSFORMERS_FALCON
        else contextlib.nullcontext()
    )
    with falcon_ctx:
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
        rtc_inference_delay=args.rtc_inference_delay,
        contract=contract,
    )

    if args.compile:
        _warmup_compiled_policy(policy_server, contract=contract)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy_server,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={
            "env": contract.policy_type,
            "policy_type": config.type,
            "camera_map": contract.camera_map,
            "client_color_space": "bgr",
            "action_dim": contract.client_action_dim,
            "action_horizon": policy_server._checkpoint_action_horizon,  # noqa: SLF001
            "fps": 30,
            "rtc_enabled": args.rtc,
            "rtc_inference_delay": args.rtc_inference_delay if args.rtc else None,
            **(
                {
                    "rtc_protocol_version": _RTC_PROTOCOL_VERSION,
                    "rtc_requires_query_step": True,
                    "rtc_requires_dynamic_inference_delay": True,
                }
                if args.rtc
                else {}
            ),
        },
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    p = argparse.ArgumentParser(
        description="Serve a LeRobot policy checkpoint behind the Trossen deployment adapter."
    )
    p.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help=(
            "Local pretrained_model directory. Camera mapping and state/action layout are "
            "inferred automatically from the checkpoint's own config.json -- see "
            "deployment/model_server/contract.py."
        ),
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Load the checkpoint's config.json, infer its contract, and validate it against this "
            "deployment's fixed expectations, then exit without loading model weights or "
            "starting the server. Needs no GPU; useful when introducing a new checkpoint."
        ),
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
        help=(
            "Root directory for timestamped per-run image/action logs. "
            f"Use {_REPOSITORY_DEBUG_ROOT}; omit to disable logging."
        ),
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model's inference path (sample_actions) in torch.compile. "
        "Adds a one-time trace/compile cost, paid during startup warmup rather than "
        "on the first real request, in exchange for faster steady-state inference.",
    )
    p.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        help="torch.compile mode, only used with --compile (default: reduce-overhead, "
        "which favors low per-call latency for this server's fixed-shape, batch-size-1 "
        "requests; 'max-autotune' searches harder for a faster kernel at a much longer "
        "compile time).",
    )
    p.add_argument(
        "--rtc",
        action="store_true",
        help="Enable Real-Time Chunking for inference.",
    )
    p.add_argument(
        "--rtc_inference_delay",
        type=int,
        default=4,
        help=(
            "Legacy initial latency hint in 30 Hz control steps (default: 4). "
            "RTC protocol v2 uses the dynamic rtc_inference_delay sent with each request."
        ),
    )
    p.add_argument(
        "--rtc_execution_horizon",
        type=int,
        default=50,
        help=(
            "Legacy RTCConfig compatibility value (default: 50). Protocol v2 deployment "
            "derives every live prefix endpoint as action_horizon - query_step_delta."
        ),
    )
    p.add_argument(
        "--rtc_max_guidance_weight",
        type=float,
        default=12.0,
        help="RTC consistency guidance strength (default: 10.0).",
    )
    p.add_argument(
        "--rtc_attention_schedule",
        choices=[schedule.value for schedule in RTCAttentionSchedule],
        default=RTCAttentionSchedule.EXP.value,
        help="RTC prefix-attention blend schedule (default: EXP).",
    )
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
