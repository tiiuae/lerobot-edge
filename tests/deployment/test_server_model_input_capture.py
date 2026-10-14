"""The images saved with --debug_dir are exactly what the vision encoder ingests.

model_input_<camera>.png is written from the output of pi0.5's own
_preprocess_images, so it shows the image after every transform: the server's
wire conversion, the LeRobot preprocessor, the letterbox to 224x224 and the
[-1, 1] mapping. These tests run the real PI05Policy._preprocess_images, not a
stand-in, and check a pure-red camera frame comes out red, letterboxed, in the
right file. A channel swap anywhere on the way would turn it blue.
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("websockets", reason="websockets is required (see deployment/DEPLOYMENT.md)")
pytest.importorskip("msgpack", reason="msgpack is required (see deployment/DEPLOYMENT.md)")

from deployment.model_server.server_policy import PolicyServer
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

RED, GREEN, BLUE = 0, 1, 2
KEYS = ("observation.images.primary", "observation.images.secondary", "observation.images.wrist")


class _FakePolicy(torch.nn.Module):
    """Just enough of PI05Policy for its real _preprocess_images to run."""

    _preprocess_images = PI05Policy._preprocess_images

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            rtc_config=None,
            image_features=dict.fromkeys(KEYS),
            image_resolution=(224, 224),
            empty_cameras=0,
        )


def _server(tmp_path, client_color_space="rgb") -> PolicyServer:
    return PolicyServer(
        policy=_FakePolicy(),
        preprocessor=lambda value: value,
        postprocessor=lambda value: value,
        debug_dir=str(tmp_path),
        client_color_space=client_color_space,
    )


def _flush(server: PolicyServer) -> None:
    """Wait for the background writer: the queue is FIFO, so once this marker
    runs every earlier write has finished."""
    done = threading.Event()
    server._debug_queue.put(done.set)
    assert done.wait(10), "debug writer did not drain"


def _red_chw(height: int = 480, width: int = 640, order: str = "rgb") -> np.ndarray:
    frame = np.zeros((3, height, width), dtype=np.uint8)
    frame[RED if order == "rgb" else BLUE] = 255
    return frame


def _run(server: PolicyServer, call_dir, frames: dict) -> None:
    """Wire images -> server conversion -> the model's own preprocessing."""
    batch = {key: server._image_to_tensor(key, frame) for key, frame in frames.items()}
    server._model_input_dir = call_dir
    try:
        server._policy._preprocess_images(batch)
    finally:
        server._model_input_dir = None
    _flush(server)


def _load(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def test_native_rgb_frame_reaches_the_encoder_as_rgb(tmp_path):
    server = _server(tmp_path)
    _run(server, tmp_path, {KEYS[0]: _red_chw()})
    image = _load(tmp_path / "model_input_primary.png")
    content = image[28:196]  # 224x168 content inside the 224x224 letterbox
    assert content[..., RED].min() >= 254
    assert content[..., BLUE].max() <= 1
    assert content[..., GREEN].max() <= 1


def test_the_saved_image_shows_the_training_letterbox(tmp_path):
    server = _server(tmp_path)
    _run(server, tmp_path, {KEYS[0]: _red_chw()})
    image = _load(tmp_path / "model_input_primary.png")
    assert image.shape == (224, 224, 3)
    assert image[:28].max() <= 1  # black bar above
    assert image[196:].max() <= 1  # black bar below


def test_one_file_per_camera_named_after_the_feature(tmp_path):
    server = _server(tmp_path)
    _run(server, tmp_path, {key: _red_chw() for key in KEYS})
    for key in KEYS:
        assert (tmp_path / f"model_input_{key.rsplit('.', 1)[-1]}.png").exists()


def test_legacy_bgr_client_is_also_rgb_at_the_encoder(tmp_path):
    """Old client: squashed 224x224, BGR; server run with --client_color_space bgr."""
    server = _server(tmp_path, client_color_space="bgr")
    _run(server, tmp_path, {KEYS[0]: _red_chw(224, 224, order="bgr")})
    content = _load(tmp_path / "model_input_primary.png")[28:196]
    assert content[..., RED].min() >= 254
    assert content[..., BLUE].max() <= 1


def test_a_mismatched_setting_is_visible_in_the_saved_image(tmp_path):
    """The point of the file: if client and --client_color_space disagree, the
    PNG shows it. Here an RGB client meets a server told to expect BGR."""
    server = _server(tmp_path, client_color_space="bgr")
    _run(server, tmp_path, {KEYS[0]: _red_chw()})
    content = _load(tmp_path / "model_input_primary.png")[28:196]
    assert content[..., BLUE].min() >= 254  # red came out blue
    assert content[..., RED].max() <= 1


def test_the_model_still_gets_the_same_images(tmp_path):
    """Capturing must not change what is returned to the model."""
    frames = {KEYS[0]: _red_chw()}
    plain = _FakePolicy()
    batch = {key: _server(tmp_path / "a")._image_to_tensor(key, frame) for key, frame in frames.items()}
    expected, _ = plain._preprocess_images(batch)

    server = _server(tmp_path / "b")
    server._model_input_dir = tmp_path
    captured, _ = server._policy._preprocess_images(batch)
    server._model_input_dir = None
    for a, b in zip(expected, captured, strict=True):
        torch.testing.assert_close(a, b)


def test_nothing_is_written_outside_a_call(tmp_path):
    server = _server(tmp_path)
    batch = {KEYS[0]: server._image_to_tensor(KEYS[0], _red_chw())}
    server._policy._preprocess_images(batch)  # _model_input_dir is None
    _flush(server)
    assert not list(tmp_path.rglob("model_input_*.png"))


def test_no_capture_without_debug_dir():
    server = PolicyServer(policy=_FakePolicy(), preprocessor=lambda v: v, postprocessor=lambda v: v)
    assert server._policy._preprocess_images.__name__ == "_preprocess_images"

