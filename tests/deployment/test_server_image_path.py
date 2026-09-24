"""The pi0.5 server's image path, for current and legacy Trossen clients.

The current openpi Trossen client sends RGB at native camera resolution,
untouched. The server must pass those frames through, so pi0.5's own
resize-with-padding produces the training geometry, and must not flip channels.

Older clients squashed each 640x480 frame to 224x224 and swapped it to BGR. The
server used to accept only that shape and flipped unconditionally, so it
rejected native frames and would have inverted RGB ones. Legacy clients stay
supported through --client_color_space bgr and the square-frame restore.

A pure-red frame is used because a channel swap turns it pure blue.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deployment.model_server.server_policy import PolicyServer, build_argparser

RED, GREEN, BLUE = 0, 1, 2
FEATURE = "observation.images.primary"


class _FakePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(rtc_config=None)


def _server(client_color_space: str = "rgb") -> PolicyServer:
    return PolicyServer(
        policy=_FakePolicy(),
        preprocessor=lambda value: value,
        postprocessor=lambda value: value,
        client_color_space=client_color_space,
    )


def _red_chw(height: int, width: int, order: str = "rgb") -> np.ndarray:
    frame = np.zeros((3, height, width), dtype=np.uint8)
    frame[RED if order == "rgb" else BLUE] = 255
    return frame


def test_native_rgb_frames_pass_through_untouched():
    """Native 640x480 RGB goes to the model as-is; pi0.5 letterboxes it itself."""
    tensor = _server()._image_to_tensor(FEATURE, _red_chw(480, 640))
    assert tensor.shape == (1, 3, 480, 640)
    assert torch.all(tensor[0, RED] == 1.0)
    assert torch.all(tensor[0, BLUE] == 0.0)


def test_native_frames_are_not_resampled():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (3, 480, 640), dtype=np.uint8)
    tensor = _server()._image_to_tensor(FEATURE, frame)
    np.testing.assert_allclose(tensor[0].numpy(), frame.astype(np.float32) / 255.0)


def test_native_frames_are_no_longer_rejected():
    """The old server raised on anything but (3, 224, 224)."""
    _server()._image_to_tensor(FEATURE, _red_chw(480, 640))


def test_hwc_native_frames_are_accepted():
    hwc = np.transpose(_red_chw(480, 640), (1, 2, 0)).copy()
    tensor = _server()._image_to_tensor(FEATURE, hwc)
    assert tensor.shape == (1, 3, 480, 640)
    assert torch.all(tensor[0, RED] == 1.0)


def test_legacy_bgr_square_client_still_gets_rgb_and_the_letterbox():
    """An old client: 224x224, BGR. With --client_color_space bgr the server
    flips once and restores the 4:3 letterbox, exactly as before."""
    tensor = _server("bgr")._image_to_tensor(FEATURE, _red_chw(224, 224, order="bgr"))
    assert tensor.shape == (1, 3, 224, 224)
    content = tensor[0, :, 28:196, :]  # 224x168 content, 28 black rows top and bottom
    assert torch.all(content[RED] > 0.99)
    assert torch.all(content[BLUE] < 0.01)
    assert torch.all(tensor[0, :, :28, :] == 0.0)
    assert torch.all(tensor[0, :, 196:, :] == 0.0)


def test_default_does_not_flip_even_for_square_frames():
    tensor = _server()._image_to_tensor(FEATURE, _red_chw(224, 224))
    content = tensor[0, :, 28:196, :]
    assert torch.all(content[RED] > 0.99)
    assert torch.all(content[BLUE] < 0.01)


def test_an_unknown_colour_space_is_refused():
    with pytest.raises(ValueError, match="client_color_space"):
        _server("yuv")


def test_the_cli_defaults_to_rgb():
    args = build_argparser().parse_args(["--ckpt_path", "unused"])
    assert args.client_color_space == "rgb"
