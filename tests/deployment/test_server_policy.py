from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("websockets", reason="websockets is required (see deployment/DEPLOYMENT.md)")
pytest.importorskip("msgpack", reason="msgpack is required (see deployment/DEPLOYMENT.md)")

from deployment.model_server.server_policy import PolicyServer
from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc import RTCConfig
from lerobot.utils.constants import OBS_STATE


class _FakePolicy(torch.nn.Module):
    def __init__(self, rtc_config: RTCConfig | None) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(rtc_config=rtc_config)
        self.calls = []
        self.reset_count = 0

    def predict_action_chunk(self, observation, **kwargs):
        call_number = len(self.calls)
        self.calls.append(kwargs)
        timesteps = torch.arange(50, dtype=torch.float32, device=self.anchor.device)
        return timesteps.view(1, 50, 1).expand(1, 50, 16) + call_number * 100

    def reset(self) -> None:
        self.reset_count += 1


def _identity(value):
    return value


class _ColumnPolicy(torch.nn.Module):
    """Returns a chunk whose value at (t, d) is t + 100 * d, so every column is distinguishable."""

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(rtc_config=None)
        self.action_dim = action_dim
        self.states = []

    def predict_action_chunk(self, observation, **kwargs):
        self.states.append(observation[OBS_STATE].clone())
        steps = torch.arange(50, dtype=torch.float32).view(1, 50, 1)
        columns = torch.arange(self.action_dim, dtype=torch.float32).view(1, 1, -1)
        return steps + 100 * columns


def _client_example() -> dict:
    return {
        "images": {
            key: np.zeros((3, 480, 640), dtype=np.uint8)
            for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        },
        "state": np.arange(1000, 1014, dtype=np.float32),
        "prompt": "task",
    }


def _model_columns(columns: range) -> np.ndarray:
    return np.arange(50, dtype=np.float32)[:, None] + 100 * np.array(columns, dtype=np.float32)


def test_right_arm_7d_returns_14_joints_with_held_left_arm() -> None:
    # The aloha-webapp client rejects replies narrower than its 14 joints and
    # reads the right arm from columns 7:14.
    policy = _ColumnPolicy(action_dim=7)
    server = PolicyServer(policy, _identity, _identity, right_arm_7d=True)
    server._print_actions = lambda actions, call_idx: None

    actions = server.predict_action(_client_example())["actions"]

    torch.testing.assert_close(policy.states[0], torch.arange(1007, 1014, dtype=torch.float32).view(1, 7))
    assert actions.shape == (50, 14)
    np.testing.assert_array_equal(actions[:, :7], np.tile(np.arange(1000, 1007, dtype=np.float32), (50, 1)))
    np.testing.assert_array_equal(actions[:, 7:], _model_columns(range(0, 7)))


def test_default_mode_returns_held_left_arm_and_predicted_right_arm() -> None:
    policy = _ColumnPolicy(action_dim=16)
    server = PolicyServer(policy, _identity, _identity)
    server._print_actions = lambda actions, call_idx: None

    actions = server.predict_action(_client_example())["actions"]

    assert policy.states[0].shape == (1, 19)
    assert actions.shape == (50, 14)
    np.testing.assert_array_equal(actions[:, :7], np.tile(np.arange(1000, 1007, dtype=np.float32), (50, 1)))
    np.testing.assert_array_equal(actions[:, 7:], _model_columns(range(7, 14)))


# ---------------------------------------------------------------------------
# RTC protocol v2
# ---------------------------------------------------------------------------

_LEFT_ARM_HOLD = np.zeros(7, dtype=np.float32)


def _rtc_server(postprocessor=_identity) -> tuple[_FakePolicy, PolicyServer]:
    rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=50,
        max_guidance_weight=10.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    )
    policy = _FakePolicy(rtc_config)
    server = PolicyServer(policy, _identity, postprocessor)
    server._print_actions = lambda actions, call_idx: None
    return policy, server


def _rtc_step(server: PolicyServer, query_step, inference_delay) -> dict:
    """Run one pipeline call and return the RTC metadata it reports."""
    server._run_pipeline(
        {OBS_STATE: torch.zeros((1, 19))},
        _LEFT_ARM_HOLD,
        rtc_query_step=query_step,
        rtc_inference_delay=inference_delay,
    )
    return server._last_rtc_metadata


def _chunk_values(call_number: int, start: int) -> torch.Tensor:
    """Column 0 of _FakePolicy's chunk for one call, from timestep ``start`` on."""
    return torch.arange(start, 50, dtype=torch.float32) + call_number * 100


def test_rtc_guides_with_previous_normalized_chunk_aligned_to_query_step() -> None:
    # The postprocessor shifts every action, so a leftover taken after
    # unnormalization would show the shift.
    policy, server = _rtc_server(postprocessor=lambda action: action + 1000)

    first = _rtc_step(server, query_step=0, inference_delay=0)
    assert policy.calls[0] == {}
    assert first == {
        "rtc_query_step": 0,
        "rtc_guidance_applied": False,
        "rtc_query_delta": None,
        "rtc_overlap_horizon": None,
        "rtc_inference_delay": 0,
        "rtc_skip_reason": "no_previous_chunk",
    }

    second = _rtc_step(server, query_step=7, inference_delay=6)
    assert set(policy.calls[1]) == {"inference_delay", "prev_chunk_left_over", "execution_horizon"}
    assert policy.calls[1]["inference_delay"] == 6
    assert policy.calls[1]["execution_horizon"] == 43
    assert policy.calls[1]["prev_chunk_left_over"].shape == (1, 43, 16)
    assert torch.equal(policy.calls[1]["prev_chunk_left_over"][0, :, 0], _chunk_values(0, 7))
    assert second == {
        "rtc_query_step": 7,
        "rtc_guidance_applied": True,
        "rtc_query_delta": 7,
        "rtc_overlap_horizon": 43,
        "rtc_inference_delay": 6,
        "rtc_skip_reason": None,
    }

    # The guided chunk becomes the new baseline, anchored at its own query step.
    _rtc_step(server, query_step=20, inference_delay=3)
    assert policy.calls[2]["inference_delay"] == 3
    assert policy.calls[2]["execution_horizon"] == 37
    assert policy.calls[2]["prev_chunk_left_over"].shape == (1, 37, 16)
    assert torch.equal(policy.calls[2]["prev_chunk_left_over"][0, :, 0], _chunk_values(1, 13))

    server.reset()
    after_reset = _rtc_step(server, query_step=30, inference_delay=3)
    assert policy.calls[3] == {}
    assert after_reset["rtc_skip_reason"] == "no_previous_chunk"


@pytest.mark.parametrize(
    ("query_step", "inference_delay", "skip_reason", "query_delta", "overlap_horizon"),
    [
        (None, 4, "missing_rtc_query_step", None, None),
        (-1, 4, "invalid_rtc_query_step", None, None),
        (True, 4, "invalid_rtc_query_step", None, None),
        (15.0, 4, "invalid_rtc_query_step", None, None),
        (5, 4, "non_increasing_query_step", -5, None),
        (10, 4, "non_increasing_query_step", 0, None),
        (60, 4, "no_remaining_overlap", 50, 0),
        (75, 4, "no_remaining_overlap", 65, 0),
        (15, None, "missing_rtc_inference_delay", 5, 45),
        (15, -1, "invalid_rtc_inference_delay", 5, 45),
        (15, 2.0, "invalid_rtc_inference_delay", 5, 45),
        (55, 5, "inference_delay_exhausts_overlap", 45, 5),
        (55, 9, "inference_delay_exhausts_overlap", 45, 5),
    ],
)
def test_rtc_skips_guidance_with_reason(
    query_step, inference_delay, skip_reason, query_delta, overlap_horizon
) -> None:
    policy, server = _rtc_server()
    _rtc_step(server, query_step=10, inference_delay=0)

    metadata = _rtc_step(server, query_step=query_step, inference_delay=inference_delay)

    assert policy.calls[1] == {}
    assert metadata["rtc_guidance_applied"] is False
    assert metadata["rtc_skip_reason"] == skip_reason
    assert metadata["rtc_query_delta"] == query_delta
    assert metadata["rtc_overlap_horizon"] == overlap_horizon


def test_rtc_guides_when_delay_leaves_one_overlap_step() -> None:
    policy, server = _rtc_server()
    _rtc_step(server, query_step=10, inference_delay=0)

    metadata = _rtc_step(server, query_step=55, inference_delay=4)

    assert metadata["rtc_guidance_applied"] is True
    assert policy.calls[1]["execution_horizon"] == 5
    assert torch.equal(policy.calls[1]["prev_chunk_left_over"][0, :, 0], _chunk_values(0, 45))


@pytest.mark.parametrize(
    ("skip_query_step", "skip_delay", "next_query_step", "baseline_call", "leftover_start"),
    [
        # An out-of-order response is not part of the client's stream: keep the old baseline.
        pytest.param(5, 2, 20, 0, 10, id="stale_query_keeps_baseline"),
        # Same query step: the newer chunk replaces the baseline.
        pytest.param(10, 2, 20, 1, 10, id="repeated_query_replaces_baseline"),
        # Skipped for a reason that still leaves a valid, newer chunk: it becomes the baseline.
        pytest.param(15, None, 20, 1, 5, id="missing_delay_replaces_baseline"),
        pytest.param(55, 5, 60, 1, 5, id="exhausted_overlap_replaces_baseline"),
        pytest.param(60, 2, 70, 1, 10, id="no_overlap_replaces_baseline"),
    ],
)
def test_rtc_baseline_after_skipped_guidance(
    skip_query_step, skip_delay, next_query_step, baseline_call, leftover_start
) -> None:
    policy, server = _rtc_server()
    _rtc_step(server, query_step=10, inference_delay=0)
    _rtc_step(server, query_step=skip_query_step, inference_delay=skip_delay)

    metadata = _rtc_step(server, query_step=next_query_step, inference_delay=2)

    assert metadata["rtc_guidance_applied"] is True
    assert torch.equal(
        policy.calls[2]["prev_chunk_left_over"][0, :, 0], _chunk_values(baseline_call, leftover_start)
    )
    assert policy.calls[2]["execution_horizon"] == 50 - leftover_start


def test_rtc_untagged_request_clears_baseline() -> None:
    policy, server = _rtc_server()
    _rtc_step(server, query_step=10, inference_delay=0)
    _rtc_step(server, query_step=None, inference_delay=2)

    metadata = _rtc_step(server, query_step=20, inference_delay=2)

    assert policy.calls[2] == {}
    assert metadata["rtc_skip_reason"] == "no_previous_chunk"


def test_rtc_kwargs_and_metadata_are_omitted_when_disabled() -> None:
    policy = _FakePolicy(rtc_config=None)
    server = PolicyServer(policy, _identity, _identity)
    server._print_actions = lambda actions, call_idx: None

    first = server.predict_action({**_client_example(), "rtc_query_step": 0, "rtc_inference_delay": 0})
    second = server.predict_action({**_client_example(), "rtc_query_step": 7, "rtc_inference_delay": 2})

    assert policy.calls == [{}, {}]
    assert set(first) == {"actions"}
    assert set(second) == {"actions"}


def test_rtc_response_carries_query_metadata() -> None:
    _, server = _rtc_server()

    server.predict_action({**_client_example(), "rtc_query_step": 0, "rtc_inference_delay": 0})
    response = server.predict_action({**_client_example(), "rtc_query_step": 7, "rtc_inference_delay": 2})

    assert response["actions"].shape == (50, 14)
    assert response["rtc_query_step"] == 7
    assert response["rtc_guidance_applied"] is True
    assert response["rtc_skip_reason"] is None


def test_prompt_change_clears_rtc_prefix_before_inference() -> None:
    policy, server = _rtc_server()

    server.predict_action({**_client_example(), "rtc_query_step": 0, "rtc_inference_delay": 0})
    server.predict_action({**_client_example(), "rtc_query_step": 5, "rtc_inference_delay": 2})
    response = server.predict_action(
        {**_client_example(), "prompt": "new task", "rtc_query_step": 10, "rtc_inference_delay": 2}
    )

    assert policy.calls[1]["prev_chunk_left_over"] is not None
    assert policy.calls[2] == {}
    assert response["rtc_skip_reason"] == "no_previous_chunk"
    assert policy.reset_count == 1


def test_rtc_reset_flag_clears_prefix_before_inference() -> None:
    policy, server = _rtc_server()

    server.predict_action({**_client_example(), "rtc_query_step": 0, "rtc_inference_delay": 0})
    response = server.predict_action(
        {**_client_example(), "rtc_query_step": 5, "rtc_inference_delay": 2, "rtc_reset": True}
    )

    assert policy.calls[1] == {}
    assert response["rtc_skip_reason"] == "no_previous_chunk"
    assert policy.reset_count == 1


def test_rtc_reset_flag_must_be_boolean() -> None:
    _, server = _rtc_server()

    with pytest.raises(ValueError, match="rtc_reset must be a boolean"):
        server.predict_action({**_client_example(), "rtc_query_step": 0, "rtc_reset": "yes"})


def test_legacy_rtc_steps_executed_is_ignored(caplog: pytest.LogCaptureFixture) -> None:
    policy, server = _rtc_server()
    server.predict_action({**_client_example(), "rtc_query_step": 0, "rtc_inference_delay": 0})

    with caplog.at_level("WARNING"):
        response = server.predict_action({**_client_example(), "rtc_steps_executed": 7})

    assert policy.calls[1] == {}
    assert response["rtc_skip_reason"] == "missing_rtc_query_step"
    assert "rtc_steps_executed" in caplog.text
