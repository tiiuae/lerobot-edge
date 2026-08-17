from types import SimpleNamespace

import torch

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


def test_rtc_passes_previous_normalized_chunk_and_runtime_delay() -> None:
    rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=10,
        max_guidance_weight=10.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    )
    policy = _FakePolicy(rtc_config)
    server = PolicyServer(policy, _identity, _identity, rtc_inference_delay=4)
    observation = {OBS_STATE: torch.zeros((1, 19))}

    first_actions = server._run_pipeline(observation)
    second_actions = server._run_pipeline(
        observation,
        rtc_steps_executed=7,
        rtc_inference_delay=6,
    )

    assert first_actions.shape == (50, 14)
    assert second_actions.shape == (50, 14)
    assert policy.calls[0] == {"inference_delay": 4, "prev_chunk_left_over": None}
    assert policy.calls[1]["inference_delay"] == 6
    assert policy.calls[1]["prev_chunk_left_over"].shape == (1, 10, 16)
    assert torch.equal(
        policy.calls[1]["prev_chunk_left_over"][0, :, 0],
        torch.arange(7, 17, dtype=torch.float32),
    )

    server.reset()
    server._run_pipeline(observation)
    assert policy.calls[2]["prev_chunk_left_over"] is None


def test_rtc_kwargs_are_omitted_when_disabled() -> None:
    policy = _FakePolicy(rtc_config=None)
    server = PolicyServer(policy, _identity, _identity)

    server._run_pipeline({OBS_STATE: torch.zeros((1, 19))})

    assert policy.calls == [{}]


def test_prompt_change_clears_rtc_prefix_before_inference() -> None:
    rtc_config = RTCConfig(enabled=True, execution_horizon=10)
    policy = _FakePolicy(rtc_config)
    server = PolicyServer(policy, _identity, _identity, rtc_inference_delay=4)
    observation = {OBS_STATE: torch.zeros((1, 19))}
    examples = [
        {
            "images": {},
            "state": torch.zeros(14),
            "prompt": "old task",
        }
    ]

    # Exercise prompt-boundary behavior without requiring deployment cameras.
    server._adapt_example = lambda ex: {**observation, "task": ex["prompt"]}
    server._print_actions = lambda actions, call_idx: None
    server.predict_action(examples)
    server.predict_action(examples)

    examples[0]["prompt"] = "new task"
    server.predict_action(examples)

    assert policy.calls[1]["prev_chunk_left_over"] is not None
    assert policy.calls[2]["prev_chunk_left_over"] is None
    assert policy.reset_count == 1
