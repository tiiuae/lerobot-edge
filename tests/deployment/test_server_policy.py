from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deployment.model_server.contract import infer_contract, validate_universal_expectations
from deployment.model_server.server_policy import PolicyServer
from lerobot.configs import RTCAttentionSchedule
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.rtc import RTCConfig
from lerobot.utils.constants import OBS_STATE

# ---------------------------------------------------------------------------
# Synthetic configs covering every checkpoint contract seen so far
# ---------------------------------------------------------------------------


def _fake_config(*, image_features: dict, state_dim: int, action_dim: int, **overrides) -> SimpleNamespace:
    """A minimal duck-typed PreTrainedConfig-shaped object for a given camera/state/action shape."""
    fields = dict(
        type="mot_falcon",
        n_obs_steps=1,
        image_features=image_features,
        robot_state_feature=PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        action_feature=PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        num_inference_steps=10,
        max_state_dim=32,
        max_action_dim=32,
        tokenizer_max_length=200,
        dtype="bfloat16",
        empty_cameras=0,
        use_relative_actions=False,
        image_resolution=(224, 224),
        normalization_mapping={"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"},
        action_delta_indices=list(range(25)),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _image_feature(shape=(3, 480, 640)) -> PolicyFeature:
    return PolicyFeature(type=FeatureType.VISUAL, shape=shape)


# Real-checkpoint-shaped configs (identity camera names, matching mot_falcon/*_realworld pi05).
def _config_2cam_rightarm7d(**overrides) -> SimpleNamespace:
    return _fake_config(
        image_features={
            "observation.images.cam_high": _image_feature(),
            "observation.images.cam_right_wrist": _image_feature(),
        },
        state_dim=7,
        action_dim=7,
        **overrides,
    )


def _config_3cam_bimanual(**overrides) -> SimpleNamespace:
    return _fake_config(
        image_features={
            "observation.images.cam_high": _image_feature(),
            "observation.images.cam_right_wrist": _image_feature(),
            "observation.images.cam_left_wrist": _image_feature(),
        },
        state_dim=14,
        action_dim=14,
        type="pi05",
        **overrides,
    )


def _config_3cam_bimanual_base_legacy(**overrides) -> SimpleNamespace:
    return _fake_config(
        image_features={
            "observation.images.primary": _image_feature(),
            "observation.images.secondary": _image_feature(),
            "observation.images.wrist": _image_feature(),
        },
        state_dim=19,
        action_dim=16,
        type="pi05",
        **overrides,
    )


class _FakePolicy(torch.nn.Module):
    def __init__(self, rtc_config: RTCConfig | None, *, action_dim: int = 16, horizon: int = 50) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(rtc_config=rtc_config, action_delta_indices=list(range(horizon)))
        self.calls = []
        self.reset_count = 0
        self._action_dim = action_dim
        self._horizon = horizon

    def predict_action_chunk(self, observation, **kwargs):
        call_number = len(self.calls)
        self.calls.append(kwargs)
        timesteps = torch.arange(self._horizon, dtype=torch.float32, device=self.anchor.device)
        chunk = timesteps.view(1, self._horizon, 1).expand(1, self._horizon, self._action_dim)
        return chunk + call_number * 100

    def reset(self) -> None:
        self.reset_count += 1


def _identity(value):
    return value


def _make_server(policy, *, config=None, **kwargs) -> PolicyServer:
    config = config if config is not None else _config_3cam_bimanual_base_legacy()
    contract = infer_contract(config)
    return PolicyServer(policy, _identity, _identity, contract=contract, **kwargs)


# ---------------------------------------------------------------------------
# Contract inference
# ---------------------------------------------------------------------------


def test_infer_contract_right_arm_only_identity_cameras() -> None:
    contract = infer_contract(_config_2cam_rightarm7d())

    assert contract.policy_type == "mot_falcon"
    assert len(contract.cameras) == 2
    assert contract.checkpoint_state_dim == 7
    assert contract.checkpoint_action_dim == 7
    assert contract.camera_map == {"cam_high": "cam_high", "cam_right_wrist": "cam_right_wrist"}
    assert contract.tolerated_extra_client_cameras == ["cam_left_wrist"]
    assert contract.client_state_dim == 14
    assert contract.client_action_dim == 14


def test_infer_contract_full_bimanual_no_padding() -> None:
    contract = infer_contract(_config_3cam_bimanual())

    assert contract.checkpoint_state_dim == 14
    assert contract.checkpoint_action_dim == 14
    assert contract.tolerated_extra_client_cameras == []
    # Nothing held, nothing padded: a single client_slice(0,14) / model_slice(0,14) segment each.
    assert contract.checkpoint_state_layout == [
        type(contract.checkpoint_state_layout[0])("client_slice", start=0, length=14)
    ]


def test_infer_contract_bimanual_base_legacy_camera_names() -> None:
    contract = infer_contract(_config_3cam_bimanual_base_legacy())

    assert contract.checkpoint_state_dim == 19
    assert contract.checkpoint_action_dim == 16
    # primary/secondary/wrist map back to the client's own camera keys.
    assert contract.camera_map == {
        "cam_high": "primary",
        "cam_left_wrist": "secondary",
        "cam_right_wrist": "wrist",
    }


def test_infer_contract_rejects_unrecognized_camera_name() -> None:
    config = _fake_config(
        image_features={"observation.images.weird_cam": _image_feature()}, state_dim=7, action_dim=7
    )
    with pytest.raises(ValueError, match="Unrecognized checkpoint image feature"):
        infer_contract(config)


def test_infer_contract_rejects_unrecognized_state_dim() -> None:
    config = _config_2cam_rightarm7d()
    config.robot_state_feature = PolicyFeature(type=FeatureType.STATE, shape=(9,))
    with pytest.raises(ValueError, match="state dim 9"):
        infer_contract(config)


def test_infer_contract_rejects_unrecognized_action_dim() -> None:
    config = _config_2cam_rightarm7d()
    config.action_feature = PolicyFeature(type=FeatureType.ACTION, shape=(9,))
    with pytest.raises(ValueError, match="action dim 9"):
        infer_contract(config)


# ---------------------------------------------------------------------------
# Universal expectations
# ---------------------------------------------------------------------------


def test_validate_universal_expectations_passes_for_real_checkpoint_shape() -> None:
    validate_universal_expectations(_config_2cam_rightarm7d())  # must not raise


def test_validate_universal_expectations_rejects_mismatched_dtype() -> None:
    config = _config_2cam_rightarm7d(dtype="float32")
    with pytest.raises(ValueError, match="dtype"):
        validate_universal_expectations(config)


def test_validate_universal_expectations_rejects_mismatched_normalization() -> None:
    config = _config_2cam_rightarm7d(
        normalization_mapping={"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "QUANTILES"}
    )
    with pytest.raises(ValueError, match="normalization_mapping"):
        validate_universal_expectations(config)


# ---------------------------------------------------------------------------
# RTC guidance (protocol v2) and prompt-change reset behavior
# ---------------------------------------------------------------------------


def test_rtc_passes_previous_normalized_chunk_and_runtime_delay() -> None:
    rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=10,
        max_guidance_weight=10.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    )
    policy = _FakePolicy(rtc_config)
    server = _make_server(policy, rtc_inference_delay=4)
    observation = {OBS_STATE: torch.zeros((1, 19))}
    raw_client_state = np.zeros(14, dtype=np.float32)

    first_actions = server._run_pipeline(
        observation, raw_client_state, rtc_query_step=0, rtc_inference_delay=4
    )
    second_actions = server._run_pipeline(
        observation,
        raw_client_state,
        rtc_query_step=10,
        rtc_inference_delay=6,
    )

    assert first_actions.shape == (50, 14)
    assert second_actions.shape == (50, 14)
    # First call ever: no previous chunk to guide from, so no guidance kwargs are passed.
    assert policy.calls[0] == {}
    assert policy.calls[1]["inference_delay"] == 6
    # query_delta = 10 - 0 = 10, so the overlap is the last (50 - 10) = 40 steps of chunk 0.
    assert policy.calls[1]["prev_chunk_left_over"].shape == (1, 40, 16)
    assert torch.equal(
        policy.calls[1]["prev_chunk_left_over"][0, :, 0],
        torch.arange(10, 50, dtype=torch.float32),
    )

    server.reset()
    server._run_pipeline(observation, raw_client_state, rtc_query_step=0, rtc_inference_delay=4)
    assert policy.calls[2] == {}


def test_rtc_kwargs_are_omitted_when_disabled() -> None:
    policy = _FakePolicy(rtc_config=None)
    server = _make_server(policy)

    server._run_pipeline({OBS_STATE: torch.zeros((1, 19))}, np.zeros(14, dtype=np.float32))

    assert policy.calls == [{}]


def test_prompt_change_clears_rtc_prefix_before_inference() -> None:
    rtc_config = RTCConfig(enabled=True, execution_horizon=10)
    policy = _FakePolicy(rtc_config)
    server = _make_server(policy, rtc_inference_delay=4)
    observation = {OBS_STATE: torch.zeros((1, 19))}
    raw_client_state = np.zeros(14, dtype=np.float32)
    examples = [
        {
            "images": {},
            "state": raw_client_state,
            "prompt": "old task",
            "rtc_query_step": 0,
            "rtc_inference_delay": 4,
        }
    ]

    # Exercise prompt-boundary behavior without requiring deployment cameras.
    server._adapt_example = lambda ex: ({**observation, "task": ex["prompt"]}, raw_client_state)
    server._print_actions = lambda actions, call_idx: None
    server.predict_action(examples)

    examples[0]["rtc_query_step"] = 10
    server.predict_action(examples)

    examples[0]["prompt"] = "new task"
    examples[0]["rtc_query_step"] = 20
    server.predict_action(examples)

    assert policy.calls[1]["prev_chunk_left_over"] is not None
    assert policy.calls[2] == {}
    assert policy.reset_count == 1


# ---------------------------------------------------------------------------
# Observation/action adaptation (segment-based state/action layout)
# ---------------------------------------------------------------------------


def _make_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(3, 224, 224), dtype=np.uint8)


def test_adapt_example_rightarm7d_slices_state_and_holds_left_arm() -> None:
    policy = _FakePolicy(rtc_config=None, action_dim=7, horizon=25)
    server = _make_server(policy, config=_config_2cam_rightarm7d())
    state = np.arange(14, dtype=np.float32)
    ex = {
        "images": {"cam_high": _make_image(0), "cam_right_wrist": _make_image(1)},
        "state": state,
        "prompt": "do the thing",
    }

    observation, raw_client_state = server._adapt_example(ex)

    assert np.array_equal(raw_client_state, state)
    assert observation[OBS_STATE].shape == (1, 7)
    assert torch.equal(observation[OBS_STATE][0], torch.arange(7, 14, dtype=torch.float32))
    assert set(observation) >= {
        "observation.images.cam_high",
        "observation.images.cam_right_wrist",
        OBS_STATE,
        "task",
    }


def test_adapt_example_bimanual_base_pads_with_zeros() -> None:
    policy = _FakePolicy(rtc_config=None, action_dim=16, horizon=50)
    server = _make_server(policy, config=_config_3cam_bimanual_base_legacy())
    state = np.arange(14, dtype=np.float32)
    ex = {
        "images": {
            "cam_high": _make_image(0),
            "cam_left_wrist": _make_image(1),
            "cam_right_wrist": _make_image(2),
        },
        "state": state,
        "prompt": "do the thing",
    }

    observation, raw_client_state = server._adapt_example(ex)

    assert observation[OBS_STATE].shape == (1, 19)
    assert torch.equal(observation[OBS_STATE][0, :14], torch.arange(14, dtype=torch.float32))
    assert torch.equal(observation[OBS_STATE][0, 14:], torch.zeros(5))


def test_run_pipeline_reassembles_client_action_from_held_and_model_segments() -> None:
    policy = _FakePolicy(rtc_config=None, action_dim=7, horizon=25)
    server = _make_server(policy, config=_config_2cam_rightarm7d())
    observation = {OBS_STATE: torch.zeros((1, 7))}
    raw_client_state = np.concatenate([np.full(7, -1.0, dtype=np.float32), np.zeros(7, dtype=np.float32)])

    actions = server._run_pipeline(observation, raw_client_state)

    assert actions.shape == (25, 14)
    # held_client_slice(0:7) -> the measured left arm, broadcast across the horizon.
    assert np.array_equal(actions[:, :7], np.full((25, 7), -1.0, dtype=np.float32))
    # model_slice(0:7) -> the model's own 7-dim output, timestep t has value t at every column.
    assert np.array_equal(actions[:, 7], np.arange(25, dtype=np.float32))


def test_run_pipeline_full_bimanual_nothing_held() -> None:
    policy = _FakePolicy(rtc_config=None, action_dim=14, horizon=25)
    server = _make_server(policy, config=_config_3cam_bimanual())
    observation = {OBS_STATE: torch.zeros((1, 14))}
    raw_client_state = np.zeros(14, dtype=np.float32)

    actions = server._run_pipeline(observation, raw_client_state)

    assert actions.shape == (25, 14)
    # model_slice(0:14) covers the entire client action -- nothing held, nothing dropped.
    assert np.array_equal(actions[:, 0], np.arange(25, dtype=np.float32))
