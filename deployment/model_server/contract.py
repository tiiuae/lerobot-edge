"""Infers how a checkpoint maps onto the fixed Trossen wire protocol, straight from its own
config -- camera names and state/action width select a fixed rule below, no per-checkpoint file
needed. An unrecognized shape raises a clear error; extend the rule
tables here to support it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Fixed Trossen client wire contract (never varies per checkpoint) ---
CLIENT_CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CLIENT_STATE_DIM = 14
CLIENT_ACTION_DIM = 14

# Checkpoint image-feature bare names (after stripping "observation.images.") that don't match a
# client camera key directly -- the naming convention used by the original pi0.5 deployment
# checkpoints. Every checkpoint since (mot_falcon, and the "_realworld" pi05 training run) names
# its image features identically to the client's own camera keys, needing no translation at all.
_LEGACY_CAMERA_NAMES = {"primary": "cam_high", "secondary": "cam_left_wrist", "wrist": "cam_right_wrist"}


@dataclass(frozen=True)
class CameraMapping:
    """One client camera key mapped to one checkpoint image feature."""

    client_key: str
    checkpoint_feature: str
    native_shape: tuple[int, int, int]

    @property
    def feature_key(self) -> str:
        return f"observation.images.{self.checkpoint_feature}"


@dataclass(frozen=True)
class StateSegment:
    """One ordered piece of the checkpoint state vector, built from the client's raw state."""

    source: str  # "client_slice" | "zeros"
    length: int
    start: int | None = None  # required when source == "client_slice"


@dataclass(frozen=True)
class ActionSegment:
    """One ordered piece of the client action vector, built from model output + held client state."""

    source: str  # "model_slice" | "held_client_slice"
    start: int
    length: int


# State/action segment layouts, keyed by the checkpoint's own declared dimension. Every checkpoint
# contract seen so far (pi05 and mot_falcon, 2 or 3 cameras, several training runs) fits one of
# these three widths per side -- add a new entry here if a checkpoint with an unseen state/action
# width needs to be served.
_STATE_LAYOUT_BY_DIM: dict[int, list[StateSegment]] = {
    7: [StateSegment("client_slice", start=7, length=7)],  # right arm only, left arm unmodelled
    14: [StateSegment("client_slice", start=0, length=14)],  # full bimanual, no padding
    19: [  # bimanual + 5 stationary mobile-base fields the client never sends
        StateSegment("client_slice", start=0, length=14),
        StateSegment("zeros", length=5),
    ],
}
_ACTION_LAYOUT_BY_DIM: dict[int, list[ActionSegment]] = {
    7: [  # right arm only; left arm held at its measured pose
        ActionSegment("held_client_slice", start=0, length=7),
        ActionSegment("model_slice", start=0, length=7),
    ],
    14: [ActionSegment("model_slice", start=0, length=14)],  # full bimanual, nothing held
    16: [  # left7 + right7 + base_velocity2; left arm held, model's left/base columns discarded
        ActionSegment("held_client_slice", start=0, length=7),
        ActionSegment("model_slice", start=7, length=7),
    ],
}

# Fixed expectations for ANY checkpoint compatible with this specific deployment -- not
# per-checkpoint-family, just what every known-good checkpoint for this robot/wire protocol
# shares (confirmed identical across every pi05 and mot_falcon checkpoint tested so far).
UNIVERSAL_EXPECTATIONS: dict[str, object] = {
    "n_obs_steps": 1,
    "num_inference_steps": 10,
    "max_state_dim": 32,
    "max_action_dim": 32,
    "tokenizer_max_length": 200,
    "dtype": "bfloat16",
    "use_relative_actions": False,
    "empty_cameras": 0,
    "image_resolution": (224, 224),
    "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"},
}


@dataclass(frozen=True)
class CheckpointContract:
    """The inferred adapter behavior for one loaded checkpoint config."""

    policy_type: str
    cameras: list[CameraMapping]
    tolerated_extra_client_cameras: list[str] = field(default_factory=list)
    checkpoint_state_layout: list[StateSegment] = field(default_factory=list)
    checkpoint_action_dim: int = 0
    checkpoint_action_layout: list[ActionSegment] = field(default_factory=list)
    client_state_dim: int = CLIENT_STATE_DIM
    client_action_dim: int = CLIENT_ACTION_DIM

    @property
    def checkpoint_state_dim(self) -> int:
        return sum(seg.length for seg in self.checkpoint_state_layout)

    @property
    def camera_map(self) -> dict[str, str]:
        """client_key -> bare checkpoint feature name (e.g. 'primary'), in declared order."""
        return {c.client_key: c.checkpoint_feature for c in self.cameras}

    @property
    def image_shapes(self) -> dict[str, tuple[int, int, int]]:
        """checkpoint feature_key (e.g. 'observation.images.primary') -> native (C,H,W)."""
        return {c.feature_key: c.native_shape for c in self.cameras}


def _infer_cameras(config) -> list[CameraMapping]:
    cameras = []
    for feature_key, feature in (config.image_features or {}).items():
        bare_name = feature_key.removeprefix("observation.images.")
        if bare_name in CLIENT_CAMERA_KEYS:
            client_key = bare_name
        elif bare_name in _LEGACY_CAMERA_NAMES:
            client_key = _LEGACY_CAMERA_NAMES[bare_name]
        else:
            raise ValueError(
                f"Unrecognized checkpoint image feature {feature_key!r}: {bare_name!r} doesn't "
                f"match a client camera key {CLIENT_CAMERA_KEYS} or a known legacy name "
                f"{sorted(_LEGACY_CAMERA_NAMES)}. Add a mapping rule to "
                f"deployment/model_server/contract.py if this is a new, legitimate convention."
            )
        cameras.append(CameraMapping(client_key, bare_name, tuple(feature.shape)))
    if not cameras:
        raise ValueError("Checkpoint declares no image features")
    return cameras


def infer_contract(config) -> CheckpointContract:
    """Build a :class:`CheckpointContract` for ``config`` by reading its own declared shapes and
    applying the fixed rule set above.

    Raises ``ValueError`` with a specific, actionable message if the checkpoint's camera names or
    state/action width don't match a known pattern.
    """
    cameras = _infer_cameras(config)
    tolerated_extra = sorted(set(CLIENT_CAMERA_KEYS) - {c.client_key for c in cameras})

    state_feature = config.robot_state_feature
    if state_feature is None:
        raise ValueError("Checkpoint declares no state feature")
    state_dim = state_feature.shape[0]
    if state_dim not in _STATE_LAYOUT_BY_DIM:
        raise ValueError(
            f"Checkpoint state dim {state_dim} doesn't match a known layout "
            f"{sorted(_STATE_LAYOUT_BY_DIM)}. Add a rule to deployment/model_server/contract.py."
        )

    action_feature = config.action_feature
    if action_feature is None:
        raise ValueError("Checkpoint declares no action feature")
    action_dim = action_feature.shape[0]
    if action_dim not in _ACTION_LAYOUT_BY_DIM:
        raise ValueError(
            f"Checkpoint action dim {action_dim} doesn't match a known layout "
            f"{sorted(_ACTION_LAYOUT_BY_DIM)}. Add a rule to deployment/model_server/contract.py."
        )

    return CheckpointContract(
        policy_type=config.type,
        cameras=cameras,
        tolerated_extra_client_cameras=tolerated_extra,
        checkpoint_state_layout=_STATE_LAYOUT_BY_DIM[state_dim],
        checkpoint_action_dim=action_dim,
        checkpoint_action_layout=_ACTION_LAYOUT_BY_DIM[action_dim],
    )


def validate_universal_expectations(config) -> None:
    """Check the small, fixed set of expectations every checkpoint compatible with this specific
    Trossen deployment shares, regardless of policy type or training run (dtype, normalization
    scheme, image resolution, etc.) -- independent of the shape-driven inference in
    :func:`infer_contract`, which only looks at camera names and state/action width.
    """
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    for attr, expected in UNIVERSAL_EXPECTATIONS.items():
        if attr == "normalization_mapping":
            actual = {
                k: getattr(v, "value", v) for k, v in (getattr(config, attr, {}) or {}).items()
            }
        elif attr == "image_resolution":
            actual = tuple(getattr(config, attr, ()) or ())
        else:
            actual = getattr(config, attr, None)
        require(actual == expected, f"{attr} must be {expected!r} (checkpoint has {actual!r})")

    if errors:
        raise ValueError(
            "Checkpoint does not match this deployment's fixed expectations:\n- " + "\n- ".join(errors)
        )
