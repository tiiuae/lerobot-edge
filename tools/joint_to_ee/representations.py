"""Action representation math: joint-space delta/relative and EE diff (pos/orient/gripper)."""
import numpy as np

from . import constants as C
from .orientation import orientation_delta_quat, orientation_delta_rotvec

ROT_REPRS = ("quat", "rotvec", "both")


def joint_delta(action: np.ndarray, ref, state: np.ndarray) -> np.ndarray:
    """Sequential joint delta. ref=previous action, or None at t=0 (uses state joints)."""
    if ref is None:
        ref = np.zeros(C.ACT_DIM, np.float64)
        ref[: C.ACT_JOINT_DIM] = state[: C.ACT_JOINT_DIM]
    return (action - ref).astype(np.float32)


def joint_relative(action: np.ndarray, state: np.ndarray) -> np.ndarray:
    """action[t] - state[t] for joint dims; velocity dims (14..15) kept as-is."""
    ref = np.zeros(C.ACT_DIM, np.float64)
    ref[: C.ACT_JOINT_DIM] = state[: C.ACT_JOINT_DIM]
    return (action - ref).astype(np.float32)


def ee_diff(ref8: np.ndarray, cur8: np.ndarray, rot_repr: str) -> dict:
    """Structured EE difference between two [8] poses.

    Position (0:3) and gripper (7) are Euclidean. Orientation (3:7) is a proper
    relative rotation, returned as a unit quaternion ('quat' -> [8]) and/or a
    rotation vector ('rotvec' -> [7]) depending on rot_repr.
    """
    pos = (cur8[:3] - ref8[:3]).astype(np.float32)
    grip = np.float32(cur8[7] - ref8[7])
    out = {}
    if rot_repr in ("quat", "both"):
        q = orientation_delta_quat(ref8[3:7], cur8[3:7])
        out["quat"] = np.concatenate([pos, q, [grip]]).astype(np.float32)     # [8]
    if rot_repr in ("rotvec", "both"):
        rv = orientation_delta_rotvec(ref8[3:7], cur8[3:7])
        out["rotvec"] = np.concatenate([pos, rv, [grip]]).astype(np.float32)  # [7]
    return out
