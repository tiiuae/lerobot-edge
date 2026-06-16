"""meta/info.json feature definitions for the enriched columns."""
import json
from pathlib import Path

from . import constants as C

_EE8_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]
_ROTVEC7_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]


def _feature(dim, names, description):
    return {"dtype": "float32", "shape": [dim], "names": names, "description": description}


def feature_ee(side, frame_label):
    return _feature(
        C.EE_DIM,
        [f"ee_{side}_{n}" for n in _EE8_NAMES],
        f"End-effector pose of the {side} wxai arm "
        f"([x,y,z,qw,qx,qy,qz,gripper] in {frame_label} frame). "
        "FK from joint_0..joint_5 via lerobot.model.kinematics (placo), wxai_follower.urdf. "
        f"gripper = joint_6 normalized to [0,1] (URDF carriage range [0, {C.GRIPPER_OPEN}] m).",
    )


def update_info_json(info_path: Path, *, include_joint_repr: bool, ref_frame: str,
                     rot_repr: str) -> None:
    info = json.loads(info_path.read_text())
    feats = info.setdefault("features", {})
    fl = "robot_base" if ref_frame == "robot_base" else "arm_base"

    feats["observation.ee_left"] = feature_ee("left", fl)
    feats["observation.ee_right"] = feature_ee("right", fl)
    feats["action.ee_left"] = feature_ee("left", fl)
    feats["action.ee_right"] = feature_ee("right", fl)

    if include_joint_repr:
        feats["action.delta"] = _feature(
            C.ACT_DIM, None,
            "Sequential joint-space delta action[t]-action[t-1]; t=0 uses state joints. "
            "Dims 0..13=arm joints, 14..15=linear/angular velocity.")
        feats["action.relative"] = _feature(
            C.ACT_DIM, None,
            "Relative joint-space action[t]-state[t] for joint dims (0..13); "
            "velocity dims (14..15) kept as-is.")

    quat_desc = ("EE {kind} of the {side} arm: position (Euclidean), orientation as a "
                 "relative unit quaternion [qw,qx,qy,qz], gripper (Euclidean).")
    rv_desc = ("EE {kind} of the {side} arm: position (Euclidean), orientation as a "
               "rotation vector [rx,ry,rz] (rad), gripper (Euclidean).")

    for side in ("left", "right"):
        for kind in ("delta", "relative"):
            base = f"action.ee_{side}.{kind}"
            if rot_repr in ("quat", "both"):
                feats[base] = _feature(8, [f"ee_{side}_{n}" for n in _EE8_NAMES],
                                       quat_desc.format(kind=kind, side=side))
            if rot_repr == "rotvec":
                feats[base] = _feature(7, [f"ee_{side}_{n}" for n in _ROTVEC7_NAMES],
                                       rv_desc.format(kind=kind, side=side))
            if rot_repr == "both":
                feats[f"{base}.rotvec"] = _feature(7, [f"ee_{side}_{n}" for n in _ROTVEC7_NAMES],
                                                   rv_desc.format(kind=kind, side=side))

    info_path.write_text(json.dumps(info, indent=2))
