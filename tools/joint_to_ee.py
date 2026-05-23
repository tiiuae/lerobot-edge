#!/usr/bin/env python3
"""joint-to-ee.py

Enrich a LeRobot dataset (wxai / Mobile AI dual-arm robot) with:

  1. End-effector poses  (FK via lerobot.model.kinematics / placo)
       observation.ee_left   [8]  [x, y, z, qw, qx, qy, qz, gripper]  from state joints
       observation.ee_right  [8]  same for right arm
       action.ee_left        [8]  from action joints  (--include-action)
       action.ee_right       [8]  same for right arm  (--include-action)
     gripper is normalized to [0, 1]: 0 = closed, 1 = open (from URDF carriage joint limits)

  2. Action representations  (delta and relative, per episode)
       action.delta             [16]  action[t] - action[t-1]  (t=0: vs state)
       action.relative          [16]  action[t] - state[t]     (joint dims only)
       action.ee_left.delta     [8]   sequential EE delta      (--include-action)
       action.ee_right.delta    [8]   sequential EE delta      (--include-action)
       action.ee_left.relative  [8]   action EE - obs EE       (--include-action)
       action.ee_right.relative [8]   action EE - obs EE       (--include-action)

JOINT LAYOUT  (arms-first — ignores mislabeled metadata):
  state[0..6]    = left_joint_0..6   (joint_6 = gripper, excluded from FK)
  state[7..13]   = right_joint_0..6
  state[14..18]  = base info (odom_x/y/theta, linear_vel, angular_vel)
  action[0..6]   = left_joint_0..6
  action[7..13]  = right_joint_0..6
  action[14..15] = linear_vel, angular_vel

Usage:
  python tools/joint-to-ee.py --dataset /path/to/dataset
  python tools/joint-to-ee.py --dataset /path/to/dataset --frame arm
  python tools/joint-to-ee.py --dataset /path/to/dataset --include-action
  python tools/joint-to-ee.py --dataset /path/to/dataset --output /path/to/new
"""

import argparse
import contextlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.progress import track
from scipy.spatial.transform import Rotation

console = Console()

# ── placo / ROS workspace ─────────────────────────────────────────────────────
# Set ROS_PACKAGE_PATH before importing lerobot so placo can resolve
# `package://trossen_arm_description/...` mesh paths in the URDF.
_TROSSEN_WORKSPACE = Path("/home/edgeai/trossen_arm_ros")
if "ROS_PACKAGE_PATH" not in os.environ:
    os.environ["ROS_PACKAGE_PATH"] = str(_TROSSEN_WORKSPACE)

from lerobot.model.kinematics import RobotKinematics  # noqa: E402

# ── wxai kinematics config ────────────────────────────────────────────────────
_WXAI_FOLLOWER_URDF = (
    _TROSSEN_WORKSPACE
    / "trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf"
)
_WXAI_ARM_JOINTS = [f"joint_{i}" for i in range(6)]  # joint_0..joint_5; gripper excluded
_WXAI_EE_FRAME   = "ee_gripper_link"

# Arm mount offsets in robot base_link frame (from mobile_ai.urdf)
_LEFT_MOUNT_XYZ  = np.array([0.331,  0.3, 0.831])
_RIGHT_MOUNT_XYZ = np.array([0.331, -0.3, 0.831])

# arms-first index slices — FK uses joints 0..5 (joint_6 = gripper, excluded)
_OBS_LEFT_JOINTS  = list(range(0, 6))   # state[0..5]
_OBS_RIGHT_JOINTS = list(range(7, 13))  # state[7..12]
_ACT_LEFT_JOINTS  = list(range(0, 6))   # action[0..5]
_ACT_RIGHT_JOINTS = list(range(7, 13))  # action[7..12]

_ACT_DIM       = 16   # total action vector length
_ACT_JOINT_DIM = 14   # action[0..13] are joint positions (both arms)

_LEFT_GRIPPER_IDX  = 6    # state[6]  / action[6]
_RIGHT_GRIPPER_IDX = 13   # state[13] / action[13]
_GRIPPER_OPEN   = 0.044   # m, from URDF left/right_carriage_joint upper limit
_GRIPPER_CLOSED = 0.0     # m


# ── helpers ───────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _silence_fd(fd: int):
    """Redirect a raw file descriptor to /dev/null (suppresses C++ noise from placo)."""
    saved   = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


def _make_kinematics() -> RobotKinematics:
    if not _WXAI_FOLLOWER_URDF.exists():
        raise FileNotFoundError(
            f"wxai follower URDF not found: {_WXAI_FOLLOWER_URDF}\n"
            "Set _TROSSEN_WORKSPACE at the top of this script."
        )
    with _silence_fd(1), _silence_fd(2):
        return RobotKinematics(
            urdf_path=str(_WXAI_FOLLOWER_URDF),
            target_frame_name=_WXAI_EE_FRAME,
            joint_names=_WXAI_ARM_JOINTS,
        )


def _arm_fk(kin: RobotKinematics, q_rad: np.ndarray) -> np.ndarray:
    """FK for 6 joints (radians) → 4×4 SE3 in arm base_link frame.
    RobotKinematics.forward_kinematics expects degrees."""
    return kin.forward_kinematics(np.rad2deg(q_rad))


def _apply_mount(tf: np.ndarray, mount_xyz: np.ndarray | None) -> np.ndarray:
    """Pre-multiply by T(mount_xyz) to express the pose in robot base_link frame."""
    if mount_xyz is None:
        return tf
    t_mount = np.eye(4, dtype=np.float64)
    t_mount[:3, 3] = mount_xyz
    return t_mount @ tf


def _se3_to_pose7(tf: np.ndarray) -> np.ndarray:
    """4×4 SE3 → [x, y, z, qw, qx, qy, qz] float32."""
    pos    = tf[:3, 3].astype(np.float32)
    qxyz_w = Rotation.from_matrix(tf[:3, :3]).as_quat()  # scipy: [qx, qy, qz, qw]
    return np.array(
        [pos[0], pos[1], pos[2], qxyz_w[3], qxyz_w[0], qxyz_w[1], qxyz_w[2]],
        dtype=np.float32,
    )


def _normalize_gripper(val: float) -> float:
    return float(np.clip((val - _GRIPPER_CLOSED) / (_GRIPPER_OPEN - _GRIPPER_CLOSED), 0.0, 1.0))


def _fk_pose8(kin: RobotKinematics, q_arr: np.ndarray, joint_idx: list[int],
              mount_xyz: np.ndarray | None, gripper_val: float) -> np.ndarray:
    """FK pose [x,y,z,qw,qx,qy,qz] + normalized gripper → [8] float32."""
    pose7 = _se3_to_pose7(_apply_mount(_arm_fk(kin, q_arr[joint_idx]), mount_xyz))
    return np.append(pose7, np.float32(_normalize_gripper(gripper_val)))


# ── per-parquet processing ────────────────────────────────────────────────────

def _enrich_table(
    tbl: pa.Table,
    kin: RobotKinematics,
    left_mount: np.ndarray | None,
    right_mount: np.ndarray | None,
    include_action: bool,
    include_joint_repr: bool = True,
) -> pa.Table:
    """Add all new columns to a pyarrow Table and return the augmented table."""
    n = len(tbl)

    episodes   = tbl["episode_index"].to_pylist()
    frame_idxs = tbl["frame_index"].to_pylist()
    states     = [np.asarray(s, np.float64) for s in tbl["observation.state"].to_pylist()]
    actions    = [np.asarray(a, np.float64) for a in tbl["action"].to_pylist()]

    # Output buffers — EE is 8-dim: [x, y, z, qw, qx, qy, qz, gripper_normalized]
    obs_ee_L = np.zeros((n, 8), np.float32)
    obs_ee_R = np.zeros((n, 8), np.float32)
    act_ee_L = np.zeros((n, 8), np.float32) if include_action else None
    act_ee_R = np.zeros((n, 8), np.float32) if include_action else None

    act_delta    = np.zeros((n, _ACT_DIM), np.float32) if include_joint_repr else None
    act_rel      = np.zeros((n, _ACT_DIM), np.float32) if include_joint_repr else None
    ee_L_delta   = np.zeros((n, 8), np.float32) if include_action else None
    ee_R_delta   = np.zeros((n, 8), np.float32) if include_action else None
    ee_L_rel     = np.zeros((n, 8), np.float32) if include_action else None
    ee_R_rel     = np.zeros((n, 8), np.float32) if include_action else None

    # Group rows by episode, sort by frame_index for correct delta ordering
    ep_rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, (ep, fr) in enumerate(zip(episodes, frame_idxs)):
        ep_rows[ep].append((fr, i))

    for ep_rows_list in ep_rows.values():
        ep_rows_list.sort()

        prev_action:   np.ndarray | None = None
        prev_act_ee_L: np.ndarray | None = None
        prev_act_ee_R: np.ndarray | None = None

        for frame_in_ep, (_, row_i) in enumerate(ep_rows_list):
            state  = states[row_i]
            action = actions[row_i]

            # ── EE poses via FK (+ normalized gripper as 8th dim) ─────────
            obs_ee_L[row_i] = _fk_pose8(kin, state,  _OBS_LEFT_JOINTS,  left_mount,  state[_LEFT_GRIPPER_IDX])
            obs_ee_R[row_i] = _fk_pose8(kin, state,  _OBS_RIGHT_JOINTS, right_mount, state[_RIGHT_GRIPPER_IDX])

            if include_action:
                act_ee_L[row_i] = _fk_pose8(kin, action, _ACT_LEFT_JOINTS,  left_mount,  action[_LEFT_GRIPPER_IDX])
                act_ee_R[row_i] = _fk_pose8(kin, action, _ACT_RIGHT_JOINTS, right_mount, action[_RIGHT_GRIPPER_IDX])

            # ── joint action delta ─────────────────────────────────────────
            # t=0: delta vs state joint positions (no prior action exists)
            # t>0: delta vs previous action
            if include_joint_repr:
                if frame_in_ep == 0:
                    ref = np.zeros(_ACT_DIM, np.float64)
                    ref[:_ACT_JOINT_DIM] = state[:_ACT_JOINT_DIM]
                else:
                    ref = prev_action
                act_delta[row_i] = (action - ref).astype(np.float32)

                # action[t] - state[t] for joint dims; velocity dims kept as-is
                rel_ref = np.zeros(_ACT_DIM, np.float64)
                rel_ref[:_ACT_JOINT_DIM] = state[:_ACT_JOINT_DIM]
                act_rel[row_i] = (action - rel_ref).astype(np.float32)

            # ── EE action delta / relative ─────────────────────────────────
            if include_action:
                cur_L     = act_ee_L[row_i]
                cur_R     = act_ee_R[row_i]
                cur_obs_L = obs_ee_L[row_i]
                cur_obs_R = obs_ee_R[row_i]

                if frame_in_ep == 0:
                    ee_L_delta[row_i] = cur_L - cur_obs_L
                    ee_R_delta[row_i] = cur_R - cur_obs_R
                else:
                    ee_L_delta[row_i] = cur_L - prev_act_ee_L
                    ee_R_delta[row_i] = cur_R - prev_act_ee_R

                # relative: how far the action EE target is from the current obs EE
                ee_L_rel[row_i] = cur_L - cur_obs_L
                ee_R_rel[row_i] = cur_R - cur_obs_R

                prev_act_ee_L = cur_L.copy()
                prev_act_ee_R = cur_R.copy()

            prev_action = action.copy()

    def _col(arr2d):
        return pa.array(arr2d.tolist(), type=pa.list_(pa.float32()))

    new_cols = {
        "observation.ee_left":  _col(obs_ee_L),
        "observation.ee_right": _col(obs_ee_R),
    }
    if include_joint_repr:
        new_cols["action.delta"]    = _col(act_delta)
        new_cols["action.relative"] = _col(act_rel)
    if include_action:
        new_cols["action.ee_left"]           = _col(act_ee_L)
        new_cols["action.ee_right"]          = _col(act_ee_R)
        new_cols["action.ee_left.delta"]     = _col(ee_L_delta)
        new_cols["action.ee_right.delta"]    = _col(ee_R_delta)
        new_cols["action.ee_left.relative"]  = _col(ee_L_rel)
        new_cols["action.ee_right.relative"] = _col(ee_R_rel)

    for col_name, col_data in new_cols.items():
        tbl = tbl.append_column(col_name, col_data)
    return tbl


# ── meta/info.json ────────────────────────────────────────────────────────────

def _feature_ee(side: str, frame_label: str) -> dict:
    return {
        "dtype": "float32",
        "shape": [8],
        "names": [f"ee_{side}_{n}" for n in ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]],
        "description": (
            f"End-effector pose of the {side} wxai arm "
            f"([x,y,z,qw,qx,qy,qz,gripper] in {frame_label} frame). "
            "FK from joint_0..joint_5 via lerobot.model.kinematics (placo), wxai_follower.urdf. "
            f"gripper = joint_6 normalized to [0,1]: 0=closed, 1=open "
            f"(URDF carriage joint range [0, {_GRIPPER_OPEN}] m)."
        ),
    }


def _feature_repr(dim: int, description: str) -> dict:
    return {"dtype": "float32", "shape": [dim], "names": None, "description": description}


def _update_info_json(
    info_path: Path, include_action: bool, include_joint_repr: bool, ref_frame: str
) -> None:
    info  = json.loads(info_path.read_text())
    feats = info.setdefault("features", {})
    fl    = "robot_base" if ref_frame == "robot_base" else "arm_base"

    feats["observation.ee_left"]  = _feature_ee("left",  fl)
    feats["observation.ee_right"] = _feature_ee("right", fl)
    if include_joint_repr:
        feats["action.delta"] = _feature_repr(
            _ACT_DIM,
            "Sequential delta of joint-space action: action[t]-action[t-1]. "
            "First frame uses state joints as reference. "
            "Dims 0..13=arm joints; 14..15=linear/angular velocity.",
        )
        feats["action.relative"] = _feature_repr(
            _ACT_DIM,
            "Relative joint-space action: action[t]-state[t] for joint dims (0..13). "
            "Velocity dims (14..15) kept as-is. Centered around zero for stable training.",
        )
    if include_action:
        feats["action.ee_left"]           = _feature_ee("left",  fl)
        feats["action.ee_right"]          = _feature_ee("right", fl)
        feats["action.ee_left.delta"]     = _feature_repr(
            8, "Sequential delta of left EE action pose [xyz,quat,gripper]. t=0: action_ee-obs_ee.")
        feats["action.ee_right.delta"]    = _feature_repr(
            8, "Sequential delta of right EE action pose [xyz,quat,gripper]. t=0: action_ee-obs_ee.")
        feats["action.ee_left.relative"]  = _feature_repr(
            8, "Left EE action relative to current obs EE: action.ee_left[t]-observation.ee_left[t].")
        feats["action.ee_right.relative"] = _feature_repr(
            8, "Right EE action relative to current obs EE: action.ee_right[t]-observation.ee_right[t].")

    info_path.write_text(json.dumps(info, indent=2))


# ── main entry point ──────────────────────────────────────────────────────────

def process_dataset(
    dataset_root: Path,
    output_root: Path,
    ref_frame: str,
    include_action: bool,
    include_joint_repr: bool = True,
) -> None:
    in_place = output_root.resolve() == dataset_root.resolve()

    # Guard against double-enrichment
    info_path = dataset_root / "meta" / "info.json"
    if info_path.exists():
        existing  = set(json.loads(info_path.read_text()).get("features", {}).keys())
        guard     = {"observation.ee_left", "observation.ee_right"}
        if include_joint_repr:
            guard |= {"action.delta", "action.relative"}
        if include_action:
            guard |= {"action.ee_left", "action.ee_right"}
        collision = guard & existing
        if collision:
            raise ValueError(
                f"Features already exist: {sorted(collision)}.\n"
                "Remove them first or start from a freshly merged dataset."
            )

    console.print(f"  joint layout : [dim]arms-first (ignores mislabeled metadata)[/]")
    console.print(f"    obs  left  → state[dim]{_OBS_LEFT_JOINTS}[/]")
    console.print(f"    obs  right → state[dim]{_OBS_RIGHT_JOINTS}[/]")
    if include_action:
        console.print(f"    act  left  → action[dim]{_ACT_LEFT_JOINTS}[/]")
        console.print(f"    act  right → action[dim]{_ACT_RIGHT_JOINTS}[/]")
    console.print(f"  ref frame         : [bold]{ref_frame}[/]")
    console.print(f"  include joint repr: [bold]{include_joint_repr}[/]")
    console.print(f"  include action EE : [bold]{include_action}[/]")

    left_mount  = _LEFT_MOUNT_XYZ  if ref_frame == "robot_base" else None
    right_mount = _RIGHT_MOUNT_XYZ if ref_frame == "robot_base" else None

    console.print(f"  building placo kinematics from [bold]{_WXAI_FOLLOWER_URDF.name}[/] …")
    kin = _make_kinematics()

    pq_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not pq_files:
        raise FileNotFoundError(f"No parquet files under {dataset_root / 'data'}")

    if in_place:
        # Write enriched files to a temp dir, then atomic swap
        tmp_root = Path(tempfile.mkdtemp(
            prefix=f"_{dataset_root.name}_ee_tmp_", dir=dataset_root.parent
        ))
        shutil.copytree(dataset_root, tmp_root, dirs_exist_ok=True)
        work_root = tmp_root
    else:
        shutil.copytree(dataset_root, output_root, dirs_exist_ok=False)
        work_root = output_root

    for pq_src in track(pq_files, description=f"  Enriching {len(pq_files)} parquet file(s)…"):
        pq_dst = work_root / pq_src.relative_to(dataset_root)
        tbl = pq.read_table(pq_src)
        tbl = _enrich_table(tbl, kin, left_mount, right_mount, include_action, include_joint_repr)
        pq.write_table(tbl, pq_dst)

    _update_info_json(work_root / "meta" / "info.json", include_action, include_joint_repr, ref_frame)

    if in_place:
        shutil.rmtree(dataset_root)
        work_root.rename(dataset_root)
        console.print(f"  [green]✔[/]  Dataset enriched in-place at: [bold]{dataset_root}[/]")
    else:
        console.print(f"  [green]✔[/]  Enriched dataset written to: [bold]{work_root}[/]")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Enrich a LeRobot dataset with EE poses and action representations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", required=True, type=Path,
                   help="Path to the LeRobot dataset root directory.")
    p.add_argument("--output",  type=Path, default=None,
                   help="Output directory (default: modify dataset in-place).")
    p.add_argument("--frame",   choices=["arm", "robot_base"], default="robot_base",
                   help=(
                       "Reference frame for EE poses. "
                       "'robot_base' (default) = robot base_link including mount offset; "
                       "'arm' = each arm's own base_link frame."
                   ))
    p.add_argument("--include-action", action="store_true",
                   help=(
                       "Also compute EE poses and delta/relative representations "
                       "for action joints (action.ee_left/right and their delta/relative)."
                   ))
    p.add_argument("--no-joint-repr", action="store_true",
                   help=(
                       "Skip computing joint-space action representations "
                       "(action.delta and action.relative)."
                   ))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dataset_root = args.dataset.expanduser().resolve()
    output_root  = args.output.expanduser().resolve() if args.output else dataset_root

    process_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        ref_frame=args.frame,
        include_action=args.include_action,
        include_joint_repr=not args.no_joint_repr,
    )
