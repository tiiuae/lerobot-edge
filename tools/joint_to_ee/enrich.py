"""Per-parquet enrichment: compute and append EE + representation columns."""
from collections import defaultdict

import numpy as np
import pyarrow as pa

from . import constants as C
from .kinematics import fk_pose8
from .representations import ee_diff, joint_delta, joint_relative


def _col(arr2d):
    return pa.array(arr2d.tolist(), type=pa.list_(pa.float32()))


def _delta_col_names(base: str, rot_repr: str):
    """Return list of (col_name, repr_key) for a delta/relative base column."""
    if rot_repr == "quat":
        return [(base, "quat")]
    if rot_repr == "rotvec":
        return [(base, "rotvec")]
    return [(base, "quat"), (f"{base}.rotvec", "rotvec")]   # both


def enrich_table(tbl, kin, left_mount, right_mount, *, include_joint_repr=True,
                 rot_repr="both"):
    """Append EE poses + action representations to a pyarrow Table. Action EE is
    always computed (the old --include-action toggle has been removed)."""
    n = len(tbl)
    episodes = tbl["episode_index"].to_pylist()
    frame_idxs = tbl["frame_index"].to_pylist()
    states = [np.asarray(s, np.float64) for s in tbl["observation.state"].to_pylist()]
    actions = [np.asarray(a, np.float64) for a in tbl["action"].to_pylist()]

    obs_ee_L = np.zeros((n, C.EE_DIM), np.float32)
    obs_ee_R = np.zeros((n, C.EE_DIM), np.float32)
    act_ee_L = np.zeros((n, C.EE_DIM), np.float32)
    act_ee_R = np.zeros((n, C.EE_DIM), np.float32)

    act_delta = np.zeros((n, C.ACT_DIM), np.float32) if include_joint_repr else None
    act_rel = np.zeros((n, C.ACT_DIM), np.float32) if include_joint_repr else None

    # EE delta/relative buffers keyed by (side, kind, repr) -> (n, dim) array
    ee_buf = {}
    for side in ("L", "R"):
        for kind in ("delta", "relative"):
            if rot_repr in ("quat", "both"):
                ee_buf[(side, kind, "quat")] = np.zeros((n, 8), np.float32)
            if rot_repr in ("rotvec", "both"):
                ee_buf[(side, kind, "rotvec")] = np.zeros((n, 7), np.float32)

    ep_rows = defaultdict(list)
    for i, (ep, fr) in enumerate(zip(episodes, frame_idxs)):
        ep_rows[ep].append((fr, i))

    for rows in ep_rows.values():
        rows.sort()
        prev_action = prev_act_L = prev_act_R = None

        for frame_in_ep, (_, ri) in enumerate(rows):
            state, action = states[ri], actions[ri]

            obs_ee_L[ri] = fk_pose8(kin, state, C.OBS_LEFT_JOINTS, left_mount, state[C.LEFT_GRIPPER_IDX])
            obs_ee_R[ri] = fk_pose8(kin, state, C.OBS_RIGHT_JOINTS, right_mount, state[C.RIGHT_GRIPPER_IDX])
            act_ee_L[ri] = fk_pose8(kin, action, C.ACT_LEFT_JOINTS, left_mount, action[C.LEFT_GRIPPER_IDX])
            act_ee_R[ri] = fk_pose8(kin, action, C.ACT_RIGHT_JOINTS, right_mount, action[C.RIGHT_GRIPPER_IDX])

            if include_joint_repr:
                act_delta[ri] = joint_delta(action, prev_action, state)
                act_rel[ri] = joint_relative(action, state)

            # EE delta: ref = previous action EE (t=0: obs EE). relative: ref = obs EE.
            for side, cur, prev, obs in (
                ("L", act_ee_L[ri], prev_act_L, obs_ee_L[ri]),
                ("R", act_ee_R[ri], prev_act_R, obs_ee_R[ri]),
            ):
                delta_ref = obs if frame_in_ep == 0 else prev
                for repr_key, vec in ee_diff(delta_ref, cur, rot_repr).items():
                    ee_buf[(side, "delta", repr_key)][ri] = vec
                for repr_key, vec in ee_diff(obs, cur, rot_repr).items():
                    ee_buf[(side, "relative", repr_key)][ri] = vec

            prev_act_L, prev_act_R = act_ee_L[ri].copy(), act_ee_R[ri].copy()
            prev_action = action.copy()

    new_cols = {
        "observation.ee_left": _col(obs_ee_L),
        "observation.ee_right": _col(obs_ee_R),
        "action.ee_left": _col(act_ee_L),
        "action.ee_right": _col(act_ee_R),
    }
    if include_joint_repr:
        new_cols["action.delta"] = _col(act_delta)
        new_cols["action.relative"] = _col(act_rel)

    side_name = {"L": "left", "R": "right"}
    for side in ("L", "R"):
        for kind in ("delta", "relative"):
            base = f"action.ee_{side_name[side]}.{kind}"
            for col_name, repr_key in _delta_col_names(base, rot_repr):
                new_cols[col_name] = _col(ee_buf[(side, kind, repr_key)])

    for name, data in new_cols.items():
        tbl = tbl.append_column(name, data)
    return tbl
