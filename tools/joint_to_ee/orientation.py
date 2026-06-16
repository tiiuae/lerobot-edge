"""Geometrically correct orientation differences for EE delta/relative.

Quaternion storage convention in this codebase: [qw, qx, qy, qz].
scipy.spatial.transform.Rotation uses [qx, qy, qz, qw]; convert at the boundary.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _to_scipy(q_wxyz) -> Rotation:
    qw, qx, qy, qz = q_wxyz
    return Rotation.from_quat([qx, qy, qz, qw])


def _canonical_wxyz(rot: Rotation) -> np.ndarray:
    qx, qy, qz, qw = rot.as_quat()
    if qw < 0:                       # enforce qw >= 0 for a unique representation
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    return np.array([qw, qx, qy, qz], dtype=np.float32)


def relative_rotation(q_ref_wxyz, q_cur_wxyz) -> Rotation:
    """Rotation from ref to cur expressed in the ref frame: R_ref^{-1} * R_cur."""
    return _to_scipy(q_ref_wxyz).inv() * _to_scipy(q_cur_wxyz)


def orientation_delta_quat(q_ref_wxyz, q_cur_wxyz) -> np.ndarray:
    """Relative orientation as a canonical unit quaternion [qw, qx, qy, qz] (float32)."""
    return _canonical_wxyz(relative_rotation(q_ref_wxyz, q_cur_wxyz))


def orientation_delta_rotvec(q_ref_wxyz, q_cur_wxyz) -> np.ndarray:
    """Relative orientation as a rotation vector (axis * angle, radians) [rx, ry, rz]."""
    return relative_rotation(q_ref_wxyz, q_cur_wxyz).as_rotvec().astype(np.float32)
