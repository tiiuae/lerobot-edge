import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from joint_to_ee.orientation import (
    orientation_delta_quat,
    orientation_delta_rotvec,
)

WXYZ_IDENTITY = [1.0, 0.0, 0.0, 0.0]


def _wxyz(rot: Rotation):
    qx, qy, qz, qw = rot.as_quat()
    return [qw, qx, qy, qz]


def test_identity_delta_is_zero():
    assert np.allclose(orientation_delta_quat(WXYZ_IDENTITY, WXYZ_IDENTITY), WXYZ_IDENTITY)
    assert np.allclose(orientation_delta_rotvec(WXYZ_IDENTITY, WXYZ_IDENTITY), [0, 0, 0], atol=1e-6)


def test_rotvec_known_90deg_about_z():
    q_cur = _wxyz(Rotation.from_euler("z", 90, degrees=True))
    rv = orientation_delta_rotvec(WXYZ_IDENTITY, q_cur)
    assert np.isclose(np.linalg.norm(rv), np.pi / 2, atol=1e-5)
    assert np.allclose(rv / np.linalg.norm(rv), [0, 0, 1], atol=1e-5)


def test_quat_result_is_unit_and_canonical():
    q_ref = _wxyz(Rotation.from_euler("xyz", [20, -35, 80], degrees=True))
    q_cur = _wxyz(Rotation.from_euler("xyz", [5, 60, -10], degrees=True))
    q = orientation_delta_quat(q_ref, q_cur)
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)   # unit
    assert q[0] >= 0.0                                      # canonical qw >= 0


def test_quat_and_rotvec_agree_in_angle():
    q_ref = _wxyz(Rotation.from_euler("y", 30, degrees=True))
    q_cur = _wxyz(Rotation.from_euler("y", 75, degrees=True))
    q = orientation_delta_quat(q_ref, q_cur)        # [qw,qx,qy,qz]
    rv = orientation_delta_rotvec(q_ref, q_cur)
    angle_from_quat = 2 * np.arccos(np.clip(q[0], -1, 1))
    assert np.isclose(angle_from_quat, np.linalg.norm(rv), atol=1e-5)
    assert np.isclose(np.linalg.norm(rv), np.deg2rad(45), atol=1e-5)
