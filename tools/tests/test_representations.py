import numpy as np
from scipy.spatial.transform import Rotation

from joint_to_ee.representations import ee_diff, joint_delta, joint_relative
from joint_to_ee import constants as C


def _pose8(xyz, rot: Rotation, grip):
    qx, qy, qz, qw = rot.as_quat()
    return np.array([*xyz, qw, qx, qy, qz, grip], dtype=np.float32)


def test_ee_diff_quat_shape_and_position():
    ref = _pose8([0, 0, 0], Rotation.identity(), 0.0)
    cur = _pose8([0.1, -0.2, 0.3], Rotation.from_euler("z", 90, degrees=True), 1.0)
    out = ee_diff(ref, cur, "both")
    assert out["quat"].shape == (8,)
    assert out["rotvec"].shape == (7,)
    assert np.allclose(out["quat"][:3], [0.1, -0.2, 0.3], atol=1e-6)   # Euclidean position
    assert np.isclose(out["quat"][7], 1.0)                            # gripper diff
    assert np.isclose(out["rotvec"][6], 1.0)                          # gripper diff
    # rotvec orientation block = 90deg about z
    assert np.isclose(np.linalg.norm(out["rotvec"][3:6]), np.pi / 2, atol=1e-5)


def test_ee_diff_repr_selection():
    ref = _pose8([0, 0, 0], Rotation.identity(), 0.0)
    cur = _pose8([1, 0, 0], Rotation.identity(), 0.0)
    assert set(ee_diff(ref, cur, "quat")) == {"quat"}
    assert set(ee_diff(ref, cur, "rotvec")) == {"rotvec"}
    assert set(ee_diff(ref, cur, "both")) == {"quat", "rotvec"}


def test_joint_delta_first_frame_uses_state():
    state = np.arange(C.ACT_DIM, dtype=np.float64)
    action = state + 1.0
    d = joint_delta(action, ref=None, state=state)
    # ref built from state joints (0..13), velocity ref = 0
    expected = action.copy()
    expected[: C.ACT_JOINT_DIM] -= state[: C.ACT_JOINT_DIM]
    assert np.allclose(d, expected.astype(np.float32))


def test_joint_relative_velocity_dims_passthrough():
    state = np.zeros(C.ACT_DIM); state[:C.ACT_JOINT_DIM] = 2.0
    action = np.ones(C.ACT_DIM) * 5.0
    r = joint_relative(action, state)
    assert np.allclose(r[: C.ACT_JOINT_DIM], 3.0)         # 5 - 2
    assert np.allclose(r[C.ACT_JOINT_DIM:], 5.0)          # velocity kept as-is
