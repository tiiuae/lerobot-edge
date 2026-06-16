import numpy as np
from joint_to_ee.poses import normalize_gripper, se3_to_pose7
from joint_to_ee import constants as C


def test_normalize_gripper_endpoints_and_clip():
    assert normalize_gripper(C.GRIPPER_CLOSED) == 0.0
    assert normalize_gripper(C.GRIPPER_OPEN) == 1.0
    assert normalize_gripper(-1.0) == 0.0        # clipped low
    assert normalize_gripper(99.0) == 1.0        # clipped high
    assert np.isclose(normalize_gripper(C.GRIPPER_OPEN / 2), 0.5)


def test_se3_to_pose7_identity():
    pose = se3_to_pose7(np.eye(4))
    assert np.allclose(pose[:3], [0, 0, 0])
    assert np.allclose(pose[3:], [1, 0, 0, 0])   # qw,qx,qy,qz = identity
    assert pose.dtype == np.float32
