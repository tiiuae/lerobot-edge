"""placo / RobotKinematics setup and FK helpers."""
import contextlib
import os

import numpy as np

from . import constants as C
from .poses import normalize_gripper, se3_to_pose7

# Set ROS_PACKAGE_PATH before importing lerobot so placo can resolve package:// meshes.
if "ROS_PACKAGE_PATH" not in os.environ:
    os.environ["ROS_PACKAGE_PATH"] = str(C.TROSSEN_WORKSPACE)

from lerobot.model.kinematics import RobotKinematics  # noqa: E402


@contextlib.contextmanager
def _silence_fd(fd: int):
    """Redirect a raw file descriptor to /dev/null (suppresses placo C++ noise)."""
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


def make_kinematics() -> RobotKinematics:
    if not C.WXAI_FOLLOWER_URDF.exists():
        raise FileNotFoundError(
            f"wxai follower URDF not found: {C.WXAI_FOLLOWER_URDF}\n"
            "Set TROSSEN_WORKSPACE in joint_to_ee/constants.py."
        )
    with _silence_fd(1), _silence_fd(2):
        return RobotKinematics(
            urdf_path=str(C.WXAI_FOLLOWER_URDF),
            target_frame_name=C.WXAI_EE_FRAME,
            joint_names=C.WXAI_ARM_JOINTS,
        )


def arm_fk(kin: RobotKinematics, q_rad: np.ndarray) -> np.ndarray:
    """FK for 6 joints (radians) -> 4x4 SE3 in arm base_link frame (placo wants degrees)."""
    return kin.forward_kinematics(np.rad2deg(q_rad))


def apply_mount(tf: np.ndarray, mount_xyz) -> np.ndarray:
    """Pre-multiply by T(mount_xyz) to express the pose in robot base_link frame."""
    if mount_xyz is None:
        return tf
    t_mount = np.eye(4, dtype=np.float64)
    t_mount[:3, 3] = mount_xyz
    return t_mount @ tf


def fk_pose8(kin, q_arr, joint_idx, mount_xyz, gripper_val) -> np.ndarray:
    """FK pose [x,y,z,qw,qx,qy,qz] + normalized gripper -> [8] float32."""
    pose7 = se3_to_pose7(apply_mount(arm_fk(kin, q_arr[joint_idx]), mount_xyz))
    return np.append(pose7, np.float32(normalize_gripper(gripper_val)))
