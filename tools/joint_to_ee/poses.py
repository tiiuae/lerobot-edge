"""SE3 → pose conversions and gripper normalization."""
import numpy as np
from scipy.spatial.transform import Rotation

from . import constants as C


def normalize_gripper(val: float) -> float:
    return float(np.clip((val - C.GRIPPER_CLOSED) / (C.GRIPPER_OPEN - C.GRIPPER_CLOSED), 0.0, 1.0))


def se3_to_pose7(tf: np.ndarray) -> np.ndarray:
    """4x4 SE3 -> [x, y, z, qw, qx, qy, qz] float32."""
    pos = tf[:3, 3].astype(np.float32)
    qxyz_w = Rotation.from_matrix(tf[:3, :3]).as_quat()   # scipy: [qx,qy,qz,qw]
    return np.array(
        [pos[0], pos[1], pos[2], qxyz_w[3], qxyz_w[0], qxyz_w[1], qxyz_w[2]],
        dtype=np.float32,
    )
