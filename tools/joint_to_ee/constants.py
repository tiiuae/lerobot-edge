"""Static configuration: URDF paths, joint layout, mounts, gripper limits, dims."""
from pathlib import Path

import numpy as np

# ROS workspace so placo can resolve package:// mesh paths in the URDF.
TROSSEN_WORKSPACE = Path("/home/edgeai/trossen_arm_ros")

WXAI_FOLLOWER_URDF = (
    TROSSEN_WORKSPACE / "trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf"
)
WXAI_ARM_JOINTS = [f"joint_{i}" for i in range(6)]   # joint_0..joint_5; gripper excluded
WXAI_EE_FRAME = "ee_gripper_link"

# Arm mount offsets in robot base_link frame (verified against mobile_ai.urdf).
LEFT_MOUNT_XYZ = np.array([0.331, 0.3, 0.831])
RIGHT_MOUNT_XYZ = np.array([0.331, -0.3, 0.831])

# arms-first index slices — FK uses joints 0..5 (joint_6 = gripper, excluded).
OBS_LEFT_JOINTS = list(range(0, 6))
OBS_RIGHT_JOINTS = list(range(7, 13))
ACT_LEFT_JOINTS = list(range(0, 6))
ACT_RIGHT_JOINTS = list(range(7, 13))

ACT_DIM = 16        # total action vector length
ACT_JOINT_DIM = 14  # action[0..13] are joint positions (both arms)

LEFT_GRIPPER_IDX = 6
RIGHT_GRIPPER_IDX = 13
GRIPPER_OPEN = 0.044    # m, URDF carriage_joint upper limit
GRIPPER_CLOSED = 0.0    # m

EE_DIM = 8          # [x,y,z,qw,qx,qy,qz,gripper]
