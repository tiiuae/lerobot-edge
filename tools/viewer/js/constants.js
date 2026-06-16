// Static constants: joint maps, layout indices, mount offsets, colour palette.
export const JOINT_MAP = {
  left_joint_0:  'follower_left_joint_0',
  left_joint_1:  'follower_left_joint_1',
  left_joint_2:  'follower_left_joint_2',
  left_joint_3:  'follower_left_joint_3',
  left_joint_4:  'follower_left_joint_4',
  left_joint_5:  'follower_left_joint_5',
  right_joint_0: 'follower_right_joint_0',
  right_joint_1: 'follower_right_joint_1',
  right_joint_2: 'follower_right_joint_2',
  right_joint_3: 'follower_right_joint_3',
  right_joint_4: 'follower_right_joint_4',
  right_joint_5: 'follower_right_joint_5',
};
export const GRIPPER_MAP = {
  left_joint_6:  ['follower_left_right_carriage_joint',  'follower_left_left_carriage_joint'],
  right_joint_6: ['follower_right_right_carriage_joint', 'follower_right_left_carriage_joint'],
};

// arms-first layout — metadata names are mislabeled, use fixed indices instead.
export const STATE_IDX = {
  left_joint_0: 0,  left_joint_1: 1,  left_joint_2: 2,  left_joint_3: 3,
  left_joint_4: 4,  left_joint_5: 5,  left_joint_6: 6,
  right_joint_0: 7, right_joint_1: 8, right_joint_2: 9, right_joint_3: 10,
  right_joint_4: 11, right_joint_5: 12, right_joint_6: 13,
};

export const LEFT_MOUNT  = [0.331,  0.3, 0.831];
export const RIGHT_MOUNT = [0.331, -0.3, 0.831];

export const BG    = 0xf0f2f5;
export const GRID1 = 0x999999;
export const GRID2 = 0xcccccc;

export const C_OBS_L = 0x2266cc;
export const C_OBS_R = 0xcc2211;
export const C_SEC_L = 0x00bbdd;
export const C_SEC_R = 0xdd7700;
export const C_ERR   = 0xffdd00;
export const C_DELTA = 0x22cc55;
export const C_REL   = 0xaa44ff;
