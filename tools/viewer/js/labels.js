// Column-aware axis labels for EE / representation vectors.
// Handles 7-dim (legacy quat, no gripper), 8-dim (quat + gripper), and .rotvec (7-dim).
export function eeLabels(columnKey, len) {
  const isRotvec = columnKey.endsWith('.rotvec');
  if (isRotvec) {                        // [x,y,z, rx,ry,rz, (gripper)]
    const base = ['x', 'y', 'z', 'rx', 'ry', 'rz'];
    return len >= 7 ? [...base, 'gripper'] : base.slice(0, len);
  }
  const quat = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'];   // pose / quat-delta
  if (len === 8) return [...quat, 'gripper'];
  if (len === 7) return quat;
  if (len === 3) return ['dx', 'dy', 'dz'];
  return Array.from({ length: len }, (_, i) => `[${i}]`);
}
