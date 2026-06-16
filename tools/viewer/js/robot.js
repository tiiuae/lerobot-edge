// URDF loading, joint application, and forward kinematics.

import * as THREE from 'three';
import * as K from './constants.js';
import { S } from './state.js';
import { transformJointValue } from './calibration.js';
import { jRosRoot, makeURDFLoader } from './scene.js';

// ── Coordinate-frame correction ───────────────────────────────────────────────
// The URDF root is base_footprint. URDF-loader places it at scene origin, so
// matrixWorld (FK) positions are in base_footprint frame.  The dataset stores
// EE positions in base_link frame (canonical ROS frame).  We read base_link's
// world position after the robot loads and apply it as an additive offset to
// all dataset-derived positions so both frames agree.
// baseLinkOffset is const because we mutate its contents, not reassign it.
export const baseLinkOffset = new THREE.Vector3(0, 0, 0);

export let robot = null;

// ── URDF loading ──────────────────────────────────────────────────────────────
export async function loadRobot() {
  const loader = makeURDFLoader();
  robot = await new Promise((resolve, reject) => {
    loader.load('/robot.urdf', r => {
      r.traverse(child => {
        if (child.isMesh) {
          child.material = new THREE.MeshPhongMaterial({ color: 0x8899aa, shininess: 60, specular: 0x445566 });
          child.castShadow = child.receiveShadow = true;
        }
      });
      resolve(r);
    }, null, reject);
  });
  jRosRoot.add(robot);

  // Determine base_link offset so dataset positions (base_link frame) can be
  // displayed in the scene world frame (base_footprint frame).
  robot.updateMatrixWorld(true);
  const blNode = robot.getObjectByName('base_link');
  if (blNode) baseLinkOffset.setFromMatrixPosition(blNode.matrixWorld);
}

// ── Joint application ─────────────────────────────────────────────────────────
export function buildValuesByName(data, fallbackIdxMap) {
  const byName = {};
  for (const [name, idx] of Object.entries(fallbackIdxMap))
    if (idx < data.length) byName[name] = data[idx];
  return byName;
}

export function applyRobotJoints(rb, byName) {
  for (const [dsName, urdfName] of Object.entries(K.JOINT_MAP))
    if (dsName in byName) rb.setJointValue(urdfName, transformJointValue(dsName, byName[dsName]));
  for (const [dsName, urdfNames] of Object.entries(K.GRIPPER_MAP))
    if (dsName in byName) {
      const v = transformJointValue(dsName, byName[dsName]);
      for (const u of urdfNames) rb.setJointValue(u, v);
    }
}

export function getEEWorldPos(robotObj, linkName) {
  robotObj.updateMatrixWorld(true);
  const link = robotObj.getObjectByName(linkName);
  return link ? new THREE.Vector3().setFromMatrixPosition(link.matrixWorld) : null;
}

export function getEEWorldPose(robotObj, linkName) {
  robotObj.updateMatrixWorld(true);
  const link = robotObj.getObjectByName(linkName);
  if (!link) return null;
  const pos = new THREE.Vector3(), quat = new THREE.Quaternion(), scl = new THREE.Vector3();
  link.matrixWorld.decompose(pos, quat, scl);
  return { pos, quat };
}

// Drive the URDF from observation.state (State mode) or action (Action mode).
export function applyJoints(frame) {
  if (!robot) return;
  const data = S.leftMode === 'state' ? frame['observation.state'] : frame['action'];
  if (!data) return;
  const byName = buildValuesByName(data, K.STATE_IDX);
  applyRobotJoints(robot, byName);
}
