// Markers, trails, EE per-frame update, and trajectory builders.

import * as THREE from 'three';
import * as K from './constants.js';
import { S } from './state.js';
import { baseLinkOffset, robot, buildValuesByName, applyRobotJoints, getEEWorldPos, applyJoints } from './robot.js';
import {
  jRosRoot, eRosRoot,
  jObsLeftMark, jObsRightMark,
  eeLeftFrame, eeRightFrame,
  eeObsLeftMark, eeObsRightMark,
  eeSecLeftMark, eeSecRightMark,
  errLeftLine, errRightLine,
  arrowLeft, arrowRight,
  sceneRefs,
  makeLine, setLinePoints, updateArrow,
} from './scene.js';

// ── Coordinate helpers ────────────────────────────────────────────────────────
// Convert a dataset [x,y,z,...] array from base_link frame to scene world frame.
export function worldPos(arr) {
  return new THREE.Vector3(
    arr[0] + baseLinkOffset.x,
    arr[1] + baseLinkOffset.y,
    arr[2] + baseLinkOffset.z,
  );
}

// pose7 = [x, y, z, qw, qx, qy, qz] in base_link frame → placed in scene world frame.
export function applyPose(obj, pose7) {
  const [x, y, z, qw, qx, qy, qz] = pose7;
  obj.position.set(x + baseLinkOffset.x, y + baseLinkOffset.y, z + baseLinkOffset.z);
  obj.quaternion.set(qx, qy, qz, qw);
}

export function posV3(pose7) {
  return new THREE.Vector3(pose7[0], pose7[1], pose7[2]);
}

// ── Trajectory builders ───────────────────────────────────────────────────────
export function clearTrail(trail, root) {
  if (trail) root.remove(trail);
  return null;
}

export function buildObsTrajectories() {
  sceneRefs.jObsLeftTrail  = clearTrail(sceneRefs.jObsLeftTrail,  jRosRoot);
  sceneRefs.jObsRightTrail = clearTrail(sceneRefs.jObsRightTrail, jRosRoot);
  sceneRefs.eeObsLeftTrail = clearTrail(sceneRefs.eeObsLeftTrail, eRosRoot);
  sceneRefs.eeObsRightTrail= clearTrail(sceneRefs.eeObsRightTrail,eRosRoot);

  if (S.hasEELeft) {
    const pts = S.frames.map(f => f['observation.ee_left']).filter(Boolean).map(worldPos);
    sceneRefs.eeObsLeftTrail = makeLine(pts, K.C_OBS_L, 0.6); eRosRoot.add(sceneRefs.eeObsLeftTrail);
    sceneRefs.jObsLeftTrail  = makeLine(pts, K.C_OBS_L, 0.6); jRosRoot.add(sceneRefs.jObsLeftTrail);
  }
  if (S.hasEERight) {
    const pts = S.frames.map(f => f['observation.ee_right']).filter(Boolean).map(worldPos);
    sceneRefs.eeObsRightTrail = makeLine(pts, K.C_OBS_R, 0.6); eRosRoot.add(sceneRefs.eeObsRightTrail);
    sceneRefs.jObsRightTrail  = makeLine(pts, K.C_OBS_R, 0.6); jRosRoot.add(sceneRefs.jObsRightTrail);
  }
}

// Precompute FK EE trajectory from URDF (using observation.state).
// Used only in FK Validation mode.
export function buildFKTrajectories() {
  sceneRefs.eeSecLeftTrail  = clearTrail(sceneRefs.eeSecLeftTrail,  eRosRoot);
  sceneRefs.eeSecRightTrail = clearTrail(sceneRefs.eeSecRightTrail, eRosRoot);

  if (!robot || !S.frames.length) return;

  // For FK validation always use observation.state (we validate the conversion script)
  const ptsL = [], ptsR = [];
  for (const frame of S.frames) {
    const data = frame['observation.state'];
    if (!data) continue;
    applyRobotJoints(robot, buildValuesByName(data, K.STATE_IDX));
    const lPos = getEEWorldPos(robot, 'follower_left_ee_gripper_link');
    const rPos = getEEWorldPos(robot, 'follower_right_ee_gripper_link');
    if (lPos) ptsL.push(lPos.clone());
    if (rPos) ptsR.push(rPos.clone());
  }
  if (ptsL.length) { sceneRefs.eeSecLeftTrail  = makeLine(ptsL, K.C_SEC_L, 0.5); eRosRoot.add(sceneRefs.eeSecLeftTrail); }
  if (ptsR.length) { sceneRefs.eeSecRightTrail = makeLine(ptsR, K.C_SEC_R, 0.5); eRosRoot.add(sceneRefs.eeSecRightTrail); }

  // Restore current frame
  if (S.frames[S.frameIdx]) applyJoints(S.frames[S.frameIdx]);
}

// Build action.ee trajectories for Obs vs Action EE mode.
export function buildActionEETrajectories() {
  sceneRefs.eeSecLeftTrail  = clearTrail(sceneRefs.eeSecLeftTrail,  eRosRoot);
  sceneRefs.eeSecRightTrail = clearTrail(sceneRefs.eeSecRightTrail, eRosRoot);

  if (!S.hasActionEE) return;
  const ptsL = S.frames.map(f => f['action.ee_left'] ).filter(Boolean).map(worldPos);
  const ptsR = S.frames.map(f => f['action.ee_right']).filter(Boolean).map(worldPos);
  if (ptsL.length) { sceneRefs.eeSecLeftTrail  = makeLine(ptsL, K.C_SEC_L, 0.5); eRosRoot.add(sceneRefs.eeSecLeftTrail); }
  if (ptsR.length) { sceneRefs.eeSecRightTrail = makeLine(ptsR, K.C_SEC_R, 0.5); eRosRoot.add(sceneRefs.eeSecRightTrail); }
}

// Rebuild secondary trajectories for the current rightMode.
export function buildSecondaryTrajectories() {
  if (S.rightMode === 'fk') {
    buildFKTrajectories();
  } else if (S.rightMode === 'obs_action') {
    buildActionEETrajectories();
  } else {
    // delta / relative — no secondary trail needed
    sceneRefs.eeSecLeftTrail  = clearTrail(sceneRefs.eeSecLeftTrail,  eRosRoot);
    sceneRefs.eeSecRightTrail = clearTrail(sceneRefs.eeSecRightTrail, eRosRoot);
  }
}

// ── Per-frame EE update ───────────────────────────────────────────────────────
export function hideAllSecondary() {
  eeSecLeftMark.visible  = eeSecRightMark.visible  = false;
  errLeftLine.visible    = errRightLine.visible     = false;
  arrowLeft.visible      = arrowRight.visible       = false;
}

export function updateModeInfo(text) {
  const el = document.getElementById('modeInfo');
  if (el) el.textContent = text;
}

export function updateEE(frame) {
  const obsL = frame['observation.ee_left'];
  const obsR = frame['observation.ee_right'];

  // ── Left panel: small obs EE dots overlaid on URDF
  if (obsL && S.hasEELeft) { jObsLeftMark.position.copy(worldPos(obsL)); jObsLeftMark.visible = true; }
  if (obsR && S.hasEERight){ jObsRightMark.position.copy(worldPos(obsR));jObsRightMark.visible = true; }

  // ── Right panel: observation EE (always shown)
  if (obsL && S.hasEELeft) {
    applyPose(eeLeftFrame, obsL); eeLeftFrame.visible = true;
    eeObsLeftMark.position.copy(worldPos(obsL)); eeObsLeftMark.visible = true;
  }
  if (obsR && S.hasEERight) {
    applyPose(eeRightFrame, obsR); eeRightFrame.visible = true;
    eeObsRightMark.position.copy(worldPos(obsR)); eeObsRightMark.visible = true;
  }

  hideAllSecondary();

  // ── Right panel: mode-specific secondary objects
  if (S.rightMode === 'fk') {
    // Live FK from URDF (robot joints already applied by applyJoints)
    const fkL = robot ? getEEWorldPos(robot, 'follower_left_ee_gripper_link')  : null;
    const fkR = robot ? getEEWorldPos(robot, 'follower_right_ee_gripper_link') : null;
    if (fkL) { eeSecLeftMark.position.copy(fkL);  eeSecLeftMark.visible  = true; }
    if (fkR) { eeSecRightMark.position.copy(fkR); eeSecRightMark.visible = true; }
    if (obsL && fkL && S.hasEELeft)  { setLinePoints(errLeftLine,  worldPos(obsL), fkL); errLeftLine.visible  = true; }
    if (obsR && fkR && S.hasEERight) { setLinePoints(errRightLine, worldPos(obsR), fkR); errRightLine.visible = true; }

  } else if (S.rightMode === 'obs_action') {
    const actL = frame['action.ee_left'];
    const actR = frame['action.ee_right'];
    if (actL && S.hasActionEE) { eeSecLeftMark.position.copy(worldPos(actL));  eeSecLeftMark.visible  = true; }
    if (actR && S.hasActionEE) { eeSecRightMark.position.copy(worldPos(actR)); eeSecRightMark.visible = true; }
    if (obsL && actL && S.hasEELeft && S.hasActionEE)  { setLinePoints(errLeftLine,  worldPos(obsL), worldPos(actL)); errLeftLine.visible  = true; }
    if (obsR && actR && S.hasEERight && S.hasActionEE) { setLinePoints(errRightLine, worldPos(obsR), worldPos(actR)); errRightLine.visible = true; }

  } else if (S.rightMode === 'ee_delta') {
    const dL = frame['action.ee_left.delta'];
    const dR = frame['action.ee_right.delta'];
    let magL = 0, magR = 0;
    if (dL && obsL && S.hasEEDelta) {
      arrowLeft.setColor(K.C_DELTA);
      magL = updateArrow(arrowLeft,  worldPos(obsL), dL[0], dL[1], dL[2]);
    }
    if (dR && obsR && S.hasEEDelta) {
      arrowRight.setColor(K.C_DELTA);
      magR = updateArrow(arrowRight, worldPos(obsR), dR[0], dR[1], dR[2]);
    }
    updateModeInfo(`Δ EE L: ${(magL*1000).toFixed(1)} mm  |  Δ EE R: ${(magR*1000).toFixed(1)} mm`);

  } else if (S.rightMode === 'ee_relative') {
    const rL = frame['action.ee_left.relative'];
    const rR = frame['action.ee_right.relative'];
    let magL = 0, magR = 0;
    if (rL && obsL && S.hasEERel) {
      arrowLeft.setColor(K.C_REL);
      magL = updateArrow(arrowLeft,  worldPos(obsL), rL[0], rL[1], rL[2]);
    }
    if (rR && obsR && S.hasEERel) {
      arrowRight.setColor(K.C_REL);
      magR = updateArrow(arrowRight, worldPos(obsR), rR[0], rR[1], rR[2]);
    }
    updateModeInfo(`Gap L: ${(magL*1000).toFixed(1)} mm  |  Gap R: ${(magR*1000).toFixed(1)} mm`);
  }

  if (S.rightMode !== 'ee_delta' && S.rightMode !== 'ee_relative') updateModeInfo('');
}
