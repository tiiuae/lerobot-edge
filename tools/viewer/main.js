/**
 * Robot Dataset Viewer — main.js
 *
 * LEFT panel  — joint space (observation.state or action drives the URDF).
 *   Toggle via [State] / [Action] buttons.
 *
 * RIGHT panel — four selectable modes:
 *   FK Validation   obs.ee (blue/red) vs live FK from URDF (cyan/orange).
 *                   Yellow line = FK error. Use to validate joint_to_ee.py.
 *   Obs vs Action   obs.ee (blue/red) vs action.ee (cyan/orange).
 *                   Yellow line = tracking gap.
 *   EE Δ            obs.ee (blue/red) + green arrows = action.ee.delta direction.
 *   EE Relative     obs.ee (blue/red) + purple arrows = action.ee - obs.ee vector.
 *
 * Coordinate convention: ROS z-up world. camera.up = (0,0,1).
 */

import * as THREE        from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ColladaLoader }  from 'three/examples/jsm/loaders/ColladaLoader.js';
import URDFLoader         from 'urdf-loader';

// ── Global playback state ─────────────────────────────────────────────────────
let frames   = [];
let frameIdx = 0;
let playing  = false;
let playTimer = null;

// ── Mode state ────────────────────────────────────────────────────────────────
let leftMode  = 'state';       // 'state' | 'action'
let rightMode = 'fk';          // 'fk' | 'obs_action' | 'ee_delta' | 'ee_relative'

// ── Per-dataset feature flags ─────────────────────────────────────────────────
let hasEELeft    = false;
let hasEERight   = false;
let hasActionEE  = false;   // action.ee_left/right columns present
let hasEEDelta   = false;   // action.ee_left.delta columns present
let hasEERel     = false;   // action.ee_left.relative columns present

// ── Joint name mapping: dataset state name → URDF joint name ──────────────────
const JOINT_MAP = {
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
const GRIPPER_MAP = {
  left_joint_6:  ['follower_left_right_carriage_joint',  'follower_left_left_carriage_joint'],
  right_joint_6: ['follower_right_right_carriage_joint', 'follower_right_left_carriage_joint'],
};

// ── Joint calibration ─────────────────────────────────────────────────────────
const CALIB_KEY = 'lerobot-viewer-joint-calibration-v3';
const SIDES = ['left', 'right'];

function defaultCalib() {
  const c = {};
  for (const side of SIDES)
    for (let i = 0; i <= 6; i++)
      c[`${side}_joint_${i}`] = { offset: 0, sign: 1 };
  return c;
}
function loadJointCalib() {
  try {
    const s = localStorage.getItem(CALIB_KEY);
    if (s) return { ...defaultCalib(), ...JSON.parse(s) };
  } catch (e) { console.warn('calib load failed', e); }
  return defaultCalib();
}
function saveJointCalib() {
  try { localStorage.setItem(CALIB_KEY, JSON.stringify(jointCalib)); }
  catch (e) { console.warn('calib save failed', e); }
}
let jointCalib = loadJointCalib();

function transformJointValue(dsName, raw) {
  const c = jointCalib[dsName];
  return c ? c.sign * raw + c.offset : raw;
}

// Arms-first layout — metadata names are mislabeled, use fixed indices instead.
//   state/action [0..6]  = left  arm joints 0..6  (joint_6 = gripper)
//   state/action [7..13] = right arm joints 0..6
//   state  [14..18]      = base info
//   action [14..15]      = linear_vel, angular_vel
const STATE_IDX = {
  left_joint_0: 0,  left_joint_1: 1,  left_joint_2: 2,  left_joint_3: 3,
  left_joint_4: 4,  left_joint_5: 5,  left_joint_6: 6,
  right_joint_0: 7, right_joint_1: 8, right_joint_2: 9, right_joint_3: 10,
  right_joint_4: 11, right_joint_5: 12, right_joint_6: 13,
};

const LEFT_MOUNT  = [0.331,  0.3, 0.831];
const RIGHT_MOUNT = [0.331, -0.3, 0.831];

// ── Colour palette ────────────────────────────────────────────────────────────
const BG     = 0xf0f2f5;
const GRID1  = 0x999999;
const GRID2  = 0xcccccc;

const C_OBS_L   = 0x2266cc;   // observation EE left  — blue
const C_OBS_R   = 0xcc2211;   // observation EE right — red
const C_SEC_L   = 0x00bbdd;   // secondary left  (FK / action EE) — cyan
const C_SEC_R   = 0xdd7700;   // secondary right (FK / action EE) — orange
const C_ERR     = 0xffdd00;   // error / gap line  — yellow
const C_DELTA   = 0x22cc55;   // delta arrow       — green
const C_REL     = 0xaa44ff;   // relative arrow    — purple

// ── Shared helpers ────────────────────────────────────────────────────────────
function makeFloor(scene) {
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(10, 10),
    new THREE.ShadowMaterial({ opacity: 0.15 })
  );
  floor.receiveShadow = true;
  scene.add(floor);
  const grid = new THREE.GridHelper(6, 30, GRID1, GRID2);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);
}

function makeCamera(canvas, pos, target) {
  const panel = canvas.parentElement;
  const cam   = new THREE.PerspectiveCamera(45, panel.clientWidth / panel.clientHeight, 0.01, 50);
  cam.position.set(...pos);
  cam.up.set(0, 0, 1);
  const ctrl = new OrbitControls(cam, canvas);
  ctrl.target.set(...target);
  ctrl.enableDamping = true;
  ctrl.dampingFactor = 0.08;
  ctrl.update();
  return { cam, ctrl };
}

function makeLights(scene) {
  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  scene.add(new THREE.HemisphereLight(0x88aacc, 0xaa9977, 0.5));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.position.set(3, 2, 5);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.camera.near  = 0.1;
  sun.shadow.camera.far   = 20;
  sun.shadow.camera.left  = sun.shadow.camera.bottom = -3;
  sun.shadow.camera.right = sun.shadow.camera.top    =  3;
  scene.add(sun);
}

function makeURDFLoader() {
  const loader = new URDFLoader();
  loader.packages = { trossen_arm_description: '/pkg/trossen_arm_description' };
  loader.loadMeshCb = (path, manager, done) => {
    if (/\.dae$/i.test(path)) {
      const cl = new ColladaLoader(manager);
      cl.load(path, dae => {
        dae.scene.rotation.set(0, 0, 0);
        dae.scene.updateMatrixWorld(true);
        done(dae.scene);
      }, null, err => done(null, err));
    } else {
      loader.defaultMeshLoader(path, manager, done);
    }
  };
  return loader;
}

function makeAxisFrame(size, colors) {
  const g = new THREE.Group();
  const dirs = [new THREE.Vector3(1,0,0), new THREE.Vector3(0,1,0), new THREE.Vector3(0,0,1)];
  dirs.forEach((d, i) =>
    g.add(new THREE.ArrowHelper(d, new THREE.Vector3(), size, colors[i], size*0.28, size*0.14)));
  return g;
}

function makeLine(points, color, opacity) {
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color, opacity, transparent: opacity < 1 })
  );
}

function makeUpdatableLine(color) {
  const buf = new Float32Array(6);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(buf, 3));
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color }));
  line.visible = false;
  return line;
}

function setLinePoints(line, a, b) {
  const arr = line.geometry.attributes.position.array;
  arr[0]=a.x; arr[1]=a.y; arr[2]=a.z;
  arr[3]=b.x; arr[4]=b.y; arr[5]=b.z;
  line.geometry.attributes.position.needsUpdate = true;
}

// pose7 = [x, y, z, qw, qx, qy, qz]
function applyPose(obj, pose7) {
  const [x, y, z, qw, qx, qy, qz] = pose7;
  obj.position.set(x, y, z);
  obj.quaternion.set(qx, qy, qz, qw);
}

function posV3(pose7) {
  return new THREE.Vector3(pose7[0], pose7[1], pose7[2]);
}

// Update an ArrowHelper to point from origin in direction of delta_xyz.
// Returns magnitude (metres). Hides arrow if magnitude is negligible.
const ARROW_SCALE = 8.0;   // visual scale factor — delta in metres, arrows in scene units
function updateArrow(arrow, originV3, dx, dy, dz) {
  const mag = Math.sqrt(dx*dx + dy*dy + dz*dz);
  if (mag < 1e-6) { arrow.visible = false; return 0; }
  arrow.position.copy(originV3);
  arrow.setDirection(new THREE.Vector3(dx/mag, dy/mag, dz/mag));
  const len = Math.max(mag * ARROW_SCALE, 0.02);
  arrow.setLength(len, Math.min(0.04, len * 0.25), Math.min(0.02, len * 0.12));
  arrow.visible = true;
  return mag;
}

// ── LEFT PANEL ────────────────────────────────────────────────────────────────
let jRenderer, jScene, jCamera, jControls, jRosRoot;
let robot = null;
let jObsLeftMark, jObsRightMark;
let jObsLeftTrail, jObsRightTrail;

function setupJointScene() {
  const canvas = document.getElementById('jointCanvas');
  jRenderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  jRenderer.setPixelRatio(window.devicePixelRatio);
  jRenderer.shadowMap.enabled = true;
  jRenderer.shadowMap.type    = THREE.PCFSoftShadowMap;

  jScene = new THREE.Scene();
  jScene.background = new THREE.Color(BG);
  jScene.fog = new THREE.Fog(BG, 10, 25);

  jRosRoot = new THREE.Group();
  jScene.add(jRosRoot);
  makeLights(jScene);
  makeFloor(jScene);

  ({ cam: jCamera, ctrl: jControls } = makeCamera(canvas, [3.0, -3.5, 2.0], [0.0, 0.0, 1.0]));

  const sGeo = new THREE.SphereGeometry(0.016, 10, 10);
  jObsLeftMark  = new THREE.Mesh(sGeo, new THREE.MeshPhongMaterial({ color: C_OBS_L, emissive: 0x112244 }));
  jObsRightMark = new THREE.Mesh(sGeo, new THREE.MeshPhongMaterial({ color: C_OBS_R, emissive: 0x441108 }));
  jRosRoot.add(jObsLeftMark, jObsRightMark);
  jObsLeftMark.visible = jObsRightMark.visible = false;
}

// ── RIGHT PANEL ───────────────────────────────────────────────────────────────
let eRenderer, eScene, eCamera, eControls, eRosRoot;

// Observation EE (blue/red) — always shown when available
let eeLeftFrame, eeRightFrame;
let eeObsLeftMark, eeObsRightMark;
let eeObsLeftTrail, eeObsRightTrail;

// Secondary EE (cyan/orange) — FK EE in fk mode, action.ee in obs_action mode
let eeSecLeftMark, eeSecRightMark;
let eeSecLeftTrail, eeSecRightTrail;

// Gap/error lines (yellow)
let errLeftLine, errRightLine;

// Delta / relative arrows (green / purple)
let arrowLeft, arrowRight;

function setupEEScene() {
  const canvas = document.getElementById('eeCanvas');
  eRenderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  eRenderer.setPixelRatio(window.devicePixelRatio);
  eRenderer.shadowMap.enabled = true;
  eRenderer.shadowMap.type    = THREE.PCFSoftShadowMap;

  eScene = new THREE.Scene();
  eScene.background = new THREE.Color(BG);
  eScene.fog = new THREE.Fog(BG, 10, 25);

  eRosRoot = new THREE.Group();
  eScene.add(eRosRoot);
  makeLights(eScene);
  makeFloor(eScene);

  // World origin axes + arm mount markers
  eRosRoot.add(new THREE.AxesHelper(0.2));
  const mountGeo = new THREE.SphereGeometry(0.022, 10, 10);
  const mountMat = new THREE.MeshPhongMaterial({ color: 0x334455, emissive: 0x111a22 });
  const lMount = new THREE.Mesh(mountGeo, mountMat.clone());
  const rMount = new THREE.Mesh(mountGeo, mountMat.clone());
  lMount.position.set(...LEFT_MOUNT);
  rMount.position.set(...RIGHT_MOUNT);
  eRosRoot.add(lMount, rMount);

  // Observation EE — axis frames
  eeLeftFrame  = makeAxisFrame(0.09, [0x2266cc, 0x0099cc, 0x6633cc]);
  eeRightFrame = makeAxisFrame(0.09, [0xcc2211, 0xcc6611, 0xcc2299]);
  eRosRoot.add(eeLeftFrame, eeRightFrame);
  eeLeftFrame.visible = eeRightFrame.visible = false;

  // Observation EE — solid spheres (primary)
  const sObs = new THREE.SphereGeometry(0.020, 12, 12);
  eeObsLeftMark  = new THREE.Mesh(sObs, new THREE.MeshPhongMaterial({ color: C_OBS_L, emissive: 0x112244 }));
  eeObsRightMark = new THREE.Mesh(sObs, new THREE.MeshPhongMaterial({ color: C_OBS_R, emissive: 0x441108 }));
  eRosRoot.add(eeObsLeftMark, eeObsRightMark);
  eeObsLeftMark.visible = eeObsRightMark.visible = false;

  // Secondary EE — smaller spheres (FK EE or action.ee depending on mode)
  const sSec = new THREE.SphereGeometry(0.014, 10, 10);
  eeSecLeftMark  = new THREE.Mesh(sSec, new THREE.MeshPhongMaterial({ color: C_SEC_L, emissive: 0x003344 }));
  eeSecRightMark = new THREE.Mesh(sSec, new THREE.MeshPhongMaterial({ color: C_SEC_R, emissive: 0x331100 }));
  eRosRoot.add(eeSecLeftMark, eeSecRightMark);
  eeSecLeftMark.visible = eeSecRightMark.visible = false;

  // Gap / error lines
  errLeftLine  = makeUpdatableLine(C_ERR);
  errRightLine = makeUpdatableLine(C_ERR);
  eRosRoot.add(errLeftLine, errRightLine);

  // Delta / Relative arrow helpers
  const fwd = new THREE.Vector3(1, 0, 0);
  const org = new THREE.Vector3(0, 0, 0);
  arrowLeft  = new THREE.ArrowHelper(fwd, org, 0.1, C_DELTA, 0.04, 0.02);
  arrowRight = new THREE.ArrowHelper(fwd, org, 0.1, C_DELTA, 0.04, 0.02);
  arrowLeft.visible = arrowRight.visible = false;
  eRosRoot.add(arrowLeft, arrowRight);

  ({ cam: eCamera, ctrl: eControls } = makeCamera(canvas, [3.0, -3.5, 2.0], [0.35, 0.0, 1.0]));
}

// ── URDF loading ──────────────────────────────────────────────────────────────
async function loadRobot() {
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
}

// ── Joint application ─────────────────────────────────────────────────────────
function buildValuesByName(data, fallbackIdxMap) {
  const byName = {};
  for (const [name, idx] of Object.entries(fallbackIdxMap))
    if (idx < data.length) byName[name] = data[idx];
  return byName;
}

function applyRobotJoints(rb, byName) {
  for (const [dsName, urdfName] of Object.entries(JOINT_MAP))
    if (dsName in byName) rb.setJointValue(urdfName, transformJointValue(dsName, byName[dsName]));
  for (const [dsName, urdfNames] of Object.entries(GRIPPER_MAP))
    if (dsName in byName) {
      const v = transformJointValue(dsName, byName[dsName]);
      for (const u of urdfNames) rb.setJointValue(u, v);
    }
}

function getEEWorldPos(robotObj, linkName) {
  robotObj.updateMatrixWorld(true);
  const link = robotObj.getObjectByName(linkName);
  return link ? new THREE.Vector3().setFromMatrixPosition(link.matrixWorld) : null;
}

// Drive the URDF from observation.state (State mode) or action (Action mode).
function applyJoints(frame) {
  if (!robot) return;
  const data = leftMode === 'state' ? frame['observation.state'] : frame['action'];
  if (!data) return;
  const byName = buildValuesByName(data, STATE_IDX);
  applyRobotJoints(robot, byName);
}

// ── Trajectory builders ───────────────────────────────────────────────────────
function clearTrail(trail, root) {
  if (trail) root.remove(trail);
  return null;
}

function buildObsTrajectories() {
  jObsLeftTrail  = clearTrail(jObsLeftTrail,  jRosRoot);
  jObsRightTrail = clearTrail(jObsRightTrail, jRosRoot);
  eeObsLeftTrail = clearTrail(eeObsLeftTrail, eRosRoot);
  eeObsRightTrail= clearTrail(eeObsRightTrail,eRosRoot);

  if (hasEELeft) {
    const pts = frames.map(f => f['observation.ee_left']).filter(Boolean)
      .map(p => new THREE.Vector3(p[0], p[1], p[2]));
    eeObsLeftTrail = makeLine(pts, C_OBS_L, 0.6); eRosRoot.add(eeObsLeftTrail);
    jObsLeftTrail  = makeLine(pts, C_OBS_L, 0.6); jRosRoot.add(jObsLeftTrail);
  }
  if (hasEERight) {
    const pts = frames.map(f => f['observation.ee_right']).filter(Boolean)
      .map(p => new THREE.Vector3(p[0], p[1], p[2]));
    eeObsRightTrail = makeLine(pts, C_OBS_R, 0.6); eRosRoot.add(eeObsRightTrail);
    jObsRightTrail  = makeLine(pts, C_OBS_R, 0.6); jRosRoot.add(jObsRightTrail);
  }
}

// Precompute FK EE trajectory from URDF (using leftMode data source).
// Used only in FK Validation mode.
function buildFKTrajectories() {
  eeSecLeftTrail  = clearTrail(eeSecLeftTrail,  eRosRoot);
  eeSecRightTrail = clearTrail(eeSecRightTrail, eRosRoot);

  if (!robot || !frames.length) return;

  // For FK validation always use observation.state (we validate the conversion script)
  const ptsL = [], ptsR = [];
  for (const frame of frames) {
    const data = frame['observation.state'];
    if (!data) continue;
    applyRobotJoints(robot, buildValuesByName(data, STATE_IDX));
    const lPos = getEEWorldPos(robot, 'follower_left_ee_gripper_link');
    const rPos = getEEWorldPos(robot, 'follower_right_ee_gripper_link');
    if (lPos) ptsL.push(lPos.clone());
    if (rPos) ptsR.push(rPos.clone());
  }
  if (ptsL.length) { eeSecLeftTrail  = makeLine(ptsL, C_SEC_L, 0.5); eRosRoot.add(eeSecLeftTrail); }
  if (ptsR.length) { eeSecRightTrail = makeLine(ptsR, C_SEC_R, 0.5); eRosRoot.add(eeSecRightTrail); }

  // Restore current frame
  if (frames[frameIdx]) applyJoints(frames[frameIdx]);
}

// Build action.ee trajectories for Obs vs Action EE mode.
function buildActionEETrajectories() {
  eeSecLeftTrail  = clearTrail(eeSecLeftTrail,  eRosRoot);
  eeSecRightTrail = clearTrail(eeSecRightTrail, eRosRoot);

  if (!hasActionEE) return;
  const ptsL = frames.map(f => f['action.ee_left'] ).filter(Boolean).map(p => new THREE.Vector3(p[0], p[1], p[2]));
  const ptsR = frames.map(f => f['action.ee_right']).filter(Boolean).map(p => new THREE.Vector3(p[0], p[1], p[2]));
  if (ptsL.length) { eeSecLeftTrail  = makeLine(ptsL, C_SEC_L, 0.5); eRosRoot.add(eeSecLeftTrail); }
  if (ptsR.length) { eeSecRightTrail = makeLine(ptsR, C_SEC_R, 0.5); eRosRoot.add(eeSecRightTrail); }
}

// Rebuild secondary trajectories for the current rightMode.
function buildSecondaryTrajectories() {
  if (rightMode === 'fk') {
    buildFKTrajectories();
  } else if (rightMode === 'obs_action') {
    buildActionEETrajectories();
  } else {
    // delta / relative — no secondary trail needed
    eeSecLeftTrail  = clearTrail(eeSecLeftTrail,  eRosRoot);
    eeSecRightTrail = clearTrail(eeSecRightTrail, eRosRoot);
  }
}

// ── Per-frame EE update ───────────────────────────────────────────────────────
function hideAllSecondary() {
  eeSecLeftMark.visible  = eeSecRightMark.visible  = false;
  errLeftLine.visible    = errRightLine.visible     = false;
  arrowLeft.visible      = arrowRight.visible       = false;
}

function updateEE(frame) {
  const obsL = frame['observation.ee_left'];
  const obsR = frame['observation.ee_right'];

  // ── Left panel: small obs EE dots overlaid on URDF
  if (obsL && hasEELeft) { jObsLeftMark.position.set(obsL[0], obsL[1], obsL[2]); jObsLeftMark.visible = true; }
  if (obsR && hasEERight){ jObsRightMark.position.set(obsR[0], obsR[1], obsR[2]);jObsRightMark.visible = true; }

  // ── Right panel: observation EE (always shown)
  if (obsL && hasEELeft) {
    applyPose(eeLeftFrame, obsL); eeLeftFrame.visible = true;
    eeObsLeftMark.position.set(obsL[0], obsL[1], obsL[2]); eeObsLeftMark.visible = true;
  }
  if (obsR && hasEERight) {
    applyPose(eeRightFrame, obsR); eeRightFrame.visible = true;
    eeObsRightMark.position.set(obsR[0], obsR[1], obsR[2]); eeObsRightMark.visible = true;
  }

  hideAllSecondary();

  // ── Right panel: mode-specific secondary objects
  if (rightMode === 'fk') {
    // Live FK from URDF (robot joints already applied by applyJoints)
    const fkL = robot ? getEEWorldPos(robot, 'follower_left_ee_gripper_link')  : null;
    const fkR = robot ? getEEWorldPos(robot, 'follower_right_ee_gripper_link') : null;
    if (fkL) { eeSecLeftMark.position.copy(fkL);  eeSecLeftMark.visible  = true; }
    if (fkR) { eeSecRightMark.position.copy(fkR); eeSecRightMark.visible = true; }
    if (obsL && fkL && hasEELeft)  { setLinePoints(errLeftLine,  posV3(obsL), fkL); errLeftLine.visible  = true; }
    if (obsR && fkR && hasEERight) { setLinePoints(errRightLine, posV3(obsR), fkR); errRightLine.visible = true; }

  } else if (rightMode === 'obs_action') {
    const actL = frame['action.ee_left'];
    const actR = frame['action.ee_right'];
    if (actL && hasActionEE) { eeSecLeftMark.position.set(actL[0], actL[1], actL[2]);  eeSecLeftMark.visible  = true; }
    if (actR && hasActionEE) { eeSecRightMark.position.set(actR[0], actR[1], actR[2]); eeSecRightMark.visible = true; }
    if (obsL && actL && hasEELeft && hasActionEE)  { setLinePoints(errLeftLine,  posV3(obsL), new THREE.Vector3(actL[0], actL[1], actL[2])); errLeftLine.visible  = true; }
    if (obsR && actR && hasEERight && hasActionEE) { setLinePoints(errRightLine, posV3(obsR), new THREE.Vector3(actR[0], actR[1], actR[2])); errRightLine.visible = true; }

  } else if (rightMode === 'ee_delta') {
    const dL = frame['action.ee_left.delta'];
    const dR = frame['action.ee_right.delta'];
    let magL = 0, magR = 0;
    if (dL && obsL && hasEEDelta) {
      arrowLeft.setColor(C_DELTA);
      magL = updateArrow(arrowLeft,  posV3(obsL), dL[0], dL[1], dL[2]);
    }
    if (dR && obsR && hasEEDelta) {
      arrowRight.setColor(C_DELTA);
      magR = updateArrow(arrowRight, posV3(obsR), dR[0], dR[1], dR[2]);
    }
    updateModeInfo(`Δ EE L: ${(magL*1000).toFixed(1)} mm  |  Δ EE R: ${(magR*1000).toFixed(1)} mm`);

  } else if (rightMode === 'ee_relative') {
    const rL = frame['action.ee_left.relative'];
    const rR = frame['action.ee_right.relative'];
    let magL = 0, magR = 0;
    if (rL && obsL && hasEERel) {
      arrowLeft.setColor(C_REL);
      magL = updateArrow(arrowLeft,  posV3(obsL), rL[0], rL[1], rL[2]);
    }
    if (rR && obsR && hasEERel) {
      arrowRight.setColor(C_REL);
      magR = updateArrow(arrowRight, posV3(obsR), rR[0], rR[1], rR[2]);
    }
    updateModeInfo(`Gap L: ${(magL*1000).toFixed(1)} mm  |  Gap R: ${(magR*1000).toFixed(1)} mm`);
  }

  if (rightMode !== 'ee_delta' && rightMode !== 'ee_relative') updateModeInfo('');
}

function updateModeInfo(text) {
  const el = document.getElementById('modeInfo');
  if (el) el.textContent = text;
}

// ── Mode switching ────────────────────────────────────────────────────────────
function setLeftMode(mode) {
  leftMode = mode;
  document.querySelectorAll('.left-mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
  // Rebuild FK trails when switching (FK validation is still state-based, but
  // left panel trajectory label changes)
  buildSecondaryTrajectories();
  if (frames[frameIdx]) updateFrame(frameIdx);
}

function setRightMode(mode) {
  rightMode = mode;
  document.querySelectorAll('.right-mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
  updateModeLegend();
  buildSecondaryTrajectories();
  if (frames[frameIdx]) updateFrame(frameIdx);
}

function updateModeLegend() {
  const legend = document.getElementById('eeLegend');
  if (!legend) return;
  const items = {
    fk: `
      <span class="dot obs-l"></span>Obs EE L
      <span class="dot obs-r"></span>Obs EE R
      <span class="dot sec-l"></span>FK EE L
      <span class="dot sec-r"></span>FK EE R
      <span class="dot err"></span>FK Error`,
    obs_action: `
      <span class="dot obs-l"></span>Obs EE L
      <span class="dot obs-r"></span>Obs EE R
      <span class="dot sec-l"></span>Action EE L
      <span class="dot sec-r"></span>Action EE R
      <span class="dot err"></span>Gap`,
    ee_delta: `
      <span class="dot obs-l"></span>Obs EE L
      <span class="dot obs-r"></span>Obs EE R
      <span class="dot delta"></span>Δ direction (left)
      <span class="dot delta"></span>Δ direction (right)`,
    ee_relative: `
      <span class="dot obs-l"></span>Obs EE L
      <span class="dot obs-r"></span>Obs EE R
      <span class="dot rel"></span>Gap vector (left)
      <span class="dot rel"></span>Gap vector (right)`,
  };
  legend.innerHTML = items[rightMode] ?? '';
}

// Update which mode buttons are unavailable based on dataset features.
// We use a CSS class instead of the HTML disabled attribute so pointer events
// still fire and we can show a helpful status message on click.
function updateModeButtonAvailability() {
  document.querySelectorAll('.right-mode-btn').forEach(b => {
    let unavailable = false;
    if (b.dataset.mode === 'obs_action'  && !hasActionEE) unavailable = true;
    if (b.dataset.mode === 'ee_delta'    && !hasEEDelta)  unavailable = true;
    if (b.dataset.mode === 'ee_relative' && !hasEERel)    unavailable = true;
    b.classList.toggle('unavailable', unavailable);
    b.title = unavailable
      ? 'Not available for this dataset — re-run joint_to_ee.py with --include-action'
      : '';
    // Fall back to fk if current mode becomes unavailable
    if (unavailable && rightMode === b.dataset.mode) setRightMode('fk');
  });
  document.querySelectorAll('.left-mode-btn').forEach(b =>
    b.classList.remove('unavailable'));
}

// ── Dataset / Episode loading ─────────────────────────────────────────────────
async function loadDatasets() {
  const list = await fetch('/api/datasets').then(r => r.json());
  const sel  = document.getElementById('datasetSelect');
  if (!list.length) {
    sel.innerHTML = '<option disabled>No datasets found</option>';
    setStatus('No datasets found in cache.');
    return;
  }
  sel.innerHTML = list
    .map(d => `<option value="${d.path}" data-info='${JSON.stringify(d)}'>${d.name}  (${d.total_episodes} ep, ${d.total_frames} frames)</option>`)
    .join('');
  sel.addEventListener('change', onDatasetChange);
  await onDatasetChange();
}

async function onDatasetChange() {
  const opt  = document.getElementById('datasetSelect').selectedOptions[0];
  if (!opt) return;
  const info = JSON.parse(opt.dataset.info);

  hasEELeft   = info.has_ee_left    ?? false;
  hasEERight  = info.has_ee_right   ?? false;
  hasActionEE = info.has_action_ee  ?? false;
  hasEEDelta  = info.has_ee_delta   ?? false;
  hasEERel    = info.has_ee_relative?? false;

  updateModeButtonAvailability();

  const episodes = await fetch(`/api/episodes?dataset=${encodeURIComponent(opt.value)}`).then(r => r.json());
  const epSel    = document.getElementById('episodeSelect');
  epSel.innerHTML = episodes
    .map(e => `<option value="${e.episode}">Ep ${e.episode}  (${e.frames} frames)</option>`)
    .join('');
  epSel.onchange = () => loadEpisode(opt.value, +epSel.value);
  await loadEpisode(opt.value, episodes[0]?.episode ?? 0);
}

async function loadEpisode(datasetPath, epIdx) {
  stopPlayback();
  setStatus('Loading episode…');

  const data = await fetch(
    `/api/frames?dataset=${encodeURIComponent(datasetPath)}&episode=${epIdx}`
  ).then(r => r.json());

  frames   = data.frames ?? [];
  frameIdx = 0;

  const slider = document.getElementById('frameSlider');
  slider.max   = Math.max(0, frames.length - 1);
  slider.value = 0;

  buildObsTrajectories();
  buildSecondaryTrajectories();
  updateFrame(0);
  setStatus(`${frames.length} frames at 50 fps  ·  ${(frames.length / 50).toFixed(1)} s`);
}

// ── Playback ──────────────────────────────────────────────────────────────────
function updateFrame(idx) {
  frameIdx = Math.max(0, Math.min(idx, frames.length - 1));
  document.getElementById('frameSlider').value = frameIdx;
  document.getElementById('frameCounter').textContent = `${frameIdx + 1} / ${frames.length}`;
  const f = frames[frameIdx];
  if (!f) return;
  applyJoints(f);
  updateEE(f);
}

function startPlayback() {
  if (playing || !frames.length) return;
  playing = true;
  document.getElementById('playBtn').textContent = '⏸ Pause';
  tick();
}
function stopPlayback() {
  playing = false;
  if (playTimer) { clearTimeout(playTimer); playTimer = null; }
  document.getElementById('playBtn').textContent = '▶ Play';
}
function tick() {
  if (!playing) return;
  frameIdx = (frameIdx + 1) % frames.length;
  updateFrame(frameIdx);
  const speed = parseFloat(document.getElementById('speedSelect').value);
  playTimer = setTimeout(tick, 1000 / (50 * speed));
}

function setStatus(msg) {
  document.getElementById('statusMsg').textContent = msg;
}

// ── Camera sync ───────────────────────────────────────────────────────────────
let primaryCtrl = 'j';

function setupCameraSync() {
  document.getElementById('jointCanvas').addEventListener('pointerdown', () => { primaryCtrl = 'j'; });
  document.getElementById('eeCanvas').addEventListener('pointerdown',   () => { primaryCtrl = 'e'; });
}

function syncCameras() {
  if (primaryCtrl === 'j') {
    eCamera.position.copy(jCamera.position);
    eCamera.quaternion.copy(jCamera.quaternion);
    eControls.target.copy(jControls.target);
  } else {
    jCamera.position.copy(eCamera.position);
    jCamera.quaternion.copy(eCamera.quaternion);
    jControls.target.copy(eControls.target);
  }
}

// ── Render loop ───────────────────────────────────────────────────────────────
function syncSize(renderer, camera, canvas) {
  const { clientWidth: w, clientHeight: h } = canvas.parentElement;
  if (renderer.domElement.width !== w || renderer.domElement.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

function animate() {
  requestAnimationFrame(animate);
  if (primaryCtrl === 'j') {
    jControls.update(); syncCameras(); eControls.update();
  } else {
    eControls.update(); syncCameras(); jControls.update();
  }
  syncSize(jRenderer, jCamera, document.getElementById('jointCanvas'));
  syncSize(eRenderer, eCamera, document.getElementById('eeCanvas'));
  jRenderer.render(jScene, jCamera);
  eRenderer.render(eScene, eCamera);
}

// ── Joint calibration UI ──────────────────────────────────────────────────────
function setupCalibrationUI() {
  const container = document.getElementById('calibRows');
  const panel     = document.getElementById('calibPanel');

  document.getElementById('calibBtn').addEventListener('click',   () => panel.classList.toggle('hidden'));
  document.getElementById('calibClose').addEventListener('click', () => panel.classList.add('hidden'));
  document.getElementById('calibReset').addEventListener('click', () => {
    jointCalib = defaultCalib();
    saveJointCalib();
    rebuildCalibRows();
    onCalibChange();
  });

  function snapCurrentFrameToZero(side) {
    const frame = frames[frameIdx];
    if (!frame) return;
    const state = frame['observation.state'];
    if (!state) return;
    const byName = buildValuesByName(state, STATE_IDX);
    for (let i = 0; i <= 6; i++) {
      const key = `${side}_joint_${i}`;
      if (key in byName) {
        const c = jointCalib[key];
        c.offset = -c.sign * byName[key];
      }
    }
    saveJointCalib();
    rebuildCalibRows();
    onCalibChange();
  }

  function resetArm(side) {
    for (let i = 0; i <= 6; i++)
      jointCalib[`${side}_joint_${i}`] = { offset: 0, sign: 1 };
    saveJointCalib();
    rebuildCalibRows();
    onCalibChange();
  }

  function makeArmSection(side) {
    const section = document.createElement('div');
    section.className = 'calib-section';
    const header = document.createElement('div');
    header.className = 'calib-section-header';
    header.innerHTML = `<span>${side.toUpperCase()} ARM</span>`;
    section.appendChild(header);
    for (let i = 0; i <= 6; i++) {
      const key = `${side}_joint_${i}`;
      const c   = jointCalib[key];
      const row = document.createElement('div');
      row.className = 'calib-row';
      const label = i === 6 ? 'gripper' : `joint_${i}`;
      row.innerHTML = `
        <span class="cal-label">${label}</span>
        <button class="sign-btn" data-key="${key}">${c.sign > 0 ? '+' : '−'}</button>
        <input class="cal-slider" type="range" min="-3.14159" max="3.14159" step="0.01" value="${c.offset}" data-key="${key}">
        <input class="cal-num" type="number" step="0.01" value="${c.offset.toFixed(3)}" data-key="${key}">
      `;
      section.appendChild(row);
    }
    const actions = document.createElement('div');
    actions.className = 'calib-arm-actions';
    actions.innerHTML = `
      <button class="snap-btn"   data-side="${side}">Snap current → zero</button>
      <button class="armrst-btn" data-side="${side}">Reset</button>
    `;
    section.appendChild(actions);
    return section;
  }

  function rebuildCalibRows() {
    container.innerHTML = '';
    container.appendChild(makeArmSection('left'));
    container.appendChild(makeArmSection('right'));

    container.querySelectorAll('.sign-btn').forEach(b => b.addEventListener('click', e => {
      const key = e.target.dataset.key;
      jointCalib[key].sign = -jointCalib[key].sign;
      e.target.textContent = jointCalib[key].sign > 0 ? '+' : '−';
      saveJointCalib(); onCalibChange();
    }));
    container.querySelectorAll('.cal-slider').forEach(s => s.addEventListener('input', e => {
      const key = e.target.dataset.key;
      jointCalib[key].offset = +e.target.value;
      container.querySelector(`.cal-num[data-key="${key}"]`).value = (+e.target.value).toFixed(3);
      saveJointCalib(); onCalibChange();
    }));
    container.querySelectorAll('.cal-num').forEach(n => n.addEventListener('change', e => {
      const key = e.target.dataset.key;
      jointCalib[key].offset = +e.target.value;
      container.querySelector(`.cal-slider[data-key="${key}"]`).value = +e.target.value;
      saveJointCalib(); onCalibChange();
    }));
    container.querySelectorAll('.snap-btn').forEach(b =>
      b.addEventListener('click', e => snapCurrentFrameToZero(e.target.dataset.side)));
    container.querySelectorAll('.armrst-btn').forEach(b =>
      b.addEventListener('click', e => resetArm(e.target.dataset.side)));
  }

  rebuildCalibRows();
}

let calibChangeTimer = null;
function onCalibChange() {
  if (calibChangeTimer) clearTimeout(calibChangeTimer);
  if (frames[frameIdx]) updateFrame(frameIdx);
  calibChangeTimer = setTimeout(() => {
    buildSecondaryTrajectories();
    calibChangeTimer = null;
  }, 80);
}

// ── Mode button setup ─────────────────────────────────────────────────────────
function setupModeButtons() {
  document.querySelectorAll('.left-mode-btn').forEach(b =>
    b.addEventListener('click', () => setLeftMode(b.dataset.mode)));

  document.querySelectorAll('.right-mode-btn').forEach(b =>
    b.addEventListener('click', () => {
      if (b.classList.contains('unavailable')) {
        setStatus('Mode unavailable — re-run joint_to_ee.py with --include-action to enrich this dataset');
        return;
      }
      setRightMode(b.dataset.mode);
    }));

  // Set initial active state
  document.querySelectorAll('.left-mode-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.mode === leftMode));
  document.querySelectorAll('.right-mode-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.mode === rightMode));

  updateModeLegend();
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  setupJointScene();
  setupEEScene();
  setupCameraSync();
  setupModeButtons();
  setupCalibrationUI();
  animate();

  document.getElementById('playBtn').addEventListener('click', () =>
    playing ? stopPlayback() : startPlayback());
  document.getElementById('frameSlider').addEventListener('input', e => {
    stopPlayback();
    updateFrame(+e.target.value);
  });

  const helpOverlay = document.getElementById('helpOverlay');
  document.getElementById('helpBtn').addEventListener('click',   () => helpOverlay.classList.remove('hidden'));
  document.getElementById('helpClose').addEventListener('click', () => helpOverlay.classList.add('hidden'));
  helpOverlay.addEventListener('click', e => { if (e.target === helpOverlay) helpOverlay.classList.add('hidden'); });

  setStatus('Loading datasets…');
  await loadDatasets();

  setStatus('Loading robot model…');
  loadRobot()
    .then(() => {
      buildSecondaryTrajectories();
      updateFrame(frameIdx);
      setStatus('Ready');
    })
    .catch(err => {
      console.warn('Robot model failed:', err);
      setStatus('Robot unavailable — EE panel still works');
    });
}

window.addEventListener('error', e => setStatus(`JS error: ${e.message} (${e.filename?.split('/').pop()}:${e.lineno})`));
window.addEventListener('unhandledrejection', e => setStatus(`Unhandled: ${e.reason?.message ?? e.reason}`));

init().catch(err => { console.error(err); setStatus(`Init failed: ${err.message}`); });
