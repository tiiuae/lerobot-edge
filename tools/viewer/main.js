/**
 * Robot Dataset Viewer — main.js
 *
 * LEFT  panel  (Joint Space):    URDF robot driven by observation.state.
 *                                Small dots show observation.ee_left/right for reference.
 *
 * RIGHT panel  (FK vs Dataset):  No robot body — pure EE comparison.
 *   • Solid blue/red spheres   = observation.ee_left/right  (what the dataset recorded)
 *   • Cyan/orange ghost spheres = FK EE read from the URDF after applying state joints
 *   • Yellow lines              = error between the two (invisible if FK is correct)
 *   • Solid trails              = observation EE trajectory
 *   • Faint trails              = FK EE trajectory
 *
 * Coordinate convention:
 *   ROS z-up world. camera.up = (0,0,1).
 *   ColladaLoader auto-rotation is undone so meshes stay in z-up.
 */

import * as THREE        from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ColladaLoader }  from 'three/examples/jsm/loaders/ColladaLoader.js';
import URDFLoader         from 'urdf-loader';

// ── Global playback state ─────────────────────────────────────────────────────
let frames     = [];
let frameIdx   = 0;
let playing    = false;
let playTimer  = null;
let stateNames = [];
let hasEELeft  = false;
let hasEERight = false;

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

// ── Joint calibration ────────────────────────────────────────────────────────
// Per-arm calibration: each side has independent offset/sign for each joint.
// urdf_value = sign * raw_value + offset (rad). Persisted in localStorage.
const CALIB_KEY = 'lerobot-viewer-joint-calibration-v3';
const SIDES = ['left', 'right'];

function defaultCalib() {
  const c = {};
  for (const side of SIDES) {
    for (let i = 0; i <= 6; i++) {
      c[`${side}_joint_${i}`] = { offset: 0, sign: 1 };
    }
  }
  return c;
}

function loadJointCalib() {
  try {
    const stored = localStorage.getItem(CALIB_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      return { ...defaultCalib(), ...parsed };
    }
  } catch (e) { console.warn('calib load failed', e); }
  return defaultCalib();
}

function saveJointCalib() {
  try { localStorage.setItem(CALIB_KEY, JSON.stringify(jointCalib)); }
  catch (e) { console.warn('calib save failed', e); }
}

let jointCalib = loadJointCalib();

// Apply per-arm, per-joint sign/offset transform.
// jointDsName e.g. "left_joint_3" is used directly as the calibration key.
function transformJointValue(jointDsName, raw) {
  const c = jointCalib[jointDsName];
  if (!c) return raw;
  return c.sign * raw + c.offset;
}

// True dataset layout (the meta/info.json names are mislabeled):
//   state[0..6]   = left  arm joints 0..6   (joint_6 is the gripper)
//   state[7..13]  = right arm joints 0..6
//   state[14..18] = base info (5 values, e.g., odom_x/y/theta + linear/angular_vel)
//   action[0..6]   = left  arm joints 0..6
//   action[7..13]  = right arm joints 0..6
//   action[14..15] = base linear_vel, angular_vel
//
// Verified empirically: in this dataset the left arm stays fixed at
// [0, π/3, π/6, ~0.6, 0, 0, 0] throughout the episode while the right arm
// (indices 7..13) performs the full pick-and-place motion. The metadata's
// claim that indices 0..4 are odom/velocity is wrong — those are left-arm
// joints 0..4.
const STATE_IDX = {
  left_joint_0: 0,  left_joint_1: 1,  left_joint_2: 2,
  left_joint_3: 3,  left_joint_4: 4,  left_joint_5: 5,  left_joint_6: 6,
  right_joint_0: 7, right_joint_1: 8, right_joint_2: 9,
  right_joint_3: 10, right_joint_4: 11, right_joint_5: 12, right_joint_6: 13,
};

// Action shares the same joint layout for indices 0..13. We ignore the
// dataset-provided action_names (also mislabeled) and the state_names entirely.

const LEFT_MOUNT  = [0.331,  0.3, 0.831];
const RIGHT_MOUNT = [0.331, -0.3, 0.831];

// ── Colour palette ────────────────────────────────────────────────────────────
const BG    = 0xf0f2f5;
const GRID1 = 0x999999;
const GRID2 = 0xcccccc;

// EE colours — consistent across panels
const C_OBS_L  = 0x2266cc;   // observation EE left  — blue
const C_OBS_R  = 0xcc2211;   // observation EE right — red
const C_FK_L   = 0x00bbdd;   // FK EE left           — cyan
const C_FK_R   = 0xdd7700;   // FK EE right          — orange
const C_ERR    = 0xffdd00;   // error line           — yellow

// ── Shared scene helpers ──────────────────────────────────────────────────────
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

// ── LEFT PANEL — joint-space robot ───────────────────────────────────────────
let jRenderer, jScene, jCamera, jControls, jRosRoot;
let robot = null;
// Observation EE dots — small reference spheres overlaid on the arm model
let jObsLeftMark, jObsRightMark;
// Observation EE trails
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

  ({ cam: jCamera, ctrl: jControls } = makeCamera(
    canvas, [3.0, -3.5, 2.0], [0.0, 0.0, 1.0]
  ));

  // Small observation EE dots — show where the dataset says the EE is
  const sGeo = new THREE.SphereGeometry(0.016, 10, 10);
  jObsLeftMark  = new THREE.Mesh(sGeo, new THREE.MeshPhongMaterial({ color: C_OBS_L, emissive: 0x112244 }));
  jObsRightMark = new THREE.Mesh(sGeo, new THREE.MeshPhongMaterial({ color: C_OBS_R, emissive: 0x441108 }));
  jRosRoot.add(jObsLeftMark, jObsRightMark);
  jObsLeftMark.visible = jObsRightMark.visible = false;
}

// ── RIGHT PANEL — FK vs Dataset EE comparison ────────────────────────────────
let eRenderer, eScene, eCamera, eControls, eRosRoot;
// Observation EE (from dataset)
let eeLeftFrame, eeRightFrame;
let eeObsLeftMark, eeObsRightMark;
let eeObsLeftTrail, eeObsRightTrail;
// FK EE (computed from URDF after applying state joints)
let eeFKLeftMark, eeFKRightMark;
let eeFKLeftTrail, eeFKRightTrail;
// Error lines: observation EE → FK EE
let errLeftLine, errRightLine;

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
  arr[0] = a.x; arr[1] = a.y; arr[2] = a.z;
  arr[3] = b.x; arr[4] = b.y; arr[5] = b.z;
  line.geometry.attributes.position.needsUpdate = true;
}

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

  // World-origin axes + arm mount reference dots
  eRosRoot.add(new THREE.AxesHelper(0.2));
  const mountGeo = new THREE.SphereGeometry(0.022, 10, 10);
  const mountMat = new THREE.MeshPhongMaterial({ color: 0x334455, emissive: 0x111a22 });
  const lMount = new THREE.Mesh(mountGeo, mountMat.clone());
  const rMount = new THREE.Mesh(mountGeo, mountMat.clone());
  lMount.position.set(...LEFT_MOUNT);
  rMount.position.set(...RIGHT_MOUNT);
  eRosRoot.add(lMount, rMount);

  // EE orientation frames (from observation.ee_*)
  eeLeftFrame  = makeAxisFrame(0.09, [0x2266cc, 0x0099cc, 0x6633cc]);
  eeRightFrame = makeAxisFrame(0.09, [0xcc2211, 0xcc6611, 0xcc2299]);
  eRosRoot.add(eeLeftFrame, eeRightFrame);
  eeLeftFrame.visible = eeRightFrame.visible = false;

  // Observation EE spheres (solid, larger — primary reference)
  const sObs = new THREE.SphereGeometry(0.020, 12, 12);
  eeObsLeftMark  = new THREE.Mesh(sObs, new THREE.MeshPhongMaterial({ color: C_OBS_L, emissive: 0x112244 }));
  eeObsRightMark = new THREE.Mesh(sObs, new THREE.MeshPhongMaterial({ color: C_OBS_R, emissive: 0x441108 }));
  eRosRoot.add(eeObsLeftMark, eeObsRightMark);
  eeObsLeftMark.visible = eeObsRightMark.visible = false;

  // FK EE spheres (slightly smaller, distinct colour — what the URDF kinematic chain gives)
  const sFK = new THREE.SphereGeometry(0.014, 10, 10);
  eeFKLeftMark  = new THREE.Mesh(sFK, new THREE.MeshPhongMaterial({ color: C_FK_L, emissive: 0x003344 }));
  eeFKRightMark = new THREE.Mesh(sFK, new THREE.MeshPhongMaterial({ color: C_FK_R, emissive: 0x331100 }));
  eRosRoot.add(eeFKLeftMark, eeFKRightMark);
  eeFKLeftMark.visible = eeFKRightMark.visible = false;

  // Error lines: observation EE ↔ FK EE — yellow, visually striking
  errLeftLine  = makeUpdatableLine(C_ERR);
  errRightLine = makeUpdatableLine(C_ERR);
  eRosRoot.add(errLeftLine, errRightLine);

  ({ cam: eCamera, ctrl: eControls } = makeCamera(
    canvas, [3.0, -3.5, 2.0], [0.35, 0.0, 1.0]
  ));
}

// ── URDF loading — single instance for joint panel + FK computation ───────────
async function loadRobot() {
  const loader = makeURDFLoader();
  robot = await new Promise((resolve, reject) => {
    loader.load('/robot.urdf', r => {
      r.traverse(child => {
        if (child.isMesh) {
          child.material = new THREE.MeshPhongMaterial({
            color: 0x8899aa, shininess: 60, specular: 0x445566,
          });
          child.castShadow = child.receiveShadow = true;
        }
      });
      resolve(r);
    }, null, reject);
  });
  jRosRoot.add(robot);
}

// ── Joint application ─────────────────────────────────────────────────────────
function buildValuesByName(data, names, fallbackIdxMap) {
  const byName = {};
  if (names.length) {
    names.forEach((n, i) => { if (i < data.length) byName[n] = data[i]; });
  } else {
    for (const [name, idx] of Object.entries(fallbackIdxMap)) {
      if (idx < data.length) byName[name] = data[idx];
    }
  }
  return byName;
}

function applyRobotJoints(rb, byName) {
  for (const [dsName, urdfName] of Object.entries(JOINT_MAP)) {
    if (dsName in byName) rb.setJointValue(urdfName, transformJointValue(dsName, byName[dsName]));
  }
  for (const [dsName, urdfNames] of Object.entries(GRIPPER_MAP)) {
    if (dsName in byName) {
      const v = transformJointValue(dsName, byName[dsName]);
      for (const u of urdfNames) rb.setJointValue(u, v);
    }
  }
}

function getEEWorldPos(robotObj, linkName) {
  robotObj.updateMatrixWorld(true);
  const link = robotObj.getObjectByName(linkName);
  return link ? new THREE.Vector3().setFromMatrixPosition(link.matrixWorld) : null;
}

// Drive the single robot from observation.state.
// We pass [] for names to force STATE_IDX (dataset metadata is mislabeled).
function applyJoints(frame) {
  if (!robot) return;
  const state = frame['observation.state'];
  if (!state) return;
  const byName = buildValuesByName(state, [], STATE_IDX);
  applyRobotJoints(robot, byName);
  if (frameIdx === 0) {
    const matched = Object.keys(JOINT_MAP).filter(n => n in byName);
    console.log('[joints] state:', matched.map(n => `${n}=${byName[n]?.toFixed(3)}`).join('  '));
  }
}

// ── EE visualisation helpers ──────────────────────────────────────────────────
function makeAxisFrame(size, colors) {
  const group = new THREE.Group();
  const dirs  = [new THREE.Vector3(1,0,0), new THREE.Vector3(0,1,0), new THREE.Vector3(0,0,1)];
  dirs.forEach((dir, i) => {
    group.add(new THREE.ArrowHelper(dir, new THREE.Vector3(), size, colors[i], size * 0.28, size * 0.14));
  });
  return group;
}

// pose7 = [x, y, z, qw, qx, qy, qz]
function applyPose(obj, pose7) {
  const [x, y, z, qw, qx, qy, qz] = pose7;
  obj.position.set(x, y, z);
  obj.quaternion.set(qx, qy, qz, qw);
}

function makeLine(points, color, opacity) {
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color, opacity, transparent: opacity < 1 })
  );
}

// ── Trajectory builders ───────────────────────────────────────────────────────
function buildObsTrajectories() {
  [eeObsLeftTrail, eeObsRightTrail, jObsLeftTrail, jObsRightTrail].forEach(t => {
    if (t) { (t.parent === eRosRoot ? eRosRoot : jRosRoot).remove(t); }
  });
  eeObsLeftTrail = eeObsRightTrail = jObsLeftTrail = jObsRightTrail = null;

  if (hasEELeft) {
    const pts = frames.map(f => f['observation.ee_left']).filter(Boolean)
      .map(p => new THREE.Vector3(p[0], p[1], p[2]));
    eeObsLeftTrail = makeLine(pts, C_OBS_L, 0.6);
    jObsLeftTrail  = makeLine(pts, C_OBS_L, 0.6);
    eRosRoot.add(eeObsLeftTrail);
    jRosRoot.add(jObsLeftTrail);
  }
  if (hasEERight) {
    const pts = frames.map(f => f['observation.ee_right']).filter(Boolean)
      .map(p => new THREE.Vector3(p[0], p[1], p[2]));
    eeObsRightTrail = makeLine(pts, C_OBS_R, 0.6);
    jObsRightTrail  = makeLine(pts, C_OBS_R, 0.6);
    eRosRoot.add(eeObsRightTrail);
    jRosRoot.add(jObsRightTrail);
  }
}

// Precompute FK EE trajectory by stepping through all frames with observation.state.
// Temporarily drives `robot` per-frame, then restores current frame.
function buildFKTrajectories() {
  if (eeFKLeftTrail)  eRosRoot.remove(eeFKLeftTrail);
  if (eeFKRightTrail) eRosRoot.remove(eeFKRightTrail);
  eeFKLeftTrail = eeFKRightTrail = null;

  if (!robot || !frames.length) return;

  const ptsL = [], ptsR = [];
  for (const frame of frames) {
    const state = frame['observation.state'];
    if (!state) continue;
    const byName = buildValuesByName(state, [], STATE_IDX);
    applyRobotJoints(robot, byName);
    const lPos = getEEWorldPos(robot, 'follower_left_ee_gripper_link');
    const rPos = getEEWorldPos(robot, 'follower_right_ee_gripper_link');
    if (lPos) ptsL.push(lPos.clone());
    if (rPos) ptsR.push(rPos.clone());
  }

  if (ptsL.length) { eeFKLeftTrail  = makeLine(ptsL, C_FK_L, 0.5); eRosRoot.add(eeFKLeftTrail); }
  if (ptsR.length) { eeFKRightTrail = makeLine(ptsR, C_FK_R, 0.5); eRosRoot.add(eeFKRightTrail); }

  // Restore current frame
  if (frames[frameIdx]) applyJoints(frames[frameIdx]);
}

// ── Per-frame EE update ───────────────────────────────────────────────────────
function updateEE(frame) {
  const obsL = frame['observation.ee_left'];
  const obsR = frame['observation.ee_right'];

  // FK EE: read URDF link world positions (robot joints already applied in applyJoints)
  const fkL = robot ? getEEWorldPos(robot, 'follower_left_ee_gripper_link')  : null;
  const fkR = robot ? getEEWorldPos(robot, 'follower_right_ee_gripper_link') : null;

  // ── Left panel: observation EE dots (small reference overlay on robot model)
  if (obsL && hasEELeft) {
    jObsLeftMark.position.set(obsL[0], obsL[1], obsL[2]);
    jObsLeftMark.visible = true;
  }
  if (obsR && hasEERight) {
    jObsRightMark.position.set(obsR[0], obsR[1], obsR[2]);
    jObsRightMark.visible = true;
  }

  // ── Right panel: observation EE — axis frame + solid sphere
  if (obsL && hasEELeft) {
    applyPose(eeLeftFrame, obsL);
    eeLeftFrame.visible = true;
    eeObsLeftMark.position.set(obsL[0], obsL[1], obsL[2]);
    eeObsLeftMark.visible = true;
  }
  if (obsR && hasEERight) {
    applyPose(eeRightFrame, obsR);
    eeRightFrame.visible = true;
    eeObsRightMark.position.set(obsR[0], obsR[1], obsR[2]);
    eeObsRightMark.visible = true;
  }

  // ── Right panel: FK EE spheres
  if (fkL) { eeFKLeftMark.position.copy(fkL);  eeFKLeftMark.visible  = true; }
  if (fkR) { eeFKRightMark.position.copy(fkR); eeFKRightMark.visible = true; }

  // ── Right panel: error lines  (observation EE → FK EE)
  if (obsL && fkL && hasEELeft) {
    setLinePoints(errLeftLine, new THREE.Vector3(obsL[0], obsL[1], obsL[2]), fkL);
    errLeftLine.visible = true;
  }
  if (obsR && fkR && hasEERight) {
    setLinePoints(errRightLine, new THREE.Vector3(obsR[0], obsR[1], obsR[2]), fkR);
    errRightLine.visible = true;
  }
}

// ── Dataset / Episode loading ──────────────────────────────────────────────────
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
  const opt = document.getElementById('datasetSelect').selectedOptions[0];
  if (!opt) return;
  const info = JSON.parse(opt.dataset.info);
  stateNames = info.state_names ?? [];
  hasEELeft  = info.has_ee_left;
  hasEERight = info.has_ee_right;

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
  buildFKTrajectories();   // no-op until robot is loaded
  updateFrame(0);
  setStatus(`${frames.length} frames at 50 fps  ·  ${(frames.length / 50).toFixed(1)} s`);
}

// ── Playback controls ─────────────────────────────────────────────────────────
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
// Per-arm sections — left and right calibrated independently. Each arm has a
// "snap current frame to zero" button that auto-computes offsets so that the
// current frame's joints render as URDF home.
function setupCalibrationUI() {
  const container = document.getElementById('calibRows');
  const panel     = document.getElementById('calibPanel');
  const btn       = document.getElementById('calibBtn');
  const closeBtn  = document.getElementById('calibClose');
  const resetBtn  = document.getElementById('calibReset');

  btn.addEventListener('click', () => panel.classList.toggle('hidden'));
  closeBtn.addEventListener('click', () => panel.classList.add('hidden'));
  resetBtn.addEventListener('click', () => {
    jointCalib = defaultCalib();
    saveJointCalib();
    rebuildCalibRows();
    onCalibChange();
  });

  // Set per-arm offsets so that the CURRENT frame's joint values map to URDF zero.
  // Useful when you know the arm was physically at home pose at the displayed frame.
  function snapCurrentFrameToZero(side) {
    const frame = frames[frameIdx];
    if (!frame) return;
    const state = frame['observation.state'];
    if (!state) return;
    const byName = buildValuesByName(state, [], STATE_IDX);
    for (let i = 0; i <= 6; i++) {
      const key = `${side}_joint_${i}`;
      if (key in byName) {
        const c = jointCalib[key];
        // urdf = sign * raw + offset = 0  ⇒  offset = -sign * raw
        c.offset = -c.sign * byName[key];
      }
    }
    saveJointCalib();
    rebuildCalibRows();
    onCalibChange();
  }

  // Set per-arm to identity (offset=0, sign=+1)
  function resetArm(side) {
    for (let i = 0; i <= 6; i++) {
      jointCalib[`${side}_joint_${i}`] = { offset: 0, sign: 1 };
    }
    saveJointCalib();
    rebuildCalibRows();
    onCalibChange();
  }

  function makeArmSection(side) {
    const section = document.createElement('div');
    section.className = 'calib-section';
    section.dataset.side = side;
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
      <button class="snap-btn"  data-side="${side}" title="Set offsets so the current frame's joint values map to URDF home pose">Snap current → zero</button>
      <button class="armrst-btn" data-side="${side}" title="Reset this arm's calibration to identity">Reset</button>
    `;
    section.appendChild(actions);

    return section;
  }

  function rebuildCalibRows() {
    container.innerHTML = '';
    container.appendChild(makeArmSection('left'));
    container.appendChild(makeArmSection('right'));

    container.querySelectorAll('.sign-btn').forEach(b => {
      b.addEventListener('click', e => {
        const key = e.target.dataset.key;
        const c = jointCalib[key];
        c.sign = -c.sign;
        e.target.textContent = c.sign > 0 ? '+' : '−';
        saveJointCalib();
        onCalibChange();
      });
    });
    container.querySelectorAll('.cal-slider').forEach(s => {
      s.addEventListener('input', e => {
        const key = e.target.dataset.key;
        const v = +e.target.value;
        jointCalib[key].offset = v;
        container.querySelector(`.cal-num[data-key="${key}"]`).value = v.toFixed(3);
        saveJointCalib();
        onCalibChange();
      });
    });
    container.querySelectorAll('.cal-num').forEach(n => {
      n.addEventListener('change', e => {
        const key = e.target.dataset.key;
        const v = +e.target.value;
        jointCalib[key].offset = v;
        container.querySelector(`.cal-slider[data-key="${key}"]`).value = v;
        saveJointCalib();
        onCalibChange();
      });
    });
    container.querySelectorAll('.snap-btn').forEach(b => {
      b.addEventListener('click', e => snapCurrentFrameToZero(e.target.dataset.side));
    });
    container.querySelectorAll('.armrst-btn').forEach(b => {
      b.addEventListener('click', e => resetArm(e.target.dataset.side));
    });
  }

  rebuildCalibRows();
}

// Called when calibration changes — rebuilds FK trail and re-applies current frame.
let calibChangeTimer = null;
function onCalibChange() {
  // Debounce: slider drag fires many events; rebuilding FK over all frames is the
  // expensive part, so defer it slightly.
  if (calibChangeTimer) clearTimeout(calibChangeTimer);
  if (frames[frameIdx]) updateFrame(frameIdx);  // immediate: current frame updates
  calibChangeTimer = setTimeout(() => {
    buildFKTrajectories();
    calibChangeTimer = null;
  }, 80);
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  setupJointScene();
  setupEEScene();
  setupCameraSync();
  setupCalibrationUI();
  animate();

  document.getElementById('playBtn').addEventListener('click', () => {
    playing ? stopPlayback() : startPlayback();
  });
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
      buildFKTrajectories();
      updateFrame(frameIdx);
      setStatus('Ready');
    })
    .catch(err => {
      console.warn('Robot model failed:', err);
      setStatus('Robot unavailable — EE panel still works');
    });
}

window.addEventListener('error', e => {
  setStatus(`JS error: ${e.message} (${e.filename?.split('/').pop()}:${e.lineno})`);
});
window.addEventListener('unhandledrejection', e => {
  setStatus(`Unhandled rejection: ${e.reason?.message ?? e.reason}`);
});

init().catch(err => {
  console.error(err);
  setStatus(`Init failed: ${err.message}`);
});
