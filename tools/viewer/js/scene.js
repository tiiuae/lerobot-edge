// Three.js scene setup: scenes, cameras, renderers, lights, floor, render loop.

import * as THREE        from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ColladaLoader }  from 'three/examples/jsm/loaders/ColladaLoader.js';
import { Line2 }         from 'three/addons/lines/Line2.js';
import { LineGeometry }  from 'three/addons/lines/LineGeometry.js';
import { LineMaterial }  from 'three/addons/lines/LineMaterial.js';
import URDFLoader         from 'urdf-loader';
import * as K from './constants.js';
import { VC } from './vis-config.js';

// ── Singleton scene variables ─────────────────────────────────────────────────
export let jRenderer = null, jScene = null, jCamera = null, jControls = null, jRosRoot = null;
export let jObsLeftMark = null, jObsRightMark = null;

export let eRenderer = null, eScene = null, eCamera = null, eControls = null, eRosRoot = null;
export let eeLeftFrame = null, eeRightFrame = null;
export let eeObsLeftMark = null, eeObsRightMark = null;
export let eeSecLeftMark = null, eeSecRightMark = null;
export let errLeftLine = null, errRightLine = null;
export let arrowLeft = null, arrowRight = null;

// Trail references — stored in an object so other modules can reassign them
// without violating ES module live-binding rules (you cannot reassign an
// imported binding from outside the declaring module).
export const sceneRefs = {
  jObsLeftTrail:  null,
  jObsRightTrail: null,
  eeObsLeftTrail:  null,
  eeObsRightTrail: null,
  eeSecLeftTrail:  null,
  eeSecRightTrail: null,
};

// ── Shared helpers ────────────────────────────────────────────────────────────
export function makeFloor(scene) {
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(10, 10),
    new THREE.ShadowMaterial({ opacity: 0.15 })
  );
  floor.receiveShadow = true;
  scene.add(floor);
  const grid = new THREE.GridHelper(6, 30, K.GRID1, K.GRID2);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);
}

export function makeCamera(canvas, pos, target) {
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

export function makeLights(scene) {
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

export function makeURDFLoader() {
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

export function makeAxisFrame(size, colors) {
  const g = new THREE.Group();
  const dirs = [new THREE.Vector3(1,0,0), new THREE.Vector3(0,1,0), new THREE.Vector3(0,0,1)];
  dirs.forEach((d, i) =>
    g.add(new THREE.ArrowHelper(d, new THREE.Vector3(), size, colors[i], size*0.28, size*0.14)));
  return g;
}

// Tracks all active LineMaterial instances so syncSize can update their resolution.
export const trailMaterials = new Set();

export function makeLine(points, color, opacity) {
  if (points.length < 2) return null;
  const geo = new LineGeometry();
  geo.setPositions(points.flatMap(p => [p.x, p.y, p.z]));
  const mat = new LineMaterial({
    color: new THREE.Color(color),
    opacity,
    transparent: opacity < 1,
    linewidth: VC.trailLineWidth,
    resolution: new THREE.Vector2(window.innerWidth, window.innerHeight),
  });
  trailMaterials.add(mat);
  const line = new Line2(geo, mat);
  line.frustumCulled = false;
  return line;
}

export function makeUpdatableLine(color) {
  const buf = new Float32Array(6);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(buf, 3));
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color }));
  line.visible = false;
  return line;
}

export function applySphereScale(scale) {
  for (const m of [jObsLeftMark, jObsRightMark, eeObsLeftMark, eeObsRightMark, eeSecLeftMark, eeSecRightMark])
    if (m) m.scale.setScalar(scale);
}

export function setLinePoints(line, a, b) {
  const arr = line.geometry.attributes.position.array;
  arr[0]=a.x; arr[1]=a.y; arr[2]=a.z;
  arr[3]=b.x; arr[4]=b.y; arr[5]=b.z;
  line.geometry.attributes.position.needsUpdate = true;
}

// Update an ArrowHelper to point from origin in direction of delta_xyz.
// Returns magnitude (metres). Hides arrow if magnitude is negligible.
export const ARROW_SCALE = 8.0;  // visual scale factor — delta in metres, arrows in scene units
export function updateArrow(arrow, originV3, dx, dy, dz) {
  const mag = Math.sqrt(dx*dx + dy*dy + dz*dz);
  if (mag < 1e-6) { arrow.visible = false; return 0; }
  arrow.position.copy(originV3);
  arrow.setDirection(new THREE.Vector3(dx/mag, dy/mag, dz/mag));
  const len = Math.max(mag * ARROW_SCALE, 0.02);
  arrow.setLength(len, Math.min(0.04, len * 0.25), Math.min(0.02, len * 0.12));
  arrow.visible = true;
  return mag;
}

// ── Left panel (joint scene) ──────────────────────────────────────────────────
export function setupJointScene() {
  const canvas = document.getElementById('jointCanvas');
  jRenderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  jRenderer.setPixelRatio(window.devicePixelRatio);
  jRenderer.shadowMap.enabled = true;
  jRenderer.shadowMap.type    = THREE.PCFSoftShadowMap;

  jScene = new THREE.Scene();
  jScene.background = new THREE.Color(VC.bgColor);
  jScene.fog = new THREE.Fog(VC.bgColor, 10, 25);

  jRosRoot = new THREE.Group();
  jScene.add(jRosRoot);
  makeLights(jScene);
  makeFloor(jScene);

  ({ cam: jCamera, ctrl: jControls } = makeCamera(canvas, [3.0, -3.5, 2.0], [0.0, 0.0, 1.0]));

  const sGeo = new THREE.SphereGeometry(0.016, 10, 10);
  jObsLeftMark  = new THREE.Mesh(sGeo, new THREE.MeshPhongMaterial({ color: K.C_OBS_L, emissive: 0x112244 }));
  jObsRightMark = new THREE.Mesh(sGeo, new THREE.MeshPhongMaterial({ color: K.C_OBS_R, emissive: 0x441108 }));
  jObsLeftMark.scale.setScalar(VC.sphereScale);
  jObsRightMark.scale.setScalar(VC.sphereScale);
  jRosRoot.add(jObsLeftMark, jObsRightMark);
  jObsLeftMark.visible = jObsRightMark.visible = false;
}

// ── Right panel (EE scene) ────────────────────────────────────────────────────
export function setupEEScene() {
  const canvas = document.getElementById('eeCanvas');
  eRenderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  eRenderer.setPixelRatio(window.devicePixelRatio);
  eRenderer.shadowMap.enabled = true;
  eRenderer.shadowMap.type    = THREE.PCFSoftShadowMap;

  eScene = new THREE.Scene();
  eScene.background = new THREE.Color(VC.bgColor);
  eScene.fog = new THREE.Fog(VC.bgColor, 10, 25);

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
  lMount.position.set(...K.LEFT_MOUNT);
  rMount.position.set(...K.RIGHT_MOUNT);
  eRosRoot.add(lMount, rMount);

  // Observation EE — axis frames
  eeLeftFrame  = makeAxisFrame(0.09, [0x2266cc, 0x0099cc, 0x6633cc]);
  eeRightFrame = makeAxisFrame(0.09, [0xcc2211, 0xcc6611, 0xcc2299]);
  eRosRoot.add(eeLeftFrame, eeRightFrame);
  eeLeftFrame.visible = eeRightFrame.visible = false;

  // Observation EE — solid spheres (primary)
  const sObs = new THREE.SphereGeometry(0.020, 12, 12);
  eeObsLeftMark  = new THREE.Mesh(sObs, new THREE.MeshPhongMaterial({ color: K.C_OBS_L, emissive: 0x112244 }));
  eeObsRightMark = new THREE.Mesh(sObs, new THREE.MeshPhongMaterial({ color: K.C_OBS_R, emissive: 0x441108 }));
  eeObsLeftMark.scale.setScalar(VC.sphereScale);
  eeObsRightMark.scale.setScalar(VC.sphereScale);
  eRosRoot.add(eeObsLeftMark, eeObsRightMark);
  eeObsLeftMark.visible = eeObsRightMark.visible = false;

  // Secondary EE — smaller spheres (FK EE or action.ee depending on mode)
  const sSec = new THREE.SphereGeometry(0.014, 10, 10);
  eeSecLeftMark  = new THREE.Mesh(sSec, new THREE.MeshPhongMaterial({ color: K.C_SEC_L, emissive: 0x003344 }));
  eeSecRightMark = new THREE.Mesh(sSec, new THREE.MeshPhongMaterial({ color: K.C_SEC_R, emissive: 0x331100 }));
  eeSecLeftMark.scale.setScalar(VC.sphereScale);
  eeSecRightMark.scale.setScalar(VC.sphereScale);
  eRosRoot.add(eeSecLeftMark, eeSecRightMark);
  eeSecLeftMark.visible = eeSecRightMark.visible = false;

  // Gap / error lines
  errLeftLine  = makeUpdatableLine(K.C_ERR);
  errRightLine = makeUpdatableLine(K.C_ERR);
  eRosRoot.add(errLeftLine, errRightLine);

  // Delta / Relative arrow helpers
  const fwd = new THREE.Vector3(1, 0, 0);
  const org = new THREE.Vector3(0, 0, 0);
  arrowLeft  = new THREE.ArrowHelper(fwd, org, 0.1, K.C_DELTA, 0.04, 0.02);
  arrowRight = new THREE.ArrowHelper(fwd, org, 0.1, K.C_DELTA, 0.04, 0.02);
  arrowLeft.visible = arrowRight.visible = false;
  eRosRoot.add(arrowLeft, arrowRight);

  ({ cam: eCamera, ctrl: eControls } = makeCamera(canvas, [3.0, -3.5, 2.0], [0.35, 0.0, 1.0]));
}

// ── Camera snap to preset views ───────────────────────────────────────────────
// Uses jControls (primary) as reference; camera sync propagates to eCamera.
// Top view uses slight Y offset to avoid gimbal lock (camera.up is Z).
export function snapCamera(view) {
  const t = jControls.target;
  const D = 4;
  const presets = {
    top:    [t.x,       t.y - 0.5, t.z + D],
    bottom: [t.x,       t.y - 0.5, t.z - D],
    front:  [t.x + D,   t.y,       t.z    ],
    back:   [t.x - D,   t.y,       t.z    ],
    left:   [t.x,       t.y + D,   t.z    ],
    right:  [t.x,       t.y - D,   t.z    ],
    iso:    [t.x + 3.0, t.y - 3.5, t.z + 2.0],
  };
  const pos = presets[view];
  if (!pos) return;
  jCamera.position.set(...pos);
  jControls.update();
}

// ── Camera sync ───────────────────────────────────────────────────────────────
let primaryCtrl = 'j';  // private — only used in setupCameraSync and animate

export function setupCameraSync() {
  document.getElementById('jointCanvas').addEventListener('pointerdown', () => { primaryCtrl = 'j'; });
  document.getElementById('eeCanvas').addEventListener('pointerdown',   () => { primaryCtrl = 'e'; });
}

export function syncCameras() {
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
export function syncSize(renderer, camera, canvas) {
  const { clientWidth: w, clientHeight: h } = canvas.parentElement;
  if (renderer.domElement.width !== w || renderer.domElement.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    for (const mat of trailMaterials) mat.resolution.set(w, h);
  }
}

export function animate() {
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
