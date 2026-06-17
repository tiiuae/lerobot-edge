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

import { S } from './js/state.js';
import { setupJointScene, setupEEScene, setupCameraSync, animate, snapCamera } from './js/scene.js';
import { loadRobot } from './js/robot.js';
import { buildSecondaryTrajectories } from './js/overlays.js';
import { setupModeButtons } from './js/modes.js';
import { setupCalibrationUI } from './js/calibration.js';
import { setupDataPanel } from './js/datapanel.js';
import { setupGraphPanel } from './js/graphs.js';
import { setupSettingsPanel } from './js/settings.js';
import { updateFrame, startPlayback, stopPlayback, setStatus } from './js/playback.js';
import { loadDatasets, initDatasetBrowser } from './js/api.js';

async function init() {
  setupJointScene();
  setupEEScene();
  setupCameraSync();
  setupModeButtons();
  setupCalibrationUI();
  setupDataPanel();
  setupGraphPanel();
  setupSettingsPanel();
  animate();

  // Camera view preset buttons (present in both panels)
  document.querySelectorAll('.cam-btn').forEach(btn => {
    btn.addEventListener('click', () => snapCamera(btn.dataset.view));
  });

  document.getElementById('playBtn').addEventListener('click', () =>
    S.playing ? stopPlayback() : startPlayback());
  document.getElementById('frameSlider').addEventListener('input', e => {
    stopPlayback();
    updateFrame(+e.target.value);
  });

  const helpOverlay = document.getElementById('helpOverlay');
  document.getElementById('helpBtn').addEventListener('click',   () => helpOverlay.classList.remove('hidden'));
  document.getElementById('helpClose').addEventListener('click', () => helpOverlay.classList.add('hidden'));
  helpOverlay.addEventListener('click', e => { if (e.target === helpOverlay) helpOverlay.classList.add('hidden'); });

  setStatus('Loading datasets…');
  initDatasetBrowser();
  await loadDatasets();

  setStatus('Loading robot model…');
  loadRobot()
    .then(() => {
      buildSecondaryTrajectories();
      updateFrame(S.frameIdx);
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
