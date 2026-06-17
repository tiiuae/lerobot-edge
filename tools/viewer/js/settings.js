// Settings panel: visual configuration UI (bg color, trail colors/opacity, chart width).

import { VC, saveVC, resetVC } from './vis-config.js';
import { buildObsTrajectories, buildSecondaryTrajectories } from './overlays.js';
import { buildGraphCharts } from './graphs.js';
import { jScene, eScene } from './scene.js';
import { makeDraggable } from './calibration.js';

export function setupSettingsPanel() {
  const btn   = document.getElementById('styleBtn');
  const panel = document.getElementById('settingsPanel');
  if (!btn || !panel) return;

  btn.addEventListener('click', () => {
    const hidden = panel.classList.toggle('hidden');
    btn.classList.toggle('active', !hidden);
  });
  document.getElementById('settingsClose').addEventListener('click', () => {
    panel.classList.add('hidden');
    btn.classList.remove('active');
  });
  makeDraggable(panel, panel.querySelector('.settings-hdr'));

  // Populate inputs with persisted values
  set('vc-bg-color',          VC.bgColor);
  set('vc-obs-l',             VC.trailObsLColor);
  set('vc-obs-r',             VC.trailObsRColor);
  set('vc-sec-l',             VC.trailSecLColor);
  set('vc-sec-r',             VC.trailSecRColor);
  set('vc-opacity',           VC.trailOpacity);
  set('vc-chart-width',       VC.chartBorderWidth);
  setTxt('vc-opacity-val',    VC.trailOpacity.toFixed(2));
  setTxt('vc-chart-width-val',VC.chartBorderWidth.toFixed(1));

  wire('vc-bg-color', v => {
    VC.bgColor = v;
    applyBg(v);
    saveVC();
  });

  wire('vc-obs-l', v => { VC.trailObsLColor = v; buildObsTrajectories();        saveVC(); });
  wire('vc-obs-r', v => { VC.trailObsRColor = v; buildObsTrajectories();        saveVC(); });
  wire('vc-sec-l', v => { VC.trailSecLColor = v; buildSecondaryTrajectories(); saveVC(); });
  wire('vc-sec-r', v => { VC.trailSecRColor = v; buildSecondaryTrajectories(); saveVC(); });

  wire('vc-opacity', v => {
    VC.trailOpacity = +v;
    setTxt('vc-opacity-val', (+v).toFixed(2));
    buildObsTrajectories();
    buildSecondaryTrajectories();
    saveVC();
  });

  wire('vc-chart-width', v => {
    VC.chartBorderWidth = +v;
    setTxt('vc-chart-width-val', (+v).toFixed(1));
    buildGraphCharts();
    saveVC();
  });

  document.getElementById('vcResetBtn').addEventListener('click', () => {
    resetVC();
    set('vc-bg-color',          VC.bgColor);
    set('vc-obs-l',             VC.trailObsLColor);
    set('vc-obs-r',             VC.trailObsRColor);
    set('vc-sec-l',             VC.trailSecLColor);
    set('vc-sec-r',             VC.trailSecRColor);
    set('vc-opacity',           VC.trailOpacity);
    set('vc-chart-width',       VC.chartBorderWidth);
    setTxt('vc-opacity-val',    VC.trailOpacity.toFixed(2));
    setTxt('vc-chart-width-val',VC.chartBorderWidth.toFixed(1));
    applyBg(VC.bgColor);
    buildObsTrajectories();
    buildSecondaryTrajectories();
    buildGraphCharts();
  });
}

function applyBg(hex) {
  if (!jScene || !eScene) return;
  jScene.background.set(hex); eScene.background.set(hex);
  jScene.fog.color.set(hex);  eScene.fog.color.set(hex);
}

function wire(id, fn) {
  const el = document.getElementById(id);
  if (el) el.addEventListener('input', e => fn(e.target.value));
}

function set(id, val) {
  const el = document.getElementById(id);
  if (el) el.value = val;
}

function setTxt(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
