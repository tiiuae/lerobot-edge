// Joint calibration UI: loading/saving calibration, draggable panel, snap-to-zero.

import * as K from './constants.js';
import { S } from './state.js';
import { buildValuesByName } from './robot.js';
// Circular imports are fine at runtime — these functions are only called from
// event handlers, never at module evaluation time.
import { updateFrame } from './playback.js';
import { buildSecondaryTrajectories } from './overlays.js';

// ── Calibration persistence ───────────────────────────────────────────────────
export const CALIB_KEY = 'lerobot-viewer-joint-calibration-v3';
export const SIDES = ['left', 'right'];

export function defaultCalib() {
  const c = {};
  for (const side of SIDES)
    for (let i = 0; i <= 6; i++)
      c[`${side}_joint_${i}`] = { offset: 0, sign: 1 };
  return c;
}

export function loadJointCalib() {
  try {
    const s = localStorage.getItem(CALIB_KEY);
    if (s) return { ...defaultCalib(), ...JSON.parse(s) };
  } catch (e) { console.warn('calib load failed', e); }
  return defaultCalib();
}

export function saveJointCalib() {
  try { localStorage.setItem(CALIB_KEY, JSON.stringify(jointCalib)); }
  catch (e) { console.warn('calib save failed', e); }
}

export let jointCalib = loadJointCalib();

export function transformJointValue(dsName, raw) {
  const c = jointCalib[dsName];
  return c ? c.sign * raw + c.offset : raw;
}

// ── Shared drag utility ───────────────────────────────────────────────────────
export function makeDraggable(panel, handle) {
  handle.addEventListener('mousedown', e => {
    if (e.target.tagName === 'BUTTON') return;
    const r = panel.getBoundingClientRect();
    panel.style.left   = r.left + 'px';
    panel.style.top    = r.top  + 'px';
    panel.style.right  = 'unset';
    panel.style.bottom = 'unset';
    let ox = r.left, oy = r.top, sx = e.clientX, sy = e.clientY;
    handle.style.cursor = 'grabbing';
    const onMove = mv => {
      panel.style.left = (ox + mv.clientX - sx) + 'px';
      panel.style.top  = (oy + mv.clientY - sy) + 'px';
    };
    const onUp = () => {
      handle.style.cursor = 'grab';
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup',   onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup',   onUp);
    e.preventDefault();
  });
}

// ── Calibration change handler ────────────────────────────────────────────────
let calibChangeTimer = null;
export function onCalibChange() {
  if (calibChangeTimer) clearTimeout(calibChangeTimer);
  if (S.frames[S.frameIdx]) updateFrame(S.frameIdx);
  calibChangeTimer = setTimeout(() => {
    buildSecondaryTrajectories();
    calibChangeTimer = null;
  }, 80);
}

// ── Calibration UI ────────────────────────────────────────────────────────────
export function setupCalibrationUI() {
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
    const frame = S.frames[S.frameIdx];
    if (!frame) return;
    const state = frame['observation.state'];
    if (!state) return;
    const byName = buildValuesByName(state, K.STATE_IDX);
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
