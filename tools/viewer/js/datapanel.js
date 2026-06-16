// Data inspector panel: setup, building sections, per-frame update.

import { S } from './state.js';
import { makeDraggable } from './calibration.js';

export function setupDataPanel() {
  const btn   = document.getElementById('dataBtn');
  const panel = document.getElementById('dataPanel');
  btn.addEventListener('click', () => {
    const hidden = panel.classList.toggle('hidden');
    btn.classList.toggle('active', !hidden);
    if (!hidden && S.frames[S.frameIdx]) updateDataPanel(S.frames[S.frameIdx]);
  });
  document.getElementById('dataClose').addEventListener('click', () => {
    panel.classList.add('hidden');
    btn.classList.remove('active');
  });
  makeDraggable(panel, panel.querySelector('.data-panel-hdr'));
}

export function buildDataPanelSections() {
  const body = document.getElementById('dataPanelBody');
  body.innerHTML = '';
  S.dataValMap = {};
  if (!S.frames.length) return;

  const sample = S.frames[0];

  function makeSec(title) {
    const sec = document.createElement('div');
    sec.className = 'data-section';
    const hdr = document.createElement('div');
    hdr.className = 'data-section-hdr';
    hdr.textContent = title;
    sec.appendChild(hdr);
    return sec;
  }

  function addSubHdr(parent, text, cls) {
    const el = document.createElement('div');
    el.className = `data-sub-hdr ${cls}`;
    el.textContent = text;
    parent.appendChild(el);
  }

  function addRow(parent, label, elId) {
    const row = document.createElement('div');
    row.className = 'data-row';
    row.innerHTML = `<span class="dk">${label}</span><span class="dv" id="${elId}">—</span>`;
    parent.appendChild(row);
    return row.querySelector('.dv');
  }

  // Named array section (state / action) — groups values by left_*/right_*/other
  function buildNamedArray(frameKey, title, names) {
    const arr = sample[frameKey];
    if (!arr || !Array.isArray(arr)) return;
    const sec = makeSec(title);
    const els = [];

    const groups = { left: [], right: [], extra: [] };
    for (let i = 0; i < arr.length; i++) {
      const name = names[i] || `[${i}]`;
      const grp  = name.startsWith('left_') ? 'left'
                 : name.startsWith('right_') ? 'right' : 'extra';
      groups[grp].push({ i, name });
    }

    for (const [grp, cls, label] of [['left','sub-left','LEFT ARM'],['right','sub-right','RIGHT ARM'],['extra','sub-extra','EXTRA']]) {
      if (!groups[grp].length) continue;
      addSubHdr(sec, label, cls);
      for (const { i, name } of groups[grp]) {
        const lbl = name.replace(/^left_/, '').replace(/^right_/, '');
        els[i] = addRow(sec, lbl, `dv-${frameKey.replace(/\./g,'-')}-${i}`);
      }
    }
    body.appendChild(sec);
    S.dataValMap[frameKey] = els;
  }

  // Pose / vector section — fixed labels based on array length
  function buildVecSection(frameKey, title) {
    const arr = sample[frameKey];
    if (!arr || !Array.isArray(arr)) return;
    const labels = arr.length === 7 ? ['x','y','z','qw','qx','qy','qz']
                 : arr.length === 3 ? ['dx','dy','dz']
                 : arr.map((_, i) => `[${i}]`);
    const sec = makeSec(title);
    const key = frameKey.replace(/[.\s]/g, '-');
    const els = labels.map((l, i) => addRow(sec, l, `dv-${key}-${i}`));
    body.appendChild(sec);
    S.dataValMap[frameKey] = els;
  }

  buildNamedArray('observation.state', 'OBS STATE',  S.stateNames);
  buildNamedArray('action',            'ACTION',      S.actionNames);
  buildVecSection('observation.ee_left',         'OBS EE LEFT');
  buildVecSection('observation.ee_right',        'OBS EE RIGHT');
  buildVecSection('action.ee_left',              'ACTION EE LEFT');
  buildVecSection('action.ee_right',             'ACTION EE RIGHT');
  buildVecSection('action.ee_left.delta',        'EE DELTA LEFT');
  buildVecSection('action.ee_right.delta',       'EE DELTA RIGHT');
  buildVecSection('action.ee_left.relative',     'EE RELATIVE LEFT');
  buildVecSection('action.ee_right.relative',    'EE RELATIVE RIGHT');
}

export function updateDataPanel(frame) {
  if (document.getElementById('dataPanel').classList.contains('hidden')) return;
  for (const [key, els] of Object.entries(S.dataValMap)) {
    const data = frame[key];
    if (!data || !Array.isArray(data)) continue;
    for (let i = 0; i < els.length; i++) {
      if (els[i] && i < data.length) els[i].textContent = data[i].toFixed(4);
    }
  }
}
