// Chart.js graphs panel: setup, build, render.

import { S } from './state.js';
import { makeDraggable } from './calibration.js';
import { eeLabels } from './labels.js';
// stopPlayback and updateFrame are imported here; circular references resolve
// at runtime since these are only used inside event callbacks.
import { stopPlayback, updateFrame } from './playback.js';

export const SERIES_COLORS = [
    '#ff6b6b','#4a9eff','#50fa7b','#ffb86c','#bd93f9',
    '#ff79c6','#8be9fd','#f1fa8c','#ff6e6e','#5af78e',
    '#caa9fa','#ffca6a','#1dc9a4','#ff92d0','#6272a4',
    '#44bc9f','#e06c75','#61afef','#98c379','#d19a66',
    '#c678dd','#56b6c2','#e5c07b','#be5046','#2ecc71',
];

export let graphInstances = [];  // Chart.js instances

// Custom plugin: draws the current-frame cursor line + value dots on every chart
export const frameCursorPlugin = {
  id: 'frameCursor',
  afterDatasetsDraw(chart) {
    const { ctx, chartArea, scales } = chart;
    if (!chartArea || !scales.x) return;
    const x = scales.x.getPixelForValue(S.frameIdx);
    if (x < chartArea.left || x > chartArea.right) return;

    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.45)';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(x, chartArea.top);
    ctx.lineTo(x, chartArea.bottom);
    ctx.stroke();

    for (const ds of chart.data.datasets) {
      const v = ds.data[S.frameIdx];
      if (v == null) continue;
      const y = scales.y.getPixelForValue(v);
      ctx.fillStyle = ds.borderColor;
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  },
};

export function setupGraphPanel() {
  const btn   = document.getElementById('graphBtn');
  const panel = document.getElementById('graphPanel');
  btn.addEventListener('click', () => {
    const hidden = panel.classList.toggle('hidden');
    btn.classList.toggle('active', !hidden);
    if (!hidden) renderAllCharts();
  });
  document.getElementById('graphClose').addEventListener('click', () => {
    panel.classList.add('hidden');
    btn.classList.remove('active');
  });
  makeDraggable(panel, panel.querySelector('.graph-panel-hdr'));
}

export function buildGraphCharts() {
  for (const c of graphInstances) c.destroy();
  graphInstances = [];

  const body = document.getElementById('graphPanelBody');
  body.innerHTML = '';
  if (!S.frames.length) return;

  const C = window.Chart;
  if (!C) { console.warn('Chart.js not loaded'); return; }

  const sample = S.frames[0];
  const labels = Array.from({ length: S.frames.length }, (_, i) => i);

  function colData(key, idx) {
    return S.frames.map(f => { const a = f[key]; return (a && idx < a.length) ? a[idx] : null; });
  }

  function addGroup(title, series) {
    const group = document.createElement('div');
    group.className = 'chart-group';

    const cjWrap = document.createElement('div');
    cjWrap.className = 'chart-cj-wrap';

    const canvas = document.createElement('canvas');
    cjWrap.appendChild(canvas);
    group.appendChild(cjWrap);
    body.appendChild(group);

    const inst = new C(canvas, {
      type: 'line',
      data: {
        labels,
        datasets: series.map(s => ({
          label:           s.label,
          data:            s.data,
          borderColor:     s.color,
          backgroundColor: 'transparent',
          borderWidth:     1.2,
          pointRadius:     0,
          tension:         0,
        })),
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        animation:           false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          title: {
            display: true,
            text:    title,
            color:   '#8b949e',
            font:    { size: 9, family: 'Consolas, monospace', weight: '700' },
            padding: { top: 4, bottom: 2 },
            align:   'start',
          },
          legend: {
            labels: {
              color:    '#8b949e',
              font:     { size: 9, family: 'Consolas, monospace' },
              boxWidth: 12,
              boxHeight: 2,
              padding:  6,
            },
          },
          tooltip: {
            backgroundColor: '#161b22',
            borderColor:     '#30363d',
            borderWidth:     1,
            titleColor:      '#8b949e',
            bodyColor:       '#e6edf3',
            bodyFont:        { size: 9, family: 'Consolas, monospace' },
            callbacks: {
              title: items => `frame ${items[0].label}`,
              label: item  => ` ${item.dataset.label}: ${item.parsed.y?.toFixed(4)}`,
            },
          },
          zoom: {
            zoom: { wheel: { enabled: true }, mode: 'x' },
            pan:  { enabled: true, mode: 'x' },
            limits: { x: { min: 'original', max: 'original' } },
          },
        },
        onClick(_evt, _elems, chart) {
          const pts = chart.getElementsAtEventForMode(_evt, 'index', { intersect: false }, false);
          if (!pts.length) return;
          stopPlayback();
          updateFrame(Math.max(0, Math.min(pts[0].index, S.frames.length - 1)));
        },
        scales: {
          x: {
            ticks:  { color: '#484f58', font: { size: 8, family: 'Consolas, monospace' }, maxTicksLimit: 6 },
            grid:   { color: '#21262d' },
            border: { color: '#30363d' },
          },
          y: {
            ticks:  { color: '#484f58', font: { size: 8, family: 'Consolas, monospace' }, maxTicksLimit: 4 },
            grid:   { color: '#21262d' },
            border: { color: '#30363d' },
          },
        },
      },
      plugins: [frameCursorPlugin],
    });

    graphInstances.push(inst);
  }

  function defineNamedArray(frameKey, baseTitle, names) {
    if (!sample[frameKey]) return;
    const arr = sample[frameKey];
    const left = [], right = [], extra = [];
    for (let i = 0; i < arr.length; i++) {
      const name = names[i] || `[${i}]`;
      const dest = name.startsWith('left_') ? left : name.startsWith('right_') ? right : extra;
      dest.push({ i, name });
    }
    for (const [grp, lbl] of [[left,'Left'],[right,'Right'],[extra,'Extra']]) {
      if (!grp.length) continue;
      addGroup(`${baseTitle} — ${lbl}`, grp.map(({ i, name }, j) => ({
        data:  colData(frameKey, i),
        color: SERIES_COLORS[j % SERIES_COLORS.length],
        label: name.replace(/^left_/,'').replace(/^right_/,''),
      })));
    }
  }

  function defineVec(frameKey, title) {
    const arr = sample[frameKey];
    if (!arr) return;
    const len    = arr.length;
    const lbls   = eeLabels(frameKey, len);
    const posC   = ['#ff6b6b', '#69db7c', '#74c0fc'];
    const extraC = ['#bd93f9','#ffb86c','#f1fa8c','#ff79c6','#8be9fd','#caa9fa','#ffca6a','#1dc9a4','#ff92d0','#44bc9f'];
    const colors = [...posC, ...extraC].slice(0, len);
    addGroup(title, Array.from({ length: len }, (_, i) => ({
      data: colData(frameKey, i), color: colors[i], label: lbls[i],
    })));
  }

  defineNamedArray('observation.state', 'State',  S.stateNames);
  defineNamedArray('action',            'Action', S.actionNames);
  defineVec('observation.ee_left',              'Obs EE Left');
  defineVec('observation.ee_right',             'Obs EE Right');
  defineVec('action.ee_left',                   'Action EE Left');
  defineVec('action.ee_right',                  'Action EE Right');
  defineVec('action.ee_left.delta',             'EE Delta Left');
  defineVec('action.ee_right.delta',            'EE Delta Right');
  defineVec('action.ee_left.relative',          'EE Rel Left');
  defineVec('action.ee_right.relative',         'EE Rel Right');
  defineVec('action.ee_left.delta.rotvec',      'EE Delta Rotvec Left');
  defineVec('action.ee_right.delta.rotvec',     'EE Delta Rotvec Right');
  defineVec('action.ee_left.relative.rotvec',   'EE Rel Rotvec Left');
  defineVec('action.ee_right.relative.rotvec',  'EE Rel Rotvec Right');
}

export function renderAllCharts() {
  if (document.getElementById('graphPanel').classList.contains('hidden')) return;
  for (const chart of graphInstances) chart.update('none');
}
