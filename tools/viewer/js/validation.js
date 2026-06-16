// Quantitative FK-validation: per-frame position/orientation error and episode stats.
import * as THREE from 'three';
import { S } from './state.js';
import { worldPos } from './overlays.js';
import { getEEWorldPose, robot, buildValuesByName, applyRobotJoints } from './robot.js';
import { STATE_IDX } from './constants.js';

const FK_LINKS = {
  left:  'follower_left_ee_gripper_link',
  right: 'follower_right_ee_gripper_link',
};

function obsQuat(pose) {            // pose = [x,y,z,qw,qx,qy,qz,...] — qw at index 3
  return new THREE.Quaternion(pose[4], pose[5], pose[6], pose[3]);
}

// Per-frame FK error for one arm. Returns {posMM, oriDeg} or null.
export function frameError(side, obsPose) {
  if (!robot || !obsPose || obsPose.length < 7) return null;
  const fk = getEEWorldPose(robot, FK_LINKS[side]);
  if (!fk) return null;
  const posMM  = worldPos(obsPose).distanceTo(fk.pos) * 1000;
  const oriDeg = THREE.MathUtils.radToDeg(obsQuat(obsPose).angleTo(fk.quat));
  return { posMM, oriDeg };
}

// Episode-wide stats. Replays state joints through the URDF frame-by-frame.
export function computeEpisodeStats() {
  const stats = { left: null, right: null };
  if (!robot || !S.frames.length) return stats;
  const acc = { left: { p: [], o: [] }, right: { p: [], o: [] } };
  for (const f of S.frames) {
    const st = f['observation.state'];
    if (!st) continue;
    applyRobotJoints(robot, buildValuesByName(st, STATE_IDX));
    for (const side of ['left', 'right']) {
      const obs = f[`observation.ee_${side}`];
      const e   = obs ? frameError(side, obs) : null;
      if (e) { acc[side].p.push(e.posMM); acc[side].o.push(e.oriDeg); }
    }
  }
  const rms = a => a.length ? Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length) : null;
  const max = a => a.length ? Math.max(...a) : null;
  for (const side of ['left', 'right']) {
    if (acc[side].p.length)
      stats[side] = {
        rmsPosMM:  rms(acc[side].p),
        maxPosMM:  max(acc[side].p),
        rmsOriDeg: rms(acc[side].o),
      };
  }
  // restore current frame pose
  const cf = S.frames[S.frameIdx];
  if (cf?.['observation.state'])
    applyRobotJoints(robot, buildValuesByName(cf['observation.state'], STATE_IDX));
  return stats;
}

let _errChart = null;

export function buildErrorChart() {
  if (_errChart) { _errChart.destroy(); _errChart = null; }
  const canvas = document.getElementById('fkErrChart');
  if (!canvas || !window.Chart || !robot || !S.frames.length) return;

  const left = [], right = [];
  for (const f of S.frames) {
    const st = f['observation.state'];
    if (st) applyRobotJoints(robot, buildValuesByName(st, STATE_IDX));
    const el = f['observation.ee_left']  ? frameError('left',  f['observation.ee_left'])  : null;
    const er = f['observation.ee_right'] ? frameError('right', f['observation.ee_right']) : null;
    left.push(el ? el.posMM : null);
    right.push(er ? er.posMM : null);
  }

  // restore current frame
  const cf = S.frames[S.frameIdx];
  if (cf?.['observation.state'])
    applyRobotJoints(robot, buildValuesByName(cf['observation.state'], STATE_IDX));

  _errChart = new window.Chart(canvas, {
    type: 'line',
    data: {
      labels: S.frames.map((_, i) => i),
      datasets: [
        { label: 'L pos err (mm)', data: left,  borderColor: '#2266cc', borderWidth: 1, pointRadius: 0 },
        { label: 'R pos err (mm)', data: right, borderColor: '#cc2211', borderWidth: 1, pointRadius: 0 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {
        legend: { labels: { color: '#8b949e', boxWidth: 10, font: { size: 9, family: 'Consolas, monospace' } } },
        tooltip: { enabled: false },
      },
      scales: {
        x: { ticks: { color: '#484f58', font: { size: 8, family: 'Consolas, monospace' }, maxTicksLimit: 6 },
             grid: { color: '#21262d' } },
        y: { ticks: { color: '#484f58', font: { size: 8, family: 'Consolas, monospace' } },
             beginAtZero: true, grid: { color: '#21262d' } },
      },
    },
  });
}

export function showFKStats() {
  const statsEl = document.getElementById('fkStats');
  const chartWrap = document.getElementById('fkErrChartWrap');
  if (statsEl) statsEl.removeAttribute('hidden');
  if (chartWrap) chartWrap.removeAttribute('hidden');
  const s = computeEpisodeStats();
  const line = side => s[side]
    ? `${side[0].toUpperCase()}: RMS ${s[side].rmsPosMM.toFixed(1)}mm (max ${s[side].maxPosMM.toFixed(1)}) · orient ${s[side].rmsOriDeg.toFixed(2)}°`
    : `${side[0].toUpperCase()}: —`;
  if (statsEl) statsEl.textContent = `Episode FK error   ${line('left')}   ${line('right')}`;
  buildErrorChart();
}

export function hideFKStats() {
  const statsEl = document.getElementById('fkStats');
  const chartWrap = document.getElementById('fkErrChartWrap');
  if (statsEl) statsEl.setAttribute('hidden', '');
  if (chartWrap) chartWrap.setAttribute('hidden', '');
}
