// Frame playback: updateFrame, startPlayback, stopPlayback, tick, setStatus.

import { S } from './state.js';
import { applyJoints } from './robot.js';
import { updateEE } from './overlays.js';
import { updateDataPanel } from './datapanel.js';
import { renderAllCharts } from './graphs.js';

export function updateFrame(idx) {
  S.frameIdx = Math.max(0, Math.min(idx, S.frames.length - 1));
  document.getElementById('frameSlider').value = S.frameIdx;
  document.getElementById('frameCounter').textContent = `${S.frameIdx + 1} / ${S.frames.length}`;
  const f = S.frames[S.frameIdx];
  if (!f) return;
  applyJoints(f);
  updateEE(f);
  updateDataPanel(f);
  renderAllCharts();
}

export function startPlayback() {
  if (S.playing || !S.frames.length) return;
  S.playing = true;
  document.getElementById('playBtn').textContent = '⏸ Pause';
  tick();
}

export function stopPlayback() {
  S.playing = false;
  if (S.playTimer) { clearTimeout(S.playTimer); S.playTimer = null; }
  document.getElementById('playBtn').textContent = '▶ Play';
}

export function tick() {
  if (!S.playing) return;
  S.frameIdx = (S.frameIdx + 1) % S.frames.length;
  updateFrame(S.frameIdx);
  const speed = parseFloat(document.getElementById('speedSelect').value);
  S.playTimer = setTimeout(tick, 1000 / (50 * speed));
}

export function setStatus(msg) {
  document.getElementById('statusMsg').textContent = msg;
}
