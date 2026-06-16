// Dataset and episode fetching.

import { S } from './state.js';
import { buildObsTrajectories, buildSecondaryTrajectories } from './overlays.js';
import { buildDataPanelSections } from './datapanel.js';
import { buildGraphCharts } from './graphs.js';
import { stopPlayback, setStatus, updateFrame } from './playback.js';
import { updateModeButtonAvailability } from './modes.js';

export async function loadDatasets() {
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

export async function onDatasetChange() {
  const opt  = document.getElementById('datasetSelect').selectedOptions[0];
  if (!opt) return;
  const info = JSON.parse(opt.dataset.info);

  S.hasEELeft   = info.has_ee_left    ?? false;
  S.hasEERight  = info.has_ee_right   ?? false;
  S.hasActionEE = info.has_action_ee  ?? false;
  S.hasEEDelta  = info.has_ee_delta   ?? false;
  S.hasEERel    = info.has_ee_relative?? false;
  S.stateNames  = info.state_names    ?? [];
  S.actionNames = info.action_names   ?? [];

  updateModeButtonAvailability();

  const episodes = await fetch(`/api/episodes?dataset=${encodeURIComponent(opt.value)}`).then(r => r.json());
  const epSel    = document.getElementById('episodeSelect');
  epSel.innerHTML = episodes
    .map(e => `<option value="${e.episode}">Ep ${e.episode}  (${e.frames} frames)</option>`)
    .join('');
  epSel.onchange = () => loadEpisode(opt.value, +epSel.value);
  await loadEpisode(opt.value, episodes[0]?.episode ?? 0);
}

export async function loadEpisode(datasetPath, epIdx) {
  stopPlayback();
  setStatus('Loading episode…');

  const data = await fetch(
    `/api/frames?dataset=${encodeURIComponent(datasetPath)}&episode=${epIdx}`
  ).then(r => r.json());

  S.frames   = data.frames ?? [];
  S.frameIdx = 0;

  const slider = document.getElementById('frameSlider');
  slider.max   = Math.max(0, S.frames.length - 1);
  slider.value = 0;

  buildObsTrajectories();
  buildSecondaryTrajectories();
  buildDataPanelSections();
  buildGraphCharts();
  updateFrame(0);
  setStatus(`${S.frames.length} frames at 50 fps  ·  ${(S.frames.length / 50).toFixed(1)} s`);
}
