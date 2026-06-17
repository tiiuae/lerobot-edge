// Dataset and episode fetching + folder-browser dataset selector.

import { S } from './state.js';
import { buildObsTrajectories, buildSecondaryTrajectories } from './overlays.js';
import { buildDataPanelSections } from './datapanel.js';
import { buildGraphCharts } from './graphs.js';
import { stopPlayback, setStatus, updateFrame } from './playback.js';
import { updateModeButtonAvailability } from './modes.js';
import { showFKStats } from './validation.js';

// Current dataset info (path, has_ee_* flags, etc.)
let _currentInfo = null;
// Path currently shown in the file browser
let _browserPath = null;

// ── Startup: auto-load first dataset ─────────────────────────────────────────

export async function loadDatasets() {
  const list = await fetch('/api/datasets').then(r => r.json());
  if (!list.length) {
    setDsDisplay('No datasets found');
    setStatus('No datasets found in cache.');
    return;
  }
  await selectDatasetInfo(list[0]);
}

// ── Dataset info → state → load episodes ─────────────────────────────────────

async function selectDatasetInfo(info) {
  _currentInfo = info;
  setDsDisplay(info.name || info.path.split('/').pop());

  S.hasEELeft   = info.has_ee_left    ?? false;
  S.hasEERight  = info.has_ee_right   ?? false;
  S.hasActionEE = info.has_action_ee  ?? false;
  S.hasEEDelta  = info.has_ee_delta   ?? false;
  S.hasEERel    = info.has_ee_relative?? false;
  S.stateNames  = info.state_names    ?? [];
  S.actionNames = info.action_names   ?? [];

  updateModeButtonAvailability();

  const episodes = await fetch(`/api/episodes?dataset=${encodeURIComponent(info.path)}`).then(r => r.json());
  const epSel    = document.getElementById('episodeSelect');
  epSel.innerHTML = episodes
    .map(e => `<option value="${e.episode}">Ep ${e.episode}  (${e.frames} frames)</option>`)
    .join('');
  epSel.onchange = () => loadEpisode(info.path, +epSel.value);
  await loadEpisode(info.path, episodes[0]?.episode ?? 0);
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
  if (S.rightMode === 'fk') showFKStats();
  updateFrame(0);
  setStatus(`${S.frames.length} frames at 50 fps  ·  ${(S.frames.length / 50).toFixed(1)} s`);
}

// ── File browser ─────────────────────────────────────────────────────────────

export function initDatasetBrowser() {
  document.getElementById('dsBrowseBtn')
    .addEventListener('click', openBrowser);
  document.getElementById('dsBrowserClose')
    .addEventListener('click', closeBrowser);
  document.getElementById('dsBrowserOverlay')
    .addEventListener('click', e => { if (e.target === e.currentTarget) closeBrowser(); });
}

function openBrowser() {
  // Start in the parent of the current dataset, or home if none loaded yet.
  const startPath = _currentInfo
    ? _currentInfo.path.replace(/\/[^/]+\/?$/, '') || '~'
    : '~';
  document.getElementById('dsBrowserOverlay').classList.remove('hidden');
  browseDir(startPath);
}

function closeBrowser() {
  document.getElementById('dsBrowserOverlay').classList.add('hidden');
}

async function browseDir(path) {
  const entries = document.getElementById('dsBrowserEntries');
  entries.innerHTML = '<div class="ds-browser-msg">Loading…</div>';

  try {
    const result = await fetch(`/api/files?path=${encodeURIComponent(path)}`).then(r => r.json());

    if (result.error) {
      entries.innerHTML = `<div class="ds-browser-msg err">${result.error}</div>`;
      return;
    }

    _browserPath = result.path;
    document.getElementById('dsBrowserPath').textContent = result.path;

    let html = '';
    if (result.parent) {
      html += `<div class="ds-browser-item ds-browser-up" data-path="${result.parent}">
        <span class="ds-bi-icon">↑</span><span class="ds-bi-name">..</span>
      </div>`;
    }
    for (const e of result.entries) {
      const badge = e.is_dataset ? '<span class="ds-badge">dataset</span>' : '';
      html += `<div class="ds-browser-item${e.is_dataset ? ' is-dataset' : ''}"
          data-path="${e.path}" data-is-dataset="${e.is_dataset}">
        <span class="ds-bi-icon">📁</span>
        <span class="ds-bi-name">${e.name}</span>
        ${badge}
      </div>`;
    }
    if (!result.entries.length) {
      html += '<div class="ds-browser-msg">Empty directory</div>';
    }

    entries.innerHTML = html;
    entries.querySelectorAll('.ds-browser-item').forEach(item => {
      item.addEventListener('click', async () => {
        if (item.dataset.isDataset === 'true') {
          await pickDataset(item.dataset.path);
        } else {
          browseDir(item.dataset.path);
        }
      });
    });
  } catch (e) {
    entries.innerHTML = `<div class="ds-browser-msg err">Error: ${e.message}</div>`;
  }
}

async function pickDataset(datasetPath) {
  closeBrowser();
  setStatus('Loading dataset…');
  try {
    // /api/datasets?path=<datasetPath> runs rglob from that dir and returns its own meta
    const list = await fetch(`/api/datasets?path=${encodeURIComponent(datasetPath)}`).then(r => r.json());
    const info = list.find(d => d.path === datasetPath) ?? list[0];
    if (!info) { setStatus('Could not read dataset metadata.'); return; }
    await selectDatasetInfo(info);
  } catch (e) {
    setStatus(`Failed to load dataset: ${e.message}`);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function setDsDisplay(text) {
  const el = document.getElementById('dsDisplay');
  if (el) { el.textContent = text; el.title = text; }
}
