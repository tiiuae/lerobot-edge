// Mode switching, legend, and button availability.

import { S } from './state.js';
import { buildSecondaryTrajectories } from './overlays.js';
import { updateFrame, setStatus } from './playback.js';

export function setLeftMode(mode) {
  S.leftMode = mode;
  document.querySelectorAll('.left-mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
  // Rebuild FK trails when switching (FK validation is still state-based, but
  // left panel trajectory label changes)
  buildSecondaryTrajectories();
  if (S.frames[S.frameIdx]) updateFrame(S.frameIdx);
}

export function setRightMode(mode) {
  S.rightMode = mode;
  document.querySelectorAll('.right-mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
  updateModeLegend();
  buildSecondaryTrajectories();
  if (S.frames[S.frameIdx]) updateFrame(S.frameIdx);
}

export function updateModeLegend() {
  const legend = document.getElementById('eeLegend');
  if (!legend) return;
  const items = {
    fk: `
      <span class="dot obs-l"></span>Obs EE L
      <span class="dot obs-r"></span>Obs EE R
      <span class="dot sec-l"></span>FK EE L
      <span class="dot sec-r"></span>FK EE R
      <span class="dot err"></span>FK Error`,
    obs_action: `
      <span class="dot obs-l"></span>Obs EE L
      <span class="dot obs-r"></span>Obs EE R
      <span class="dot sec-l"></span>Action EE L
      <span class="dot sec-r"></span>Action EE R
      <span class="dot err"></span>Gap`,
    ee_delta: `
      <span class="dot obs-l"></span>Obs EE L
      <span class="dot obs-r"></span>Obs EE R
      <span class="dot delta"></span>Δ direction (left)
      <span class="dot delta"></span>Δ direction (right)`,
    ee_relative: `
      <span class="dot obs-l"></span>Obs EE L
      <span class="dot obs-r"></span>Obs EE R
      <span class="dot rel"></span>Gap vector (left)
      <span class="dot rel"></span>Gap vector (right)`,
  };
  legend.innerHTML = items[S.rightMode] ?? '';
}

// Update which mode buttons are unavailable based on dataset features.
// We use a CSS class instead of the HTML disabled attribute so pointer events
// still fire and we can show a helpful status message on click.
export function updateModeButtonAvailability() {
  document.querySelectorAll('.right-mode-btn').forEach(b => {
    let unavailable = false;
    if (b.dataset.mode === 'obs_action'  && !S.hasActionEE) unavailable = true;
    if (b.dataset.mode === 'ee_delta'    && !S.hasEEDelta)  unavailable = true;
    if (b.dataset.mode === 'ee_relative' && !S.hasEERel)    unavailable = true;
    b.classList.toggle('unavailable', unavailable);
    b.title = unavailable
      ? 'Not available for this dataset — re-run joint_to_ee.py with --include-action'
      : '';
    // Fall back to fk if current mode becomes unavailable
    if (unavailable && S.rightMode === b.dataset.mode) setRightMode('fk');
  });
  document.querySelectorAll('.left-mode-btn').forEach(b =>
    b.classList.remove('unavailable'));
}

export function setupModeButtons() {
  document.querySelectorAll('.left-mode-btn').forEach(b =>
    b.addEventListener('click', () => setLeftMode(b.dataset.mode)));

  document.querySelectorAll('.right-mode-btn').forEach(b =>
    b.addEventListener('click', () => {
      if (b.classList.contains('unavailable')) {
        setStatus('Mode unavailable — re-run joint_to_ee.py with --include-action to enrich this dataset');
        return;
      }
      setRightMode(b.dataset.mode);
    }));

  // Set initial active state
  document.querySelectorAll('.left-mode-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.mode === S.leftMode));
  document.querySelectorAll('.right-mode-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.mode === S.rightMode));

  updateModeLegend();
}
