// Mutable visual configuration with localStorage persistence.
// All color values are CSS hex strings ('#rrggbb') compatible with both
// Three.js Color.set() and <input type="color">.

const DEFAULTS = {
  bgColor:          '#f0f2f5',
  trailObsLColor:   '#2266cc',
  trailObsRColor:   '#cc2211',
  trailSecLColor:   '#00bbdd',
  trailSecRColor:   '#dd7700',
  trailOpacity:     0.6,
  trailLineWidth:   2,
  sphereScale:      1.0,
  chartBorderWidth: 1.2,
};

export const VC = (() => {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem('vwrVC') || '{}') }; }
  catch { return { ...DEFAULTS }; }
})();

export function saveVC() {
  try { localStorage.setItem('vwrVC', JSON.stringify(VC)); } catch {}
}

export function resetVC() {
  Object.assign(VC, DEFAULTS);
  saveVC();
}
