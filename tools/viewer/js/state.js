// Shared mutable application state. Modules import S and read/write its fields.
export const S = {
  frames: [], frameIdx: 0, playing: false, playTimer: null,
  leftMode: 'state', rightMode: 'fk',
  hasEELeft: false, hasEERight: false, hasActionEE: false,
  hasEEDelta: false, hasEERel: false,
  stateNames: [], actionNames: [],
  dataValMap: {},
};
