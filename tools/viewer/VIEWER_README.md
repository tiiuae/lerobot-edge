# Robot Dataset Viewer — Documentation

## Overview

A browser-based 3-D visualiser that replays LeRobot datasets on the physical
robot model. It renders two side-by-side Three.js panels and drives them
frame-by-frame from recorded data.

---

## Starting the Server

```bash
# From the lerobot/ root:
python tools/server.py

# With a custom dataset cache directory:
python tools/server.py --cache /path/to/datasets

# Then open:
http://localhost:8080/viewer
```

---

## Two-Panel Layout

### LEFT panel — "Joint Space"

Renders the full URDF robot and drives its joint angles from dataset data.

A **State / Action** toggle controls which column is used to drive the URDF:

| Toggle position | Source column | Typical use |
|---|---|---|
| **State** (default) | `observation.state` | Replay what the robot actually did |
| **Action** | `action` | Replay the commanded joint targets |

Joint values are mapped to URDF joint names through `JOINT_MAP` and
`GRIPPER_MAP` (defined in `js/constants.js`). The camera can be orbited freely;
it syncs with the right panel — the left canvas is **primary** by default
(dragging either canvas syncs both).

### RIGHT panel — "EE Analysis"

Renders the same URDF robot plus end-effector overlays. The **mode selector**
above the canvas switches between four analysis modes:

#### FK Val (FK Validation)

Compares recorded EE poses against live forward-kinematics computed from the
URDF:

| Element | Colour | What it shows |
|---|---|---|
| Sphere + axis frame | Blue / Red | Baseline: `observation.ee_left` / `observation.ee_right` |
| Sphere + axis frame | Cyan / Orange | Candidate: live URDF FK result |
| Line between spheres | Yellow | FK position error vector |

Per-frame error is shown in the footer: `FK err L: X.X mm / X.X°`

Episode RMS and max statistics are displayed in the `#fkStats` bar above the
canvas. An error-vs-time chart is rendered below the canvas.

#### Obs vs Action

Compares observed and commanded EE poses:

| Element | Colour | What it shows |
|---|---|---|
| Sphere + axis frame | Blue / Red | `observation.ee_left` / `observation.ee_right` |
| Sphere + axis frame | Cyan / Orange | `action.ee_left` / `action.ee_right` |
| Line between spheres | Yellow | Tracking gap vector |

#### EE Delta

Shows the delta direction relative to the observed EE pose:

| Element | Colour | What it shows |
|---|---|---|
| Sphere + axis frame | Blue / Red | `observation.ee_left` / `observation.ee_right` |
| Arrow | Green | `action.ee.delta` direction |

#### EE Rel

Shows the relative EE vector from the observed pose:

| Element | Colour | What it shows |
|---|---|---|
| Sphere + axis frame | Blue / Red | `observation.ee_left` / `observation.ee_right` |
| Arrow | Purple | `action.ee.relative` vector |

---

## EE Vector Format

EE columns contain **8-dim** vectors: `[x, y, z, qw, qx, qy, qz, gripper]`

The viewer also reads legacy **7-dim** datasets `[x, y, z, qw, qx, qy, qz]`
(gripper dimension simply absent).

Axis labels are resolved per-dataset by `labels.js:eeLabels(key, len)` which
handles 7-dim, 8-dim, and `.rotvec`-suffixed columns.

---

## Panels and Overlays

| Button | ID | Panel / Modal |
|---|---|---|
| **Values** | `#dataBtn` | Values inspector — raw per-frame data for every column |
| **Graphs** | `#graphBtn` | Chart.js time-series graphs for state/action/EE signals |
| **Calibration** | `#calibBtn` | Joint calibration panel (per-joint offset and sign) |
| **Help** | `#helpBtn` | Help modal with keyboard shortcuts and usage notes |

---

## Camera Sync

Both canvases share camera state. Dragging either canvas syncs both.

The **left canvas is primary** by default. Each animation frame:

1. Primary `OrbitControls.update()` applies user input.
2. `syncCameras()` copies position, quaternion, and orbit target to the secondary.
3. Secondary `OrbitControls.update()` settles at the synced position.

This prevents the two `OrbitControls` instances from fighting each other.

---

## Module Structure

Source lives under `tools/viewer/js/` as native ES modules (no bundler).

| Module | Responsibility |
|---|---|
| `constants.js` | `JOINT_MAP`, `GRIPPER_MAP`, `STATE_IDX`, mount offsets, colour palette |
| `state.js` | Shared mutable singleton `S` (frames, modes, feature flags, names) |
| `labels.js` | `eeLabels(key, len)` — 7/8-dim + `.rotvec`-aware axis labels |
| `scene.js` | Three.js renderers, scenes, cameras, lights, floor, render loop, helpers |
| `robot.js` | URDF loading, joint application, FK helpers, `getEEWorldPose` |
| `overlays.js` | Markers, trails, arrows, `worldPos`, `updateEE`, trajectory builders |
| `validation.js` | Per-frame FK error (mm/deg), episode RMS/max stats, error chart |
| `modes.js` | Mode switching, legend, button availability, `setupModeButtons` |
| `calibration.js` | Joint calibration panel (offset, sign per joint) |
| `datapanel.js` | Values inspector panel (`buildDataPanelSections`, `updateDataPanel`) |
| `graphs.js` | Chart.js time-series graphs panel |
| `playback.js` | Frame playback (play/pause/tick/slider), `updateFrame`, `setStatus` |
| `api.js` | Dataset/episode fetch (`loadDatasets`, `onDatasetChange`, `loadEpisode`) |

---

## Joint Mapping

### Revolute (arm) joints — `JOINT_MAP`

```
left_joint_0  →  follower_left_joint_0   (base rotation,  axis Z)
left_joint_1  →  follower_left_joint_1   (shoulder pitch, axis +Y)
left_joint_2  →  follower_left_joint_2   (upper arm,      axis -Y)
left_joint_3  →  follower_left_joint_3   (forearm,        axis -Y)
left_joint_4  →  follower_left_joint_4   (wrist rotation, axis -Z)
left_joint_5  →  follower_left_joint_5   (wrist roll,     axis +X)
right_joint_0 … 5  →  follower_right_joint_0 … 5  (mirror of left)
```

### Prismatic (gripper) joints — `GRIPPER_MAP`

A single gripper value controls two prismatic carriages:

```
left_joint_6  →  follower_left_right_carriage_joint
               follower_left_left_carriage_joint
right_joint_6 →  follower_right_right_carriage_joint
               follower_right_left_carriage_joint
```

### Fallback indices — `STATE_IDX`

When `observation.state` does not carry named features in the dataset's
`meta/info.json`, the viewer falls back to fixed positions:

```
index 0–5    left_joint_0 … left_joint_5
index 6      left_joint_6  (gripper)
index 7–12   right_joint_0 … right_joint_5
index 13     right_joint_6 (gripper)
```

---

## Server API

| Endpoint | Query params | Returns |
|---|---|---|
| `GET /api/datasets` | — | Array of dataset descriptors |
| `GET /api/episodes` | `dataset=<path>` | `[{episode, frames}]` |
| `GET /api/frames` | `dataset=<path>&episode=<n>` | `{frames: […]}` |

Dataset descriptor fields:

```json
{
  "name": "relative/path/in/cache",
  "path": "/absolute/path",
  "total_episodes": 42,
  "total_frames": 8400,
  "has_ee_left": true,
  "has_ee_right": true,
  "has_action_ee": true,
  "has_ee_delta": false,
  "has_ee_relative": false,
  "state_names": ["left_joint_0", "…"],
  "action_names": ["left_joint_0", "…"]
}
```

`/api/frames` returns all EE columns present in the dataset:

```
observation.state, action,
observation.ee_left, observation.ee_right,
action.ee_left, action.ee_right,
action.ee_left.delta, action.ee_right.delta,
action.ee_left.relative, action.ee_right.relative
```

---

## Coordinate System

Everything lives in **ROS Z-up** space:

- **X** = forward
- **Y** = left
- **Z** = up

`camera.up = (0, 0, 1)` is set so `OrbitControls` orbits naturally around the
vertical Z axis. No global rotation is applied to the robot root.

### Why the base mesh has `rpy="-π/2 0 0"` in the URDF

The mobile base mesh (`base_link.dae`) is a Collada file. `ColladaLoader`
auto-applies a Y-up→Z-up rotation; `URDFLoader` then resets it to identity.
The URDF `<visual><origin rpy="-1.57…">` re-applies exactly the one -π/2
needed — net result is a single clean correction.

Arm link meshes (`link_N.stl`) are plain STL files authored in Z-up and
need no visual-origin rotation.

---

## Static File Layout

```
viewer/
  index.html        HTML shell, importmap (CDN Three.js, local urdf-loader)
  style.css         Dark/light theme, layout
  js/               ES modules (see module table above)
  lib/
    URDFLoader.js   urdf-loader v0.12 (local copy, avoids CORS)
    URDFClasses.js  URDFRobot, URDFJoint, URDFLink class definitions
```
