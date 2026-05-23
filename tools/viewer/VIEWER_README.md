# Robot Dataset Viewer — Code Documentation

## Overview

A browser-based 3-D visualiser that replays LeRobot datasets on top of the
physical robot model. It renders two side-by-side Three.js panels, each
containing a fully loaded URDF robot, and drives their joint angles
frame-by-frame from the recorded data.

---

## Architecture

```
browser
  └─ index.html          HTML shell + importmap (CDN Three.js, local urdf-loader)
  └─ main.js             All application logic (ES module)
  └─ style.css           Dark/light theme, layout

server
  └─ server.py           Python stdlib HTTP server
       ├─ /              serves index.html
       ├─ /main.js       serves main.js
       ├─ /robot.urdf    serves the URDF from the Trossen arm description package
       ├─ /lib/          serves urdf-loader (URDFLoader.js + URDFClasses.js)
       ├─ /pkg/          serves mesh files (STL / DAE) from the ROS package path
       └─ /api/          JSON endpoints (datasets, episodes, frames)
```

---

## The Two Panels and What Drives Each

### LEFT panel — "Joint Space"

| Item | Detail |
|------|--------|
| What is rendered | Full URDF robot model |
| What drives it | `observation.state` column from the parquet files |
| Data shape | Array of floats — one value per joint |
| Update rate | Every frame (50 fps real-time, adjustable via speed selector) |

The joint values in `observation.state` are mapped to URDF joint names through
`JOINT_MAP` and `GRIPPER_MAP` (see below).  The Three.js camera on this panel
can be orbited freely; it also syncs with the right panel.

### RIGHT panel — "End-Effector Space"

| Item | Detail |
|------|--------|
| What is rendered | Full URDF robot model (independent clone) **+** EE overlays |
| What drives the robot | Same `observation.state` joints as the left panel |
| What drives the EE overlays | `observation.ee_left` and `observation.ee_right` columns |
| EE data shape | Array of 7 floats: `[x, y, z, qw, qx, qy, qz]` in the robot base frame |

EE overlays (only shown when the dataset contains those columns):

| Overlay | Colour | What it shows |
|---------|--------|---------------|
| Sphere | Blue | Left EE position at the current frame |
| Axis arrows (3×) | Blue shades | Left EE orientation at the current frame |
| Line | Blue (semi-transparent) | Full left EE trajectory across all frames |
| Sphere | Red | Right EE position at the current frame |
| Axis arrows (3×) | Red shades | Right EE orientation at the current frame |
| Line | Red (semi-transparent) | Full right EE trajectory across all frames |
| Two grey spheres | Grey | Fixed arm mount points on the mobile base (reference only) |

---

## Joint Mapping

Dataset state names → URDF joint names are resolved in two places.

### Revolute (arm) joints — `JOINT_MAP`

```
left_joint_0  →  follower_left_joint_0   (base rotation, axis Z)
left_joint_1  →  follower_left_joint_1   (shoulder pitch,  axis +Y)
left_joint_2  →  follower_left_joint_2   (upper arm,       axis −Y)
left_joint_3  →  follower_left_joint_3   (forearm,         axis −Y)
left_joint_4  →  follower_left_joint_4   (wrist rotation,  axis −Z)
left_joint_5  →  follower_left_joint_5   (wrist roll,      axis +X)
right_joint_0 …5  →  follower_right_joint_0 …5   (mirror of left)
```

### Prismatic (gripper) joints — `GRIPPER_MAP`

A single gripper value controls two prismatic carriages that open/close:

```
left_joint_6  →  follower_left_right_carriage_joint
               follower_left_left_carriage_joint
right_joint_6 →  follower_right_right_carriage_joint
               follower_right_left_carriage_joint
```

### Fallback indices — `STATE_IDX`

When `observation.state` does not include named features in the dataset's
`meta/info.json`, the code falls back to fixed positions in the state array:

```
index 0–4   odom_x, odom_y, odom_theta, linear_vel, angular_vel
index 5–10  left_joint_0 … left_joint_5
index 11    left_joint_6  (gripper)
index 12–17 right_joint_0 … right_joint_5
index 18    right_joint_6 (gripper)
```

---

## Coordinate System

Everything lives in native **ROS Z-up** space:

- **X** = forward (robot's heading direction)
- **Y** = left
- **Z** = up

`camera.up = (0, 0, 1)` tells OrbitControls that Z is the vertical axis so
orbiting and panning behave naturally.

No global rotation is applied to the robot root — the URDF kinematic chain is
used directly in Z-up.

### Why the base mesh has `rpy="-π/2 0 0"` in the URDF

The mobile base mesh (`meshes/mobile_ai/base_link.dae`) is a Collada (DAE)
file.  Collada files exported from common CAD tools embed a Y-up → Z-up
rotation inside the file.  ColladaLoader (Three.js) detects this and applies
a −π/2 rotation to the scene node.  URDFLoader then immediately resets any
loaded mesh's quaternion to identity, so the auto-rotation is undone.  The
URDF `<visual><origin rpy="-1.57…">` re-applies exactly the −π/2 that is
needed to display the mesh correctly in Z-up, so the net result is one clean
−π/2 correction — not zero and not double.

The arm link meshes (`meshes/wxai/link_N.stl`) are plain STL files with no
embedded coordinate metadata.  They are authored in Z-up orientation and need
no visual-origin rotation, hence `rpy="0 0 0"` for all of them.

---

## Data Flow (per frame)

```
Parquet files
  │
  └─ /api/frames (server.py)
        │  returns JSON: { frames: [ {observation.state, observation.ee_left,
        │                             observation.ee_right, timestamp, …}, … ] }
        ▼
  loadEpisode() in main.js
        │  stores frames[], resets slider
        ▼
  updateFrame(idx)
        ├─ applyJoints(frame)
        │     buildStateByName()   maps array → { left_joint_0: val, … }
        │     robot.setJointValue(urdfName, val)   ← left panel URDF
        │     robotEE.setJointValue(urdfName, val) ← right panel URDF
        │
        └─ updateEE(frame)
              applyPose(eeLeftFrame,  ee_left[7])   position + quaternion
              applyPose(eeRightFrame, ee_right[7])
              eeLeftMark.position  ←  ee_left[0:3]
              eeRightMark.position ←  ee_right[0:3]
```

---

## URDF Loading

Two independent robot instances are loaded (one per panel) via `loadRobot()`.
Both are loaded in parallel with `Promise.all([loadOne(), loadOne()])` because
Three.js objects cannot be shared between two independent scene graphs.

```
makeURDFLoader()
  loader.packages = { trossen_arm_description: '/pkg/trossen_arm_description' }
  loader.loadMeshCb   ← custom handler for .dae files:
      1. ColladaLoader loads the file (applies Z-up auto-rotation internally)
      2. dae.scene.rotation.set(0,0,0)  — undo the auto-rotation
         (URDFLoader also does obj.quaternion.identity(), so this is redundant
          but makes the intent explicit)
      3. pass dae.scene to done()
      For .stl files the default loader is used (no rotation applied)
```

---

## Camera Sync

Both panels share camera state.  Whichever panel receives the last
`pointerdown` event becomes the **primary**.  Each animation frame:

1. Primary `OrbitControls.update()` is called first (applies user input).
2. `syncCameras()` copies position, quaternion, and orbit target from the
   primary camera to the secondary.
3. Secondary `OrbitControls.update()` is called (no pending input → stays at
   the synced position, with damping settling toward it).

This keeps both views locked together without the two OrbitControls fighting
each other.

---

## Server API

| Endpoint | Query params | Returns |
|----------|-------------|---------|
| `GET /api/datasets` | — | Array of dataset descriptors found under the cache dir |
| `GET /api/episodes` | `dataset=<path>` | Array of `{episode, frames}` objects |
| `GET /api/frames` | `dataset=<path>&episode=<idx>` | `{frames: […], count: N}` |

Dataset descriptor fields:

```json
{
  "name": "relative/path/in/cache",
  "path": "/absolute/path",
  "total_episodes": 42,
  "total_frames": 8400,
  "has_ee_left": true,
  "has_ee_right": true,
  "state_names": ["odom_x", …, "left_joint_0", …],
  "action_names": ["left_joint_0", …]
}
```

`has_ee_left` / `has_ee_right` control whether EE overlays are shown in the
right panel.  Datasets that lack EE columns still render correctly — only the
joint-space animation is active.

---

## File Reference

| File | Purpose |
|------|---------|
| `server.py` | Python HTTP server; serves static files + API |
| `index.html` | HTML shell; importmap; help modal markup |
| `main.js` | All Three.js/URDF/playback logic |
| `style.css` | Layout, dark theme variables, modal styles |
| `lib/URDFLoader.js` | urdf-loader v0.12 (local copy, avoids CORS) |
| `lib/URDFClasses.js` | URDFRobot, URDFJoint, URDFLink class definitions |

---

## Running

```bash
# From the lerobot/ root or viewer/ directory:
python viewer/server.py [--port 8080] [--cache ~/.cache/huggingface/lerobot]

# Then open:
http://localhost:8080
```

The `--cache` flag points to the directory that contains your downloaded
LeRobot datasets (each must have a `meta/info.json` file to be listed).
