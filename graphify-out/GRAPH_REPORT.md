# Graph Report - .  (2026-06-17)

## Corpus Check
- 13 files · ~29,718 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 376 nodes · 772 edges · 23 communities (21 shown, 2 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 28 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Joint-to-EE CLI & Config|Joint-to-EE CLI & Config]]
- [[_COMMUNITY_Viewer API & Dataset Browser|Viewer API & Dataset Browser]]
- [[_COMMUNITY_Dataset Wizard Frontend|Dataset Wizard Frontend]]
- [[_COMMUNITY_Joint Transform Constants|Joint Transform Constants]]
- [[_COMMUNITY_Viewer UI Panels|Viewer UI Panels]]
- [[_COMMUNITY_3D Scene Rendering|3D Scene Rendering]]
- [[_COMMUNITY_HTTP Server & Dataset API|HTTP Server & Dataset API]]
- [[_COMMUNITY_Orientation Math|Orientation Math]]
- [[_COMMUNITY_Viewer Modes & Calibration|Viewer Modes & Calibration]]
- [[_COMMUNITY_EE Conversion Docs|EE Conversion Docs]]
- [[_COMMUNITY_Viewer HTML Shell|Viewer HTML Shell]]
- [[_COMMUNITY_Dataset Pipeline Config|Dataset Pipeline Config]]
- [[_COMMUNITY_Robot Workspace Assets|Robot Workspace Assets]]
- [[_COMMUNITY_Data Format Conventions|Data Format Conventions]]
- [[_COMMUNITY_Viewer ES Module Wiring|Viewer ES Module Wiring]]
- [[_COMMUNITY_Robot Arm Materials|Robot Arm Materials]]
- [[_COMMUNITY_Server CLI & Docs|Server CLI & Docs]]
- [[_COMMUNITY_Test Infrastructure|Test Infrastructure]]
- [[_COMMUNITY_Relative Action Convention|Relative Action Convention]]

## God Nodes (most connected - your core abstractions)
1. `Robot Dataset Viewer Documentation (VIEWER_README.md)` - 25 edges
2. `init()` - 19 edges
3. `Robot Dataset Viewer` - 18 edges
4. `updateFrame()` - 16 edges
5. `buildSecondaryTrajectories()` - 13 edges
6. `process_dataset()` - 11 edges
7. `loadEpisode()` - 11 edges
8. `updateEE()` - 11 edges
9. `S` - 11 edges
10. `enrich_table()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `FK Validation Mode (FK Val)` --semantically_similar_to--> `Forward Kinematics (FK) via placo/RobotKinematics`  [INFERRED] [semantically similar]
  tools/viewer/VIEWER_README.md → tools/EE_CONVERSION.md
- `wizard.js (wizard frontend script)` --implements--> `Server-Sent Events Log Streaming`  [INFERRED]
  tools/viewer/wizard.html → tools/README.md
- `Dataset Viewer index.html (shell + importmap)` --implements--> `Robot Dataset Viewer Documentation (VIEWER_README.md)`  [INFERRED]
  tools/viewer/index.html → tools/viewer/VIEWER_README.md
- `wizard.js (wizard frontend script)` --references--> `Server API Endpoints (/api/datasets, /api/episodes, /api/frames)`  [INFERRED]
  tools/viewer/wizard.html → tools/viewer/VIEWER_README.md
- `File Browser Modal (filesystem navigation)` --semantically_similar_to--> `Dataset Folder Browser Modal`  [INFERRED] [semantically similar]
  tools/viewer/wizard.html → tools/viewer/index.html

## Import Cycles
- 3-file cycle: `tools/viewer/js/calibration.js -> tools/viewer/js/playback.js -> tools/viewer/js/datapanel.js -> tools/viewer/js/calibration.js`

## Hyperedges (group relationships)
- **5-Stage Dataset Processing Pipeline** — viewer_wizard_stage_conversion, viewer_wizard_stage_merge, viewer_wizard_stage_ee_conversion, viewer_wizard_stage_compress, viewer_wizard_stage_upload [EXTRACTED 1.00]
- **3D Scene Rendering Stack (Three.js + URDF Loader in Viewer)** — viewer_index_three_js, viewer_index_urdf_loader, viewer_index_joint_space_panel [INFERRED 0.85]
- **Chart Rendering Stack (Chart.js + Hammer.js + zoom plugin)** — viewer_index_chartjs, viewer_index_hammerjs, viewer_index_chartjs_zoom [EXTRACTED 1.00]

## Communities (23 total, 2 thin omitted)

### Community 0 - "Joint-to-EE CLI & Config"
Cohesion: 0.05
Nodes (57): main(), _parse_args(), Standalone CLI: python -m joint_to_ee --dataset ..., Static configuration: URDF paths, joint layout, mounts, gripper limits, dims., _col(), _delta_col_names(), enrich_table(), Per-parquet enrichment: compute and append EE + representation columns. (+49 more)

### Community 1 - "Viewer API & Dataset Browser"
Cohesion: 0.11
Nodes (42): browseDir(), closeBrowser(), initDatasetBrowser(), loadDatasets(), loadEpisode(), onDatasetChange(), openBrowser(), pickDataset() (+34 more)

### Community 2 - "Dataset Wizard Frontend"
Cohesion: 0.10
Nodes (33): ansiToHtml(), api(), apiPost(), appendLog(), browseDir(), buildConfig(), closeFileBrowser(), _datasetMeta (+25 more)

### Community 3 - "Joint Transform Constants"
Cohesion: 0.12
Nodes (33): transformJointValue(), GRIPPER_MAP, JOINT_MAP, LEFT_MOUNT, RIGHT_MOUNT, STATE_IDX, applyPose(), buildActionEETrajectories() (+25 more)

### Community 4 - "Viewer UI Panels"
Cohesion: 0.07
Nodes (35): Joint Calibration Panel, Camera View Presets, Chart.js (v4.4.7), chartjs-plugin-zoom (v2.0.1), Dataset Values Inspector Panel, EE Analysis Panel, EE Mode Buttons (FK Val / Obs-Act / EE Delta / EE Rel), Forward Kinematics Validation (+27 more)

### Community 5 - "3D Scene Rendering"
Cohesion: 0.14
Nodes (20): animate(), applySphereScale(), makeAxisFrame(), makeCamera(), makeFloor(), makeLights(), makeUpdatableLine(), sceneRefs (+12 more)

### Community 6 - "HTTP Server & Dataset API"
Cohesion: 0.18
Nodes (13): BaseHTTPRequestHandler, find_datasets(), find_direct_datasets(), get_episodes(), get_frames(), Handler, list_directory(), main() (+5 more)

### Community 7 - "Orientation Math"
Cohesion: 0.20
Nodes (17): _canonical_wxyz(), orientation_delta_quat(), orientation_delta_rotvec(), Geometrically correct orientation differences for EE delta/relative.  Quaternion, Rotation from ref to cur expressed in the ref frame: R_ref^{-1} * R_cur., Relative orientation as a canonical unit quaternion [qw, qx, qy, qz] (float32)., Relative orientation as a rotation vector (axis * angle, radians) [rx, ry, rz]., relative_rotation() (+9 more)

### Community 8 - "Viewer Modes & Calibration"
Cohesion: 0.14
Nodes (18): calibration.js ES Module (joint calibration panel), Camera Sync (dual OrbitControls primary/secondary), datapanel.js ES Module (values inspector), EE Delta Mode, EE Rel Mode, FK Validation Mode (FK Val), graphs.js ES Module (Chart.js time-series), modes.js ES Module (mode switching, setupModeButtons) (+10 more)

### Community 9 - "EE Conversion Docs"
Cohesion: 0.25
Nodes (11): config.example.yaml — Dataset Wizard Config Template, End-Effector Conversion Guide, Forward Kinematics (FK) via placo/RobotKinematics, EE Reference Frame: arm (arm-local), EE Reference Frame: robot_base (robot-centric), Mobile AI Robot (dual WidowX AI arms on wheeled base), Mount Transform T_mount (arm to robot base offset), RobotKinematics (lerobot.model.kinematics) (+3 more)

### Community 10 - "Viewer HTML Shell"
Cohesion: 0.28
Nodes (9): Landing Page (landing.html) — Tool Selection UI, Chart.js v4.4.7 (CDN), Dataset Folder Browser Modal, Three.js v0.169.0 (CDN via jsDelivr), URDFLoader.js (local copy, avoids CORS), Dataset Viewer index.html (shell + importmap), Dataset Preview UI (Chart.js time-series charts), File Browser Modal (filesystem navigation) (+1 more)

### Community 11 - "Dataset Pipeline Config"
Cohesion: 0.25
Nodes (9): config.yaml Pipeline Configuration File, Dataset Wizard (dataset-wizard.py), LeRobot Dataset API (add_features, LeRobotDataset), Dataset Pipeline Five Stages, SFTP Upload Stage (paramiko), Server-Sent Events Log Streaming, Server API Endpoints (/api/datasets, /api/episodes, /api/frames), Pipeline Flow UI (stage nodes with start/stop selectors) (+1 more)

### Community 12 - "Robot Workspace Assets"
Cohesion: 0.67
Nodes (4): Maple Wood Material, Maple Tabletop Surface Texture, Robot Workspace Surface, Trossen Arm Robot Environment

### Community 13 - "Data Format Conventions"
Cohesion: 0.50
Nodes (4): EE Vector 8-Dimensional Format, Joint Layout Arms-First Convention, Quaternion Storage Convention [qw, qx, qy, qz], labels.js ES Module (eeLabels)

### Community 15 - "Viewer ES Module Wiring"
Cohesion: 0.50
Nodes (4): main.js (viewer ES module entry point), api.js ES Module (loadDatasets, onDatasetChange, loadEpisode), constants.js ES Module, JOINT_MAP / GRIPPER_MAP / STATE_IDX (constants.js)

### Community 16 - "Robot Arm Materials"
Cohesion: 1.00
Nodes (3): Dark Matte Plastic Material, Trossen Black Surface Texture, Trossen Robot Arm

### Community 17 - "Server CLI & Docs"
Cohesion: 0.67
Nodes (3): Dataset Tools Contributing Guide, dataset-server CLI Entry Point, server.py HTTP Server + JSON API

## Knowledge Gaps
- **50 isolated node(s):** `Path`, `Path`, `ndarray`, `Rotation`, `Rotation` (+45 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Robot Dataset Viewer Documentation (VIEWER_README.md)` connect `Viewer Modes & Calibration` to `Viewer HTML Shell`, `Dataset Pipeline Config`, `Data Format Conventions`, `Viewer ES Module Wiring`, `Server CLI & Docs`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Why does `Robot Dataset Viewer` connect `Viewer UI Panels` to `Viewer HTML Shell`, `Viewer ES Module Wiring`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Why does `Dataset Wizard` connect `Viewer UI Panels` to `Viewer HTML Shell`, `Dataset Pipeline Config`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **What connects `joint_to_ee — enrich LeRobot datasets with EE poses + action representations.`, `Standalone CLI: python -m joint_to_ee --dataset ...`, `Static configuration: URDF paths, joint layout, mounts, gripper limits, dims.` to the rest of the system?**
  _83 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Joint-to-EE CLI & Config` be split into smaller, more focused modules?**
  _Cohesion score 0.0546583850931677 - nodes in this community are weakly interconnected._
- **Should `Viewer API & Dataset Browser` be split into smaller, more focused modules?**
  _Cohesion score 0.10784313725490197 - nodes in this community are weakly interconnected._
- **Should `Dataset Wizard Frontend` be split into smaller, more focused modules?**
  _Cohesion score 0.09871794871794871 - nodes in this community are weakly interconnected._