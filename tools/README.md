# Dataset Wizard Guide

The `dataset-wizard.py` script provides a complete pipeline for managing and merging robotic datasets. It handles dataset format conversion, merging multiple datasets, compression, and uploading to remote SFTP servers.

## Overview

The Dataset Wizard performs four main stages:

1. **Conversion** - Convert datasets from v2.1 to v3.0 format
2. **Merge** - Load and merge individual datasets into a single unified dataset
3. **EE Conversion** *(opt-in)* - Compute end-effector poses from joint positions using the WidowX AI arm forward kinematics, and add them as new columns to the merged dataset
4. **Upload** - Compress and upload the merged dataset to an SFTP server

You can start the pipeline from any stage using the `--start-from` option, allowing you to skip previously completed steps.

## Prerequisites

Follow the original README.md file to install lerobot.

## Configuration: config.yaml

All pipeline settings are configured in `config.yaml`. A minimal example:

```yaml
# Pipeline control
start_from: conversion   # Options: conversion | merge | ee_conversion | upload
stop_at: upload

# Dataset location
base_path: ~/.cache/huggingface/lerobot/my-user

# Output dataset name
merged_name: my-merged-dataset

# Datasets to merge (folder names relative to base_path)
datasets:
  - my-dataset-1
  - my-dataset-2
  - my-dataset-3

# SFTP upload
sftp:
  hostname: sftp.example.com
  port: 22
  username: your_username
  password: your_password
  remote_path: /remote/datasets/
```

The `datasets` list must contain directory names exactly as they appear under `base_path`. The script skips any entry whose directory is not found with a warning.

All datasets must be in v2.1 or v3.0 format — v2.1 datasets are converted automatically.

> **Security:** `config.yaml` may contain SFTP credentials. Add it to `.gitignore` and never commit it to version control.

## Configuration: EE Conversion Options

The following keys can be set in `config.yaml` to control the EE conversion stage:

```yaml
# ── EE conversion (optional) ──────────────────────────────────────────────────
# Omit or set to null to skip EE conversion entirely.
ee_frame: arm           # Options: arm | robot_base
ee_include_action: false  # Set to true to also convert action joints
```

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `ee_frame` | No | *(unset — stage skipped)* | Reference frame for EE poses. `arm` = each arm's own `base_link`; `robot_base` = robot `base_link` (includes mount offset). |
| `ee_include_action` | No | `false` | When `true`, EE poses are also computed from `action` joint columns and written as `action.ee_left` / `action.ee_right`. |

CLI flags `--ee-frame` and `--ee-include-action` override these config values.

## Running the Script

### Basic Usage

Run the complete pipeline from start to finish:

```bash
python dataset-wizard.py
```

This will:
1. Convert datasets from v2.1 to v3.0 format (if needed)
2. Merge all configured datasets
3. Compress and upload the result to the SFTP server

### Command-Line Options

#### `--start-from` - Start from a specific pipeline stage

Skip earlier stages and start from a specific point:

```bash
# Start from merge stage (skip conversion)
python dataset-wizard.py --start-from merge

# Start from upload stage (skip all earlier stages)
python dataset-wizard.py --start-from upload

# Run only EE conversion on an already-merged dataset
python dataset-wizard.py --start-from ee_conversion --stop-at ee_conversion --ee-frame arm
```

**Options:**
- `conversion` - Start from dataset format conversion (default)
- `merge` - Skip conversion, start from dataset merging
- `ee_conversion` - Skip conversion and merge, run only EE conversion
- `upload` - Skip all earlier stages, start from compression and upload

#### `--stop-at` - Stop after a specific pipeline stage

Run only up to (and including) a given stage, skipping the rest:

```bash
# Run conversion and merge only (skip EE conversion and upload)
python dataset-wizard.py --stop-at merge

# Merge and compute EE poses, skip upload
python dataset-wizard.py --stop-at ee_conversion --ee-frame arm
```

**Options:**
- `conversion` - Run only the conversion stage
- `merge` - Stop after merging
- `ee_conversion` - Stop after EE conversion (skip upload)
- `upload` - Run all stages through upload (default)

Both options can be combined freely:

```bash
# Run only the merge stage
python dataset-wizard.py --start-from merge --stop-at merge
```

#### `--ee-frame` - Enable EE conversion and set the reference frame

Enables the EE conversion stage and sets the coordinate frame for the output poses.
Without this flag (or `ee_frame` in config.yaml), the EE conversion stage is skipped.

```bash
# Compute EE poses in each arm's own base frame (arm-centric)
python dataset-wizard.py --ee-frame arm

# Compute EE poses in the robot base_link frame (includes arm mount offset)
python dataset-wizard.py --ee-frame robot_base
```

**Options:**
- `arm` - EE poses are expressed in each arm's `base_link` frame
- `robot_base` - EE poses are expressed in the robot's `base_link` frame (left arm mounted at xyz=`0.331, 0.3, 0.831`; right arm at `0.331, -0.3, 0.831`)

#### `--ee-include-action` - Also convert action joints to EE poses

When set, EE poses are also computed for the action joint columns and written as `action.ee_left` / `action.ee_right`.

```bash
python dataset-wizard.py --ee-frame arm --ee-include-action
```

### Complete Example

```bash
# Run from merge stage only, skipping conversion
python dataset-wizard.py --start-from merge
```

This will:
- Skip the conversion stage
- Merge datasets listed in `config.yaml`
- Output the merged dataset to `{base_path}/{merged_name}/`
- Compress and upload the merged result to the SFTP server

## Pipeline Stages Explained

### Stage 1: Conversion (v2.1 → v3.0)

Converts dataset format from v2.1 to v3.0 using the `convert_dataset` function.

**What happens:**
- Scans each dataset in the base path
- Converts format if necessary
- Stores converted dataset in the same location

**Triggered by:**
```bash
python dataset-wizard.py --start-from conversion
```

**Skip if:**
- Your datasets are already in v3.0 format
- You've already run the conversion stage

### Stage 2: Merge

Loads individual datasets and merges them into a single unified dataset.

**What happens:**
- Loads each dataset from the base path
- Consolidates video, data, and metadata
- Creates a new merged dataset at the output directory
- Displays episode and frame statistics

**Triggered by:**
```bash
python dataset-wizard.py --start-from merge
```

**Output:**
- Merged dataset directory at `{base_path}/{merged_name}/`
- Statistics showing total episodes and frames

### Stage 3: EE Conversion *(opt-in)*

Computes end-effector (EE) poses for each wxai arm using `lerobot.model.kinematics.RobotKinematics` (placo-backed FK over the [trossen_arm_description URDF](https://github.com/TrossenRobotics/trossen_arm_description/blob/main/urdf/macros/_wxai.urdf.xacro)).
The single-arm URDF `wxai_follower.urdf` is loaded once; the mount transform is applied separately for the `robot_base` frame.

**What gets added** (float32 list of 7 values per frame):

| New column | Content | Condition |
|---|---|---|
| `observation.ee_left` | `[x, y, z, qw, qx, qy, qz]` | Left arm joints found |
| `observation.ee_right` | `[x, y, z, qw, qx, qy, qz]` | Right arm joints found |
| `action.ee_left` | `[x, y, z, qw, qx, qy, qz]` | `--ee-include-action` set |
| `action.ee_right` | `[x, y, z, qw, qx, qy, qz]` | `--ee-include-action` set |

**Joint mapping**: the wxai mobile_ai datasets have **mislabeled** `state_names` / `action_names` metadata. The actual column layout is:

- `observation.state[0..6]` = `left_joint_0..6`,  `state[7..13]` = `right_joint_0..6`,  `state[14..18]` = base info
- `action[0..6]`            = `left_joint_0..6`,  `action[7..13]` = `right_joint_0..6`,  `action[14..15]` = `linear_vel, angular_vel`

`joint_6` (gripper, prismatic) is excluded from FK — EE is the `ee_gripper_link` at 156 mm past `link_6`. The conversion uses the **`arms-first`** layout by default, which slices by these fixed indices and ignores the bad names. See [EE_CONVERSION.md](EE_CONVERSION.md) for the full table and the `--joint-layout` flag.

**Dependencies**: `placo` is required (`pip install placo`, or `pip install 'lerobot[kinematics]'`). The script sets `ROS_PACKAGE_PATH` to `/home/edgeai/trossen_arm_ros` at import time so placo can resolve `package://` mesh paths.

**How to enable**: set `ee_frame` in `config.yaml` or pass `--ee-frame` on the CLI. Without this, the stage is skipped even if it falls within the `start_from`/`stop_at` range.

**Triggered by:**
```bash
python dataset-wizard.py --ee-frame arm          # arm base frame (default choice)
python dataset-wizard.py --ee-frame robot_base   # robot base_link frame
```

**Output**: columns are added in-place to every parquet file in the merged dataset, and `meta/info.json` is updated with the new feature definitions.

### Stage 4: Upload

Compresses the merged dataset and uploads it to the SFTP server.

**What happens:**
- Creates a ZIP file of the merged dataset
- Connects to the SFTP server using credentials from `config.yaml`
- Uploads the ZIP file with progress tracking
- Verifies successful upload

**Triggered by:**
```bash
python dataset-wizard.py --start-from upload
```

**Note:** When starting from `upload`, the script uses `{merged-name}` as the source directory.

**Requirements:**
- Valid `sftp` block in `config.yaml` with hostname, username, password, and remote_path
- `sftp.remote_path` directory must exist on the server
- Sufficient disk space on SFTP server
- Network connectivity to SFTP server

## Output Files

### During Processing
- **Converted datasets:** Stored in-place at `{base-path}/{dataset-id}/`
- **Merged dataset:** Created at `{base-path}/{merged-name}/`
- **Compressed file:** Created as `{base-path}/{merged-name}.zip`

### After Upload
- **Remote file:** Uploaded to `{sftp.remote_path}/{merged_name}.zip`

## Troubleshooting

### SFTP connection refused
```
Error: Connection refused on hostname:port
```
**Checklist:**
- Verify `sftp.hostname` and `sftp.port` in `config.yaml` are correct
- Check network connectivity to the SFTP server
- Ensure the SFTP server is running
- Check firewall rules allowing outbound connections

### Authentication failed
```
Error: Authentication failed with username/password
```
**Solution:**
- Verify `sftp.username` and `sftp.password` in `config.yaml`
- Check for special characters that may need escaping
- Ensure the SFTP account has permission to connect

### Remote path error
```
ValueError: sftp.remote_path is not set in the config file.
```
**Solution:**
- Add `remote_path: /path/to/remote/dir/` under the `sftp:` block in `config.yaml`
- Verify the directory exists on the SFTP server

### Insufficient disk space
```
Error: No space left on device
```
**Solution:**
- Check available disk space on local and remote systems
- Consider running each stage separately to clean up intermediate files
- Delete the ZIP file after successful upload if needed

### Dataset conversion errors
```
Error: Could not convert dataset from v2.1 to v3.0
```
**Solution:**
- Verify datasets are in the correct format
- Check that the dataset directories are accessible
- Review conversion logs for specific errors

## Example Workflows

### Workflow 1: Convert and Merge Only (No Upload)

Set `stop_at: merge` in `config.yaml`, then run:

```bash
python dataset-wizard.py
# or override on the CLI:
python dataset-wizard.py --stop-at merge
```

Then later, upload only the final result:

```bash
python dataset-wizard.py --start-from upload
```

### Workflow 2: Re-upload Without Re-merging

If the initial upload failed but the merge was successful:

```bash
python dataset-wizard.py --start-from upload
```

The previously merged dataset will be recompressed and uploaded.

### Workflow 3: Merge, Add EE Poses, Then Upload

Add EE conversion between merge and upload using `config.yaml`:

```yaml
# config.yaml
ee_frame: arm
ee_include_action: false
start_from: conversion
stop_at: upload
```

```bash
python dataset-wizard.py
```

Or pass the frame on the CLI without touching `config.yaml`:

```bash
python dataset-wizard.py --ee-frame arm
```

### Workflow 4: Add EE Poses to an Already-Merged Dataset

If the dataset is already merged and you just want to add EE columns, set both `start_from` and `stop_at` to `ee_conversion`.

**Option A — via `config.yaml`:**

```yaml
# config.yaml
start_from: ee_conversion
stop_at: ee_conversion
ee_frame: robot_base       # or arm
ee_include_action: false
```

```bash
python dataset-wizard.py
```

**Option B — CLI flags only (no config change needed):**

```bash
# robot_base frame (default, both arms share one coordinate system)
python dataset-wizard.py \
  --start-from ee_conversion \
  --stop-at ee_conversion \
  --ee-frame robot_base

# arm-local frame
python dataset-wizard.py \
  --start-from ee_conversion \
  --stop-at ee_conversion \
  --ee-frame arm
```

Both options modify the merged dataset in-place (atomic temp-dir swap) without re-running merge or upload. The script will raise an error if EE columns already exist, preventing accidental double-conversion.

### Workflow 5: Full Pipeline

Run the complete pipeline as configured in `config.yaml`:

```bash
python dataset-wizard.py
```

## Dataset Structure

The script expects the following directory structure:

```
base-path/
├── dataset-1/
│   ├── videos/
│   ├── metadata.json
│   └── [dataset files]
├── dataset-2/
│   ├── videos/
│   ├── metadata.json
│   └── [dataset files]
└── ... more datasets...
```

**Important:** The folder names under `base_path` must match the entries in the `datasets` list in `config.yaml`. For example, if your datasets are stored as:

```
~/.cache/huggingface/lerobot/my-user/
├── move-blue-cup-feb12-v1.1/
├── move-blue-cup-feb12-v2.1/
└── move-green-cup-13feb-v1.2/
```

Then `config.yaml` should have:

```yaml
base_path: ~/.cache/huggingface/lerobot/my-user
datasets:
  - move-blue-cup-feb12-v1.1
  - move-blue-cup-feb12-v2.1
  - move-green-cup-13feb-v1.2
```

After running, the structure will include:

```
base-path/
├── [original datasets above]
└── merged-name/
    ├── videos/
    ├── data/
    └── meta/
```

## Performance Tips

1. **Network speed:** Upload speed depends on your internet connection. Large datasets may take considerable time.
2. **Local storage:** Ensure you have enough disk space for both the original and merged datasets.
3. **Batch processing:** If merging many datasets, consider breaking large operations into smaller batches.
4. **Restart capability:** Use `--start-from` to resume interrupted operations without re-processing completed stages.

## Additional Resources

- [LeRobot Documentation](https://docs.huggingface.co/lerobot/)
- [Dataset Tools](src/lerobot/datasets/dataset_tools.py)
- [Dataset v3.0 Format](docs/source/lerobot-dataset-v3.mdx)
- [Paramiko SFTP Documentation](http://docs.paramiko.org/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the [Contributing Guide](CONTRIBUTING.md)
3. Open an issue on the [GitHub repository](https://github.com/huggingface/lerobot)
