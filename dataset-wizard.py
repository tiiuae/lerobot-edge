import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from lerobot.datasets.dataset_tools import merge_datasets, modify_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

# ── EE-specific imports ───────────────────────────────────────────────────────
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats  
from lerobot.datasets.io_utils import write_stats  
from lerobot.model.kinematics import RobotKinematics  
from lerobot.utils.rotation import Rotation 


# ── Constants ─────────────────────────────────────────────────────────────────  
MOBILE_AI_URDF_PATH = Path(__file__).with_name("mobile_ai.urdf")  
  
# Hardcode joint names exactly as they appear in the URDF  
LEFT_JOINTS  = [f"follower_left_joint_{i}"  for i in range(6)]  
RIGHT_JOINTS = [f"follower_right_joint_{i}" for i in range(6)]  
  
# Hardcode the index layout of observation.state / action vectors:  
#   [left_joint_0..5, left_gripper, right_joint_0..5, right_gripper, ...]  
LEFT_JOINT_SLICE   = slice(0, 6)  
LEFT_GRIPPER_IDX   = 6  
RIGHT_JOINT_SLICE  = slice(7, 13)  
RIGHT_GRIPPER_IDX  = 13  

EE_COMPONENTS = ("x", "y", "z", "wx", "wy", "wz")
  
  
# ── Kinematics solvers (created once, reused per row) ─────────────────────────  
def _make_solvers(urdf_path: Path) -> tuple[RobotKinematics, RobotKinematics]:  
    left  = RobotKinematics(str(urdf_path), "follower_left_ee_gripper_link",  LEFT_JOINTS)  
    right = RobotKinematics(str(urdf_path), "follower_right_ee_gripper_link", RIGHT_JOINTS)  
    return left, right  
  
  
def _fk_to_ee(kinematics: RobotKinematics, joint_angles_deg: np.ndarray) -> list[float]:  
    T = kinematics.forward_kinematics(joint_angles_deg)  
    pos    = T[:3, 3]  
    rotvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()  
    return [*pos, *rotvec]   # 6 values: x, y, z, wx, wy, wz  
  
  
# ── Converter callable (matches modify_features signature) ────────────────────  
def _make_converter(feature_key: str, left_kin: RobotKinematics, right_kin: RobotKinematics):  
    def convert(row: dict, episode_index: int, frame_index: int) -> np.ndarray:  
        v = np.asarray(row[feature_key], dtype=np.float32)  
  
        left_ee   = _fk_to_ee(left_kin,  v[LEFT_JOINT_SLICE])  
        right_ee  = _fk_to_ee(right_kin, v[RIGHT_JOINT_SLICE])  
        left_grip  = float(v[LEFT_GRIPPER_IDX])  
        right_grip = float(v[RIGHT_GRIPPER_IDX])  
  
        # Output layout: [left_ee(6), left_gripper, right_ee(6), right_gripper]  
        return np.array([*left_ee, left_grip, *right_ee, right_grip], dtype=np.float32)  
  
    return convert  
  
  
# ── Stats recomputation ───────────────────────────────────────────────────────  
def _recompute_stats(dataset: LeRobotDataset, feature_keys: list[str]) -> None:  
    data_files = sorted((dataset.root / "data").glob("*/*.parquet"))  
    features_to_compute = {k: dataset.meta.features[k] for k in feature_keys}  
    episode_stats = []  
    for data_file in data_files:  
        df = pd.read_parquet(data_file)  
        for _, ep_df in df.groupby("episode_index", sort=True):  
            episode_data = {  
                k: np.stack([np.asarray(v, dtype=np.float32) for v in ep_df[k]])  
                for k in feature_keys  
            }  
            episode_stats.append(compute_episode_stats(episode_data, features_to_compute))  
    stats = dict(dataset.meta.stats or {})  
    stats.update(aggregate_stats(episode_stats))  
    write_stats(stats, dataset.root)  
    dataset.meta.stats = stats  
  
  
# ── Main conversion function ──────────────────────────────────────────────────  
def convert_joint_to_ee_and_save_lerobot_dataset(  
    source_dataset: LeRobotDataset,  
    output_repo_id: str,  
    output_dir: Path | None = None,  
    urdf_path: Path = MOBILE_AI_URDF_PATH,  
    feature_keys: list[str] | None = None,  
    overwrite: bool = False,  
) -> LeRobotDataset:  
    urdf_path = Path(urdf_path).expanduser()  
    if output_dir is not None:  
        output_dir = Path(output_dir).expanduser()  
        if output_dir.exists():  
            if not overwrite:  
                raise FileExistsError(f"Output already exists: {output_dir}")  
            shutil.rmtree(output_dir)  
  
    left_kin, right_kin = _make_solvers(urdf_path)  
  
    # Default: convert observation.state and action  
    keys_to_convert = feature_keys or [  
        k for k in ("observation.state", "action")  
        if k in source_dataset.meta.features  
    ]  
  
    # Build the new feature_info for each key (same dtype/shape, new names)  
    ee_output_size = 14  # left(6) + left_gripper(1) + right(6) + right_gripper(1)  
    add_features = {  
        f"{key}_ee": (  
            _make_converter(key, left_kin, right_kin),  
            {  
                "dtype": "float32",  
                "shape": (ee_output_size,),  
                "names": [  
                    "left.ee.x", "left.ee.y", "left.ee.z",  
                    "left.ee.wx", "left.ee.wy", "left.ee.wz",  
                    "left.ee.gripper_pos",  
                    "right.ee.x", "right.ee.y", "right.ee.z",  
                    "right.ee.wx", "right.ee.wy", "right.ee.wz",  
                    "right.ee.gripper_pos",  
                ],  
            },  
        )  
        for key in keys_to_convert  
    }  
    ee_keys = list(add_features.keys())  
  
    # Two-pass approach: modify_features rejects adding a key that already exists,  
    # so we add with a temp '_ee' suffix first, then remove the originals.  
    with tempfile.TemporaryDirectory() as tmp_dir:  
        intermediate = modify_features(  
            dataset=source_dataset,  
            add_features=add_features,  
            output_dir=Path(tmp_dir) / "intermediate",  
            repo_id=f"{output_repo_id}_intermediate",  
        )  
        converted = modify_features(  
            dataset=intermediate,  
            remove_features=keys_to_convert,  
            output_dir=output_dir,  
            repo_id=output_repo_id,  
        )  
  
    _recompute_stats(converted, ee_keys)  
    return converted


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge datasets with optional conversion, compression, and SFTP upload.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline stages (in order):
  1. conversion   - Convert datasets from v2.1 to v3.0 format
  2. merge        - Load and merge individual datasets
  3. joint_to_ee  - Convert joint-space features to end-effector features
  4. upload       - Compress and upload to SFTP server

Examples:
  # Run all stages using default config.yaml
  python dataset-wizard.py

  # Use a custom config file
  python dataset-wizard.py --config my-config.yaml

  # Override pipeline stage range via CLI
  python dataset-wizard.py --start-from merge --stop-at merge
        """
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--start-from",
        type=str,
        choices=["conversion", "merge", "joint_to_ee", "upload"],
        default=None,
        help="Override start stage from config file"
    )
    parser.add_argument(
        "--stop-at",
        type=str,
        choices=["conversion", "merge", "joint_to_ee", "upload"],
        default=None,
        help="Override stop stage from config file"
    )
    return parser.parse_args()


args = parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

move_dataset_repo_ids = cfg["datasets"]
base_dataset_root = Path(cfg["base_path"]).expanduser()
merged_repo_id = cfg["merged_name"]

STAGES = ["conversion", "merge", "joint_to_ee", "upload"]

start_from = args.start_from or cfg.get("start_from", "conversion")
stop_at = args.stop_at or cfg.get("stop_at", "upload")

for stage_name, stage_value in (("start_from", start_from), ("stop_at", stop_at)):
    if stage_value not in STAGES:
        raise ValueError(f"{stage_name} must be one of: {', '.join(STAGES)}")

if STAGES.index(start_from) > STAGES.index(stop_at):
    raise ValueError(f"start_from ({start_from}) must come before or match stop_at ({stop_at})")

def should_run(stage: str) -> bool:
    return STAGES.index(start_from) <= STAGES.index(stage) <= STAGES.index(stop_at)

# Convert datasets from v2.1 to v3.0 format if necessary
if should_run("conversion"):
    GREEN = "\033[32m"
    RED   = "\033[31m"
    RESET = "\033[0m"

    print("Starting dataset conversion stage...")
    conversion_ok = []          # list of (repo_id, label)
    conversion_failed = []      # list of (repo_id, path, error)

    for repo_id in move_dataset_repo_ids:
        dataset_path = base_dataset_root / repo_id
        old_path = base_dataset_root / f"{repo_id}_old"

        # Already converted: {repo_id} exists at v3.0 (original renamed to {repo_id}_old)
        if dataset_path.is_dir():
            info_path = dataset_path / "meta" / "info.json"
            if info_path.exists():
                import json
                version = json.loads(info_path.read_text()).get("codebase_version", "unknown")
                if version != "v2.1":
                    conversion_ok.append((repo_id, f"already converted ({version})"))
                    continue
        elif old_path.is_dir():
            # {repo_id} is gone but {repo_id}_old exists — treat as already converted
            conversion_ok.append((repo_id, "already converted (_old present)"))
            continue
        else:
            print(f"Warning: Dataset directory not found for {repo_id} at {dataset_path}. Skipping conversion.")
            continue

        print(f"Converting dataset: {repo_id} at {dataset_path}")
        try:
            convert_dataset(
                repo_id=str(base_dataset_root.name + "/" + repo_id), # e.g. Manisha-Saleha/move-blue-cup-feb12-v1.1
                root=str(base_dataset_root.parent), # e.g. ~/.cache/huggingface/lerobot
                push_to_hub=False
            )
            conversion_ok.append((repo_id, "converted"))
        except Exception as e:
            conversion_failed.append((repo_id, dataset_path, e))
            continue

    print("\n── Conversion summary ──────────────────────────────────────────")
    for repo_id, label in conversion_ok:
        print(f"  {GREEN}✔ {repo_id}  ({label}){RESET}")
    for repo_id, dataset_path, e in conversion_failed:
        print(
            f"  {RED}✘ {repo_id}{RESET}\n"
            f"      Path   : {dataset_path}\n"
            f"      Reason : {type(e).__name__}: {e}"
        )
    print("────────────────────────────────────────────────────────────────\n")

    if conversion_failed and (should_run("merge") or should_run("joint_to_ee") or should_run("upload")):
        answer = input(
            f"{len(conversion_failed)} dataset(s) failed conversion. "
            f"Proceed with merge/upload for the {len(conversion_ok)} succeeded dataset(s)? [y/N] "
        ).strip().lower()
        if answer != "y":
            print("Aborting. Fix the failing datasets and re-run.")
            raise SystemExit(0)
else:
    print("Skipping dataset conversion stage.")



# Load each individual 'move' dataset
datasets_to_merge = []

# Define the output repository ID for the merged dataset.
# Define the output directory for the merged dataset.
output_directory = base_dataset_root / merged_repo_id

if should_run("merge"):
    print("Starting dataset loading and merging stage...")
    for repo_id in move_dataset_repo_ids:
        dataset_path = base_dataset_root / repo_id
        if dataset_path.is_dir():
            print(f"Loading dataset: {repo_id} from {dataset_path}")
            dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path)
            print(f"Loaded dataset '{repo_id}' with {dataset.num_episodes} episodes and {dataset.num_frames} frames.")
            datasets_to_merge.append(dataset)
        else:
            print(f"Warning: Dataset directory not found for {repo_id} at {dataset_path}. Skipping.")

    if output_directory.exists():
        print(f"Output directory already exists, removing: {output_directory}")
        shutil.rmtree(output_directory)

    print(f"\nMerging {len(datasets_to_merge)} datasets into {merged_repo_id} at {output_directory}...")

    # Merge the datasets
    # The `merge_datasets` function in `src/lerobot/datasets/dataset_tools.py`
    # uses the `aggregate_datasets` utility from `src/lerobot/datasets/aggregate.py`
    # to consolidate video, data, and metadata, ensuring consistency across the merged datasets,
    # as detailed in the [Dataset Transformation and Manipulation Utilities](#datasets-and-data-processing-dataset-transformation-and-manipulation-utilities) wiki section.
    merged_dataset = merge_datasets(
        datasets=datasets_to_merge,
        output_repo_id=merged_repo_id,
        output_dir=output_directory
    )

    print(f"\nSuccessfully created merged dataset at: {merged_dataset.root}")
    print(f"Total episodes in merged dataset: {merged_dataset.meta.total_episodes}")
    print(f"Total frames in merged dataset: {merged_dataset.meta.total_frames}")
else:
    print("Skipping dataset merge stage.")

if should_run("joint_to_ee"):
    print("Starting joint-to-end-effector conversion stage...")

    joint_to_ee_cfg = cfg.get("joint_to_ee", {})
    source_repo_id = joint_to_ee_cfg.get("source_repo_id", merged_repo_id)
    source_dir = Path(joint_to_ee_cfg.get("source_dir", output_directory)).expanduser()

    output_repo_id = joint_to_ee_cfg.get("output_repo_id", f"{source_repo_id}_ee")
    output_dir_cfg = joint_to_ee_cfg.get("output_dir")
    ee_output_directory = (
        Path(output_dir_cfg).expanduser()
        if output_dir_cfg is not None
        else base_dataset_root / output_repo_id
    )

    feature_keys = joint_to_ee_cfg.get("feature_keys")
    if isinstance(feature_keys, str):
        feature_keys = [feature_keys]

    urdf_path = Path(joint_to_ee_cfg.get("urdf_path", MOBILE_AI_URDF_PATH)).expanduser()
    overwrite = bool(joint_to_ee_cfg.get("overwrite", True))

    source_dataset = None
    if "merged_dataset" in globals():
        merged_root = Path(merged_dataset.root).expanduser().resolve()
        if merged_dataset.repo_id == source_repo_id and merged_root == source_dir.resolve():
            source_dataset = merged_dataset

    if source_dataset is None:
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Joint-to-EE source dataset not found: {source_dir}")
        print(f"Loading source dataset for joint-to-EE conversion: {source_repo_id} from {source_dir}")
        source_dataset = LeRobotDataset(repo_id=source_repo_id, root=source_dir)

    converted_dataset = convert_joint_to_ee_and_save_lerobot_dataset(
        source_dataset=source_dataset,
        output_repo_id=output_repo_id,
        output_dir=ee_output_directory,
        urdf_path=urdf_path,
        feature_keys=feature_keys,
        overwrite=overwrite,
    )

    output_directory = converted_dataset.root
    print(f"\nSuccessfully created end-effector dataset at: {converted_dataset.root}")
    print(f"Total episodes in end-effector dataset: {converted_dataset.meta.total_episodes}")
    print(f"Total frames in end-effector dataset: {converted_dataset.meta.total_frames}")
else:
    print("Skipping joint-to-end-effector conversion stage.")


# Compress zip the current output dataset for easier sharing and uploading
if should_run("upload"):
    print("Starting compression and upload stage...")
    zip_output_path = output_directory.with_suffix(".zip")
    print(f"\nCompressing dataset to: {zip_output_path}...")
    shutil.make_archive(str(output_directory), 'zip', str(output_directory))
    print(f"Dataset compressed successfully to: {zip_output_path}\n")

    # Send the merged dataset to the SFTP server
    sftp_cfg = cfg.get("sftp", {})
    hostname = str(sftp_cfg["hostname"])
    port = int(sftp_cfg.get("port", 22))
    username = str(sftp_cfg["username"])
    password = str(sftp_cfg["password"])
    remote_path = sftp_cfg.get("remote_path")

    if remote_path is None:
        raise ValueError("sftp.remote_path is not set in the config file.")
    remote_path = str(remote_path)

    # Create an SSH client and connect
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname, port, username, password)

    sftp_client = ssh_client.open_sftp()
    print("Connection successfully established.")

    # Upload the zip file to the SFTP server
    if not remote_path.endswith('/'):
        remote_path += '/'
    remote_file_path = str(remote_path + zip_output_path.name)

    # Get file size for progress bar
    file_size = os.path.getsize(str(zip_output_path))

    print(f"Uploading {zip_output_path} to {remote_file_path} on the SFTP server...")
    with tqdm(total=file_size, unit="B", unit_scale=True, desc=f"Uploading {zip_output_path.name}") as pbar:
        last = {"sent": 0}

        def callback(transferred: int, total: int):
            # transferred is cumulative; tqdm wants incremental updates
            pbar.update(transferred - last["sent"])
            last["sent"] = transferred

        sftp_client.put(str(zip_output_path), remote_file_path, callback=callback)

    print(f"File uploaded successfully to {remote_file_path}.")
else:
    print("Skipping compression and upload stage.")
