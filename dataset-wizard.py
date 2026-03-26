import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import paramiko
from dotenv import load_dotenv
from tqdm import tqdm

from lerobot.datasets.dataset_tools import merge_datasets, modify_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset

# Sub-features to remove after merging
FEATURES_TO_SLICE = {
    "action": ["linear_vel", "angular_vel"],
    "observation.state": ["odom_x", "odom_y", "odom_theta", "linear_vel", "angular_vel"],
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge datasets with optional conversion, compression, and SFTP upload.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline stages (in order):
  1. conversion   - Convert datasets from v2.1 to v3.0 format
  2. merge        - Load and merge individual datasets
  3. slice        - Remove unwanted sub-features from merged dataset
  4. upload       - Compress and upload to SFTP server

Examples:
  # Run all stages (default)
  python dataset-wizard.py

  # Start from merge, stop after slice (skip conversion and upload)
  python dataset-wizard.py --start-from merge --stop-at slice

  # Run only the upload stage
  python dataset-wizard.py --start-from upload

  # Custom dataset path and name
  python dataset-wizard.py --base-path /custom/path --merged-name my-dataset
        """
    )
    parser.add_argument(
        "--start-from",
        type=str,
        choices=["conversion", "merge", "slice", "upload"],
        default="conversion",
        help="Pipeline stage to start from (default: conversion)"
    )
    parser.add_argument(
        "--stop-at",
        type=str,
        choices=["conversion", "merge", "slice", "upload"],
        default="upload",
        help="Pipeline stage to stop at, inclusive (default: upload)"
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default="~/.cache/huggingface/lerobot/Manisha-Saleha",
        help="Base directory where datasets are located (default: ~/.cache/huggingface/lerobot/Manisha-Saleha)"
    )
    parser.add_argument(
        "--merged-name",
        type=str,
        default="test-aloha-dataset-merged",
        help="Name of the merged dataset repository (default: test-aloha-dataset-merged)"
    )
    return parser.parse_args()


# Define the root directory where your datasets are located
args = parse_args()
base_dataset_root = Path(args.base_path).expanduser()

STAGES = ["conversion", "merge", "slice", "upload"]

def should_run(stage: str) -> bool:
    return STAGES.index(args.start_from) <= STAGES.index(stage) <= STAGES.index(args.stop_at)

# List of individual 'move' dataset repo IDs you want to merge, explicitly excluding '_old' versions
move_dataset_repo_ids = [
    "pick-up-blue-cup-mar26-v2.4",
    "pick-up-blue-cup-mar26-v2.3",
    "pick-up-blue-cup-mar26-v2.2",
    "pick-up-blue-cup-mar26-v-2",
    "pick-up-green-cup-mar26-v-2",
    "pick-up-green-cup-mar26-v1.5",
    "pick-up-green-cup-mar26-v1.4",
    "pick-up-green-cup-mar26-v1.2",
    "pick-up-green-cup-mar26-v1.1",
    "pick-up-green-cup-mar26-v3",
    "pick-up-green-cup-mar26-v1"
]

# Convert datasets from v2.1 to v3.0 format if necessary
if should_run("conversion"):
    print("Starting dataset conversion stage...")
    for repo_id in move_dataset_repo_ids:
        dataset_path = base_dataset_root / repo_id
        if not dataset_path.is_dir():
            print(f"Warning: Dataset directory not found for {repo_id} at {dataset_path}. Skipping conversion.")
            continue
        info_path = dataset_path / "meta" / "info.json"
        if info_path.exists():
            import json
            version = json.loads(info_path.read_text()).get("codebase_version", "unknown")
            if version != "v2.1":
                print(f"Skipping '{repo_id}': already at version {version}.")
                continue
        print(f"Converting dataset: {repo_id} at {dataset_path}")
        convert_dataset(
            repo_id=str(base_dataset_root.name + "/" + repo_id), # e.g. Manisha-Saleha/move-blue-cup-feb12-v1.1
            root=str(base_dataset_root.parent), # e.g. ~/.cache/huggingface/lerobot
            push_to_hub=False
        )
else:
    print("Skipping dataset conversion stage.")



# Load each individual 'move' dataset
datasets_to_merge = []

# Define the output repository ID for the merged dataset.
merged_repo_id = args.merged_name

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


# Slice unwanted sub-features from the merged dataset
if should_run("slice"):
    print("\nStarting feature slicing stage...")

    add_features_dict = {}
    remove_features_list = []

    for feature_key, names_to_remove in FEATURES_TO_SLICE.items():
        original_info = merged_dataset.features[feature_key]
        original_names = original_info["names"]
        names_to_remove_set = set(names_to_remove)
        keep_indices = [i for i, n in enumerate(original_names) if n not in names_to_remove_set]
        new_names = [original_names[i] for i in keep_indices]
        new_shape = (len(keep_indices),)

        def make_slicer(key, indices):
            def slicer(row_dict, _ep_idx, _frame_in_ep):
                return np.array(row_dict[key], dtype=np.float32)[indices]
            return slicer

        add_features_dict[feature_key] = (
            make_slicer(feature_key, keep_indices),
            {"dtype": original_info["dtype"], "shape": new_shape, "names": new_names},
        )
        remove_features_list.append(feature_key)
        print(f"  '{feature_key}': {original_info['shape']} -> {new_shape}, removed {names_to_remove}")

    sliced_repo_id = merged_repo_id + "-sliced"
    sliced_output_dir = base_dataset_root / sliced_repo_id

    if sliced_output_dir.exists():
        print(f"Sliced output directory already exists, removing: {sliced_output_dir}")
        shutil.rmtree(sliced_output_dir)

    merged_dataset = modify_features(
        merged_dataset,
        add_features=add_features_dict,
        remove_features=remove_features_list,
        repo_id=sliced_repo_id,
        output_dir=sliced_output_dir,
    )
    output_directory = sliced_output_dir
    print(f"Sliced dataset saved to: {merged_dataset.root}")
else:
    print("Skipping feature slicing stage.")


# Compress zip the merged dataset for easier sharing and uploading
if should_run("upload"):
    print("Starting compression and upload stage...")
    zip_output_path = output_directory.with_suffix(".zip")
    print(f"\nCompressing merged dataset to: {zip_output_path}...")
    shutil.make_archive(str(output_directory), 'zip', str(output_directory))
    print(f"Dataset compressed successfully to: {zip_output_path}\n")

    # Send the merged dataset to the SFTP server
    # Connection parameters
    # Load from .env file

    load_dotenv()

    hostname = str(os.getenv("SFTP_HOSTNAME"))
    port = int(os.getenv("SFTP_PORT", 22)) # Default SFTP port
    username = str(os.getenv("SFTP_USERNAME"))
    password = str(os.getenv("SFTP_PASSWORD")) # Use a secure method to handle passwords in production (e.g., environment variables, key files)
    # Create an SSH client
    ssh_client = paramiko.SSHClient()
    # Automatically add the server's host key (use with caution in production, manual key verification is more secure)
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect to the server
    ssh_client.connect(hostname, port, username, password)

    # Start an SFTP session
    sftp_client = ssh_client.open_sftp()

    print("Connection successfully established.")

    remote_path = os.getenv("SFTP_REMOTE_PATH", None) # Ensure this path ends with a slash
    if remote_path is not None:
        remote_path = str(remote_path)

    if remote_path is None:
        raise ValueError("SFTP_REMOTE_PATH environment variable is not set. Please set it to the desired remote directory path (e.g., /remote/datasets/).")

    # Upload the zip file to the SFTP server
    # Check if remote_path ends with a slash, if not, add it
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
