import argparse
import os
import shutil
from pathlib import Path

import paramiko
from dotenv import load_dotenv

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge datasets with optional conversion, compression, and SFTP upload.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline stages:
  1. conversion   - Convert datasets from v2.1 to v3.0 format
  2. merge        - Load and merge individual datasets
  3. upload       - Compress and upload to SFTP server

Examples:
  # Run all stages (default)
  python merge_datasets.py

  # Start from merge (skip conversion, run merge and upload)
  python merge_datasets.py --start-from merge

  # Start from upload (skip conversion and merge, run upload only)
  python merge_datasets.py --start-from upload

  # Custom dataset path and name
  python merge_datasets.py --base-path /custom/path --merged-name my-dataset
        """
    )
    parser.add_argument(
        "--start-from",
        type=str,
        choices=["conversion", "merge", "upload"],
        default="conversion",
        help="Pipeline stage to start from (default: conversion)"
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

# List of individual 'move' dataset repo IDs you want to merge, explicitly excluding '_old' versions
move_dataset_repo_ids = [
    "move-blue-cup-feb12-v1.1",
    "move-blue-cup-feb12-v2.1",
    "move-blue-cup-feb12-v2.2",
    "move-green-cup-13feb-v1.1",
    "move-green-cup-13feb-v1.2",
]

# Convert datasets from v2.1 to v3.0 format if necessary
if args.start_from == "conversion":
    print("Starting dataset conversion stage...")
    for repo_id in move_dataset_repo_ids:
        dataset_path = base_dataset_root / repo_id
        if dataset_path.is_dir():
            print(f"Checking dataset format for: {repo_id} at {dataset_path}")
            convert_dataset(
                repo_id=str(base_dataset_root.name + "/" + repo_id), # e.g. Manisha-Saleha/move-blue-cup-feb12-v1.1
                root=str(base_dataset_root.parent), # e.g. ~/.cache/huggingface/lerobot
                push_to_hub=False
                )
        else:
            print(f"Warning: Dataset directory not found for {repo_id} at {dataset_path}. Skipping conversion.")
else:
    print(f"Skipping dataset conversion stage (--start-from {args.start_from})")



# Load each individual 'move' dataset
datasets_to_merge = []

# Define the output repository ID for the merged dataset.
merged_repo_id = args.merged_name

# Define the output directory for the merged dataset.
output_directory = base_dataset_root / merged_repo_id

if args.start_from in ["conversion", "merge"]:
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
    print(f"Skipping dataset merge stage (--start-from {args.start_from})")


# Compress zip the merged dataset for easier sharing and uploading
if args.start_from in ["conversion", "merge", "upload"]:
    print("Starting compression and upload stage...")
    zip_output_path = output_directory.with_suffix(".zip")
    print(f"\nCompressing merged dataset to: {zip_output_path}...")
    shutil.make_archive(str(output_directory), 'zip', str(output_directory))
    print(f"Dataset compressed successfully to: {zip_output_path}\n")

    # Send the merged dataset to the SFTP server
    # Connection parameters
    # Load from .env file

    load_dotenv()

    hostname = os.getenv("SFTP_HOSTNAME")
    port = int(os.getenv("SFTP_PORT", 22)) # Default SFTP port
    username = os.getenv("SFTP_USERNAME")
    password = os.getenv("SFTP_PASSWORD") # Use a secure method to handle passwords in production (e.g., environment variables, key files)
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

    if remote_path is None:
        raise ValueError("SFTP_REMOTE_PATH environment variable is not set. Please set it to the desired remote directory path (e.g., /remote/datasets/).")

    # Upload the zip file to the SFTP server
    remote_file_path = str(remote_path + zip_output_path.name)
    print(f"Uploading {zip_output_path} to {remote_file_path} on the SFTP server...")
    sftp_client.put(str(zip_output_path), remote_file_path)
    print(f"File uploaded successfully to {remote_file_path}.")
else:
    print(f"Skipping compression and upload stage (--start-from {args.start_from})")


