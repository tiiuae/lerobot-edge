import argparse
import os
import shutil
from pathlib import Path

import paramiko
import yaml
from tqdm import tqdm

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset



def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge datasets with optional conversion, compression, and SFTP upload.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline stages (in order):
  1. conversion   - Convert datasets from v2.1 to v3.0 format
  2. merge        - Load and merge individual datasets
  3. upload       - Compress and upload to SFTP server

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
        choices=["conversion", "merge", "upload"],
        default=None,
        help="Override start stage from config file"
    )
    parser.add_argument(
        "--stop-at",
        type=str,
        choices=["conversion", "merge", "upload"],
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

STAGES = ["conversion", "merge", "upload"]

start_from = args.start_from or cfg.get("start_from", "conversion")
stop_at = args.stop_at or cfg.get("stop_at", "upload")

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

    if conversion_failed and should_run("merge") or should_run("upload"):
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



# Compress zip the merged dataset for easier sharing and uploading
if should_run("upload"):
    print("Starting compression and upload stage...")
    zip_output_path = output_directory.with_suffix(".zip")
    print(f"\nCompressing merged dataset to: {zip_output_path}...")
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
