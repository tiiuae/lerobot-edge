# Dataset Wizard Guide

The `dataset-wizard.py` script provides a complete pipeline for managing and merging robotic datasets. It handles dataset format conversion, merging multiple datasets, compression, and uploading to remote SFTP servers.

## Overview

The Dataset Wizard performs three main stages:

1. **Conversion** - Convert datasets from v2.1 to v3.0 format
2. **Merge** - Load and merge individual datasets into a single unified dataset
3. **Upload** - Compress and upload the merged dataset to an SFTP server

You can start the pipeline from any stage using the `--start-from` option, allowing you to skip previously completed steps.

## Prerequisites

Follow the original README.md file to install lerobot.

## Configuration: Setting up the .env File

The script uses environment variables from a `.env` file for SFTP configuration. Create a `.env` file in the root directory of the project with the following variables:

### .env File Example

```env
# SFTP Server Configuration
SFTP_HOSTNAME=sftp.example.com
SFTP_PORT=22
SFTP_USERNAME=your_sftp_username
SFTP_PASSWORD=your_sftp_password
SFTP_REMOTE_PATH=/datasets/
```

### .env Configuration Details

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SFTP_HOSTNAME` | Yes | — | The hostname or IP address of your SFTP server |
| `SFTP_PORT` | No | 22 | The SFTP server port (standard SFTP port is 22) |
| `SFTP_USERNAME` | Yes | — | Your SFTP username |
| `SFTP_PASSWORD` | Yes | — | Your SFTP password |
| `SFTP_REMOTE_PATH` | Yes* | — | Remote directory path on SFTP server (e.g., `/datasets/` or `/remote/datasets/`). Required only if running the upload stage. Must be an existing directory on the server. |

**Important:** The `SFTP_REMOTE_PATH` must end with a trailing slash (e.g., `/datasets/` not `/datasets`). The script will automatically add it if missing.

### Security Best Practices

⚠️ **Warning:** Never commit your `.env` file to version control. Add it to your `.gitignore`:

```
.env
```

For production environments, consider using:
- SSH key authentication instead of password authentication
- Dedicated SFTP service accounts with restricted permissions
- Encrypted password storage or credential management systems

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

# Start from upload stage (skip conversion and merge)
python dataset-wizard.py --start-from upload
```

**Options:**
- `conversion` - Start from dataset format conversion (default)
- `merge` - Skip conversion, start from dataset merging
- `upload` - Skip conversion and merge, start from compression and upload

#### `--base-path` - Specify dataset location

Customize where the script looks for input datasets and outputs the merged dataset:

```bash
python dataset-wizard.py --base-path /custom/path/to/datasets
```

**Default:** `~/.cache/huggingface/lerobot/Manisha-Saleha`

This path should contain subdirectories for each individual dataset to merge.

#### `--merged-name` - Specify merged dataset name

Set a custom name for the output merged dataset:

```bash
python dataset-wizard.py --merged-name my-custom-dataset
```

**Default:** `test-aloha-dataset-merged`

### Complete Example

```bash
python dataset-wizard.py \
  --start-from merge \
  --base-path ~/.cache/huggingface/lerobot/my-user \
  --merged-name my-aloha-merged-dataset
```

This will:
- Skip the conversion stage
- Merge datasets from `~/.cache/huggingface/lerobot/my-user/`
- Output to `~/.cache/huggingface/lerobot/my-user/my-aloha-merged-dataset/`
- Compress and upload the result to the SFTP server

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
- Merged dataset directory at `{base-path}/{merged-name}/`
- Statistics showing total episodes and frames

### Stage 3: Upload

Compresses the merged dataset and uploads it to the SFTP server.

**What happens:**
- Creates a ZIP file of the merged dataset
- Connects to the SFTP server using credentials from `.env`
- Uploads the ZIP file with progress tracking
- Verifies successful upload

**Triggered by:**
```bash
python dataset-wizard.py --start-from upload
```

**Requirements:**
- Valid `.env` file with SFTP credentials
- `SFTP_REMOTE_PATH` directory must exist on the server
- Sufficient disk space on SFTP server
- Network connectivity to SFTP server

## Output Files

### During Processing
- **Converted datasets:** Stored in-place at `{base-path}/{dataset-id}/`
- **Merged dataset:** Created at `{base-path}/{merged-name}/`
- **Compressed file:** Created as `{base-path}/{merged-name}.zip`

### After Upload
- **Remote file:** Uploaded to `{SFTP_REMOTE_PATH}{merged-name}.zip`

## Troubleshooting

### `.env` file not found
```
Error: dotenv could not find .env file
```
**Solution:** Create a `.env` file in the project root directory with the required variables.

### SFTP connection refused
```
Error: Connection refused on SFTP_HOSTNAME:SFTP_PORT
```
**Checklist:**
- Verify `SFTP_HOSTNAME` and `SFTP_PORT` are correct
- Check network connectivity to the SFTP server
- Ensure the SFTP server is running
- Check firewall rules allowing outbound connections

### Authentication failed
```
Error: Authentication failed with username/password
```
**Solution:**
- Verify `SFTP_USERNAME` and `SFTP_PASSWORD` in the `.env` file
- Check for special characters that may need escaping
- Ensure the SFTP account has permission to connect

### Remote path error
```
Error: SFTP_REMOTE_PATH environment variable is not set
```
**Solution:**
- Add `SFTP_REMOTE_PATH=/path/to/remote/dir/` to the `.env` file
- Ensure the path ends with a trailing slash
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

Use this when you want to prepare a merged dataset without uploading immediately:

```bash
python dataset-wizard.py --start-from conversion --base-path ~/.cache/huggingface/lerobot/my-user --merged-name dataset-v1
```

Then later, upload only the final result:

```bash
python dataset-wizard.py --start-from upload --base-path ~/.cache/huggingface/lerobot/my-user --merged-name dataset-v1
```

### Workflow 2: Re-upload Without Re-merging

If the initial upload failed but the merge was successful:

```bash
python dataset-wizard.py --start-from upload --base-path ~/.cache/huggingface/lerobot/my-user --merged-name dataset-v1
```

The previously merged dataset will be recompressed and uploaded.

### Workflow 3: Full Pipeline with Custom Paths

Create a comprehensive merge and upload of a new dataset collection:

```bash
python dataset-wizard.py \
  --base-path ~/my-datasets/robot-collection \
  --merged-name robot-collection-merged-2024
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

After running, the structure will include:

```
base-path/
├── [original datasets above]
└── merged-name/
    ├── videos/
    ├── metadata.json
    └── [merged dataset files]
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
