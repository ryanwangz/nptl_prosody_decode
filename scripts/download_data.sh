#!/usr/bin/bash

#SBATCH -p owners                             # Use the owners partition
#SBATCH --job-name=dl_script_long             # Job name
#SBATCH --begin=now                           # Start the job immediately
#SBATCH --cpus-per-task=3                     # Request 3 CPU cores
#SBATCH --mem=64G                             # Request 64 GB memory
#SBATCH --time=3:00:00                        # Set maximum runtime to 3 hours
#SBATCH --output=%x_%j.out                    # Temporary log path (overridden below)
#SBATCH --error=%x_%j.err                     # Temporary log path (overridden below)

# Base directory to save data
DATA_SAVE_DIR='/home/groups/henderj/rzwang/exp_data/prosody/t12-01-30-2025/'

# Dynamically set output and error paths
LOG_DIR="${DATA_SAVE_DIR}/logs"
if [[ ! -d "$LOG_DIR" ]]; then
    mkdir -p "$LOG_DIR"
    echo "Log directory created: $LOG_DIR"
fi
#SBATCH --output="${LOG_DIR}/%x_%j.out"       # Standard output log in DATA_SAVE_DIR/logs
#SBATCH --error="${LOG_DIR}/%x_%j.err"        # Standard error log in DATA_SAVE_DIR/logs

# Function to handle downloads for a session
download_data() {
    SESSION=$1
    TYPES=("${@:2}")                          # Array of data types to download
    TARGET_DIR="${DATA_SAVE_DIR}/${SESSION}/"

    # Create target directory if it doesn't exist
    if [[ ! -d "$TARGET_DIR" ]]; then
        mkdir -p "$TARGET_DIR"
        echo "Directory created: $TARGET_DIR"
    fi

    # Loop through each data type and download
    for TYPE in "${TYPES[@]}"; do
        echo "Downloading $SESSION $TYPE"
        gsutil -m cp -r "gs://exp_sessions_nearline/${SESSION%%.*}/${SESSION}/${TYPE}" "$TARGET_DIR"
    done

    echo "----------------------------------------"
}

# to run this script, simply run sbatch <scriptname.sh>

## t12.2025.01.30------------------------------
### prosody
### ryan
download_data "t12.2025.01.30" \
    "Data/brand/brainToText/RedisMat/"