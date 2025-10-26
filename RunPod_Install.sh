#!/bin/bash
set -euo pipefail

# Change into the repository root (the directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ ! -d ".git" ]; then
    echo "Error: RunPod_Install.sh must be executed from within a cloned IndexTTS2 repository." >&2
    exit 1
fi

git fetch origin
DEFAULT_BRANCH=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's@^origin/@@')
if [ -z "${DEFAULT_BRANCH}" ]; then
    DEFAULT_BRANCH="main"
fi

# Sync to the latest default branch when possible
git checkout "${DEFAULT_BRANCH}"
git reset --hard "origin/${DEFAULT_BRANCH}"

# Create (or reuse) a virtual environment
python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

# Install FFmpeg if it is not already available
if ! command -v ffmpeg >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y ffmpeg
    else
        apt-get update
        apt-get install -y ffmpeg
    fi
fi

python -m pip install -r requirements.txt

export HF_HUB_ENABLE_HF_TRANSFER=1
python HF_model_downloader.py

deactivate || true

echo "All installed successfully"
