#!/bin/bash
set -euo pipefail

REPO_URL="https://github.com/mikeymcfish/IndexTTS2.git"
REPO_DIR="IndexTTS2"

# Clone the repository if it is not already present
if [ ! -d "${REPO_DIR}/.git" ]; then
    git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

git fetch origin
DEFAULT_BRANCH=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's@^origin/@@')
if [ -z "${DEFAULT_BRANCH}" ]; then
    DEFAULT_BRANCH="main"
fi

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
