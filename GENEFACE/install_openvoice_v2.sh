#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$HOME/voice_team/OpenVoice"

echo "[1/7] System deps"
apt-get update -y
apt-get install -y git wget unzip ffmpeg build-essential rsync

echo "[2/7] Activate conda env: voice_env"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found. Please ensure conda is installed and init-ed."
  exit 1
fi
conda activate voice_env

echo "[3/7] Clone OpenVoice to $WORKDIR"
if [ ! -d "$WORKDIR/.git" ]; then
  git clone https://github.com/myshell-ai/OpenVoice.git "$WORKDIR"
else
  echo "Repo already exists, skip clone."
fi

cd "$WORKDIR"

echo "[4/7] Install OpenVoice (editable) + upgrade pip"
python -m pip install -U pip setuptools wheel
python -m pip install -e .

echo "[5/7] Install MeloTTS + UniDic (needed by V2)"
python -m pip install "git+https://github.com/myshell-ai/MeloTTS.git"
python -m unidic download

echo "[6/7] Download V2 checkpoints -> checkpoints_v2/"
mkdir -p checkpoints_v2
TMPZIP="/tmp/checkpoints_v2.zip"

# 两个常见 S3 域名写法，任何一个能下就行
(wget -O "$TMPZIP" "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip") || \
(wget -O "$TMPZIP" "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip")

unzip -o "$TMPZIP" -d checkpoints_v2
rm -f "$TMPZIP"

# 如果解压后变成 checkpoints_v2/checkpoints_v2/... 自动拉平
if [ -f "checkpoints_v2/checkpoints_v2/converter/config.json" ] && [ ! -f "checkpoints_v2/converter/config.json" ]; then
  rsync -a checkpoints_v2/checkpoints_v2/ checkpoints_v2/
fi

echo "[7/7] Verify key files"
test -f checkpoints_v2/converter/config.json
test -f checkpoints_v2/converter/checkpoint.pth
echo "OK: OpenVoice V2 ready in $WORKDIR"
