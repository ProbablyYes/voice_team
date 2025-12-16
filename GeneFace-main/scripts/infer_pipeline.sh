#!/bin/bash
set -e

video_id=""
audio_path=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --video_id) video_id="$2"; shift 2 ;;
        --audio_path) audio_path="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$video_id" ] || [ -z "$audio_path" ]; then
    echo "Error: --video_id and --audio_path are required"
    exit 1
fi

echo "Inference Pipeline Start"
echo "Video ID (Model): $video_id"
echo "Audio Path: $audio_path"

# 1. Prepare Audio
# GeneFace expects audio in a specific place or we can pass absolute path.
# Let's copy it to a temp location to be safe and simple.
audio_filename=$(basename "$audio_path")
audio_name="${audio_filename%.*}"
mkdir -p data/raw/val_wavs
target_path="data/raw/val_wavs/$audio_filename"
if [ "$audio_path" != "$target_path" ]; then
    cp "$audio_path" "$target_path"
fi

# 2. Infer Postnet (Audio2Motion)
# We need to find the latest checkpoint or a specific one. 
# For now, let's assume we use the latest one or a fixed one if trained.
# The training script saves checkpoints. Let's find the latest one in checkpoints/${video_id}/lm3d_postnet_sync/
# If not found, default to a reasonable number or error out.

# Simple heuristic: list files, sort, pick last number. 
# Or just use a fixed large number if we trained for fixed epochs.
# Let's try to find the latest ckpt.
ckpt_dir="checkpoints/${video_id}/lm3d_postnet_sync"
if [ ! -d "$ckpt_dir" ]; then
    echo "Error: Checkpoint directory not found: $ckpt_dir"
    exit 1
fi

# Find latest checkpoint step
# Checkpoints are named like 'model_ckpt_steps_10000.ckpt'
latest_ckpt=$(ls "$ckpt_dir" | grep "model_ckpt_steps_" | sort -V | tail -n 1)
if [ -z "$latest_ckpt" ]; then
    echo "Error: No checkpoints found in $ckpt_dir"
    exit 1
fi
# Extract number: model_ckpt_steps_10000.ckpt -> 10000
ckpt_steps=$(echo "$latest_ckpt" | sed 's/model_ckpt_steps_//' | sed 's/.ckpt//')

echo "Using Postnet Checkpoint Steps: $ckpt_steps"

echo "Running Postnet Inference..."
python inference/postnet/postnet_infer.py \
    --config="${ckpt_dir}/config.yaml" \
    --hparams="infer_audio_source_name=data/raw/val_wavs/${audio_filename},infer_out_npy_name=infer_out/${video_id}/pred_lm3d/${audio_name}.npy,infer_ckpt_steps=${ckpt_steps}" \
    --reset

# 3. Infer RAD-NeRF (Rendering)
# Similarly, find latest checkpoint for RAD-NeRF Torso
radnerf_dir="checkpoints/${video_id}/lm3d_radnerf_torso"
# If torso model doesn't exist, fallback to head only? 
# For now assume torso exists as per train_pipeline.sh
if [ ! -d "$radnerf_dir" ]; then
    echo "Error: RAD-NeRF checkpoint directory not found: $radnerf_dir"
    exit 1
fi

echo "Running RAD-NeRF Inference..."
# Note: infer_lm3d_radnerf.sh uses lm3d_radnerf_infer.py
# Output video name
output_video="infer_out/${video_id}/pred_video/${audio_name}.mp4"

python inference/nerfs/lm3d_radnerf_infer.py \
    --config="${radnerf_dir}/config.yaml" \
    --hparams="infer_audio_source_name=data/raw/val_wavs/${audio_filename},infer_cond_name=infer_out/${video_id}/pred_lm3d/${audio_name}.npy,infer_out_video_name=${output_video}" \
    --infer

echo "Inference Completed!"
echo "Output Video: $output_video"
