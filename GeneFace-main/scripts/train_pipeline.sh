#!/bin/bash
set -e

video_path=""
epochs=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --video_path) video_path="$2"; shift 2 ;;
        --epochs) epochs="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$video_path" ]; then
    echo "Error: --video_path is required"
    exit 1
fi

# 1. Prepare Data
video_filename=$(basename "$video_path")
video_id="${video_filename%.*}"

echo "Processing video: $video_id"

mkdir -p data/raw/videos
target_path="data/raw/videos/$video_filename"
if [ "$video_path" != "$target_path" ]; then
    cp "$video_path" "$target_path"
fi

# 2. Prepare Configs
config_dir="egs/datasets/videos/$video_id"
if [ -d "$config_dir" ]; then
    echo "Config dir exists, skipping creation"
else
    echo "Creating config dir: $config_dir"
    mkdir -p "$config_dir"
    # Copy from May
    cp egs/datasets/videos/May/*.yaml "$config_dir/"
    
    # Replace 'May' with video_id in all yaml files
    for file in "$config_dir"/*.yaml; do
        sed -i "s/May/$video_id/g" "$file"
    done
fi

# 修改点：加入断点机制。检查预处理的终产物是否存在，存在则跳过。
processed_dataset="data/binary/videos/$video_id/trainval_dataset.npy"

if [ -f "$processed_dataset" ]; then
    echo "[INFO] 预处理产物 $processed_dataset 已存在，跳过数据预处理步骤。"
else
    echo "[INFO] 未发现预处理数据，开始运行 process_data.sh..."
    # Run data processing
    bash data_gen/nerf/process_data.sh "$video_id"
    
    # 处理完后检查是否生成成功
    if [ ! -f "$processed_dataset" ]; then
        echo "Error: 预处理执行完毕，但未发现产物 $processed_dataset，请检查日志。"
        exit 1
    fi
fi

# 3. Train Postnet
echo "Training Postnet..."
python tasks/run.py --config="$config_dir/lm3d_postnet_sync.yaml" --exp_name="$video_id/postnet" --hparams="max_updates=$epochs"

# 4. Train RAD-NeRF Head
echo "Training RAD-NeRF Head..."
python tasks/run.py --config="$config_dir/lm3d_radnerf.yaml" --exp_name="$video_id/lm3d_radnerf" --hparams="max_updates=$epochs"

# 5. Train RAD-NeRF Torso
echo "Training RAD-NeRF Torso..."
python tasks/run.py --config="$config_dir/lm3d_radnerf_torso.yaml" --exp_name="$video_id/lm3d_radnerf_torso" --hparams="max_updates=$epochs"

echo "Training pipeline completed!"