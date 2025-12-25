#!/bin/bash
# run_geneface.sh

set -e

IMAGE_NAME="geneface:latest"
WORKSPACE="/GeneFace"
GENEFACE_DIR="./GeneFace-main" # Host directory
MODEL_CACHE_DIR="./model_cache" # Host directory (shared torch hub checkpoints)

# Mock docker for testing if needed, similar to SyncTalk
mock_docker() {
    # For now, just run docker
    docker "$@"
}

# Helper to parse GPU
parse_gpu_arg() {
    local gpu_arg="$1"
    local upper_arg=$(echo "$gpu_arg" | tr '[:lower:]' '[:upper:]')
    if [[ "$upper_arg" == "CPU" ]]; then
        echo ""
        return 0
    fi
    if [[ "$upper_arg" =~ ^GPU[0-9]+$ ]]; then
        local gpu_num=$(echo "$upper_arg" | sed 's/GPU//')
        echo "--gpus device=$gpu_num"
    else
        echo "--gpus device=0"
    fi
}

# Main logic
mode="$1"
shift

if [ "$mode" == "train" ]; then
    video_path=""
    gpu_arg="GPU0"
    epochs="10000" # Default

    while [[ $# -gt 0 ]]; do
        case $1 in
            --video_path)
                video_path="$2"
                shift 2
                ;;
            --gpu)
                gpu_arg="$2"
                shift 2
                ;;
            --epochs)
                epochs="$2"
                shift 2
                ;;
            *)
                echo "Unknown argument: $1"
                exit 1
                ;;
        esac
    done

    if [ -z "$video_path" ]; then
        echo "Error: --video_path is required"
        exit 1
    fi

    gpu_param=$(parse_gpu_arg "$gpu_arg")
    geneface_abs=$(realpath "$GENEFACE_DIR")
    model_cache_abs=$(realpath "$MODEL_CACHE_DIR" 2>/dev/null || true)
    
    # Ensure directories exist
    mkdir -p "$GENEFACE_DIR/data"
    mkdir -p "$GENEFACE_DIR/checkpoints"
    mkdir -p "$GENEFACE_DIR/egs"

    echo "Starting GeneFace training..."
    echo "Video: $video_path"
    echo "GPU: $gpu_param"

    # Run Docker
    # We mount the whole GeneFace directory to /GeneFace
    # And run the internal pipeline script
    
    mock_docker run --rm $gpu_param \
        -v "$geneface_abs:$WORKSPACE" \
        ${model_cache_abs:+-v "$model_cache_abs:/root/.cache/torch/hub/checkpoints"} \
        -w "$WORKSPACE" \
        $IMAGE_NAME \
        bash scripts/train_pipeline.sh --video_path "$video_path" --epochs "$epochs"

elif [ "$mode" == "infer" ]; then
    model_dir="" # This is actually the video_id / model name
    audio_path=""
    gpu_arg="GPU0"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --model_dir)
                model_dir="$2"
                shift 2
                ;;
            --audio_path)
                audio_path="$2"
                shift 2
                ;;
            --gpu)
                gpu_arg="$2"
                shift 2
                ;;
            *)
                echo "Unknown argument: $1"
                exit 1
                ;;
        esac
    done

    if [ -z "$model_dir" ] || [ -z "$audio_path" ]; then
        echo "Error: --model_dir and --audio_path are required"
        exit 1
    fi

    # Handle Audio Path
    # We need to make the audio file accessible inside the container.
    # We'll mount the directory containing the audio file.
    audio_abs_path=$(realpath "$audio_path")
    audio_dir=$(dirname "$audio_abs_path")
    audio_filename=$(basename "$audio_abs_path")
    
    # Container path for audio
    container_audio_dir="/input_audio"
    container_audio_path="$container_audio_dir/$audio_filename"

    gpu_param=$(parse_gpu_arg "$gpu_arg")
    geneface_abs=$(realpath "$GENEFACE_DIR")
    model_cache_abs=$(realpath "$MODEL_CACHE_DIR" 2>/dev/null || true)

    echo "Starting GeneFace inference..."
    echo "Model: $model_dir"
    echo "Audio: $audio_path"
    echo "GPU: $gpu_param"

    mock_docker run --rm $gpu_param \
        -v "$geneface_abs:$WORKSPACE" \
        ${model_cache_abs:+-v "$model_cache_abs:/root/.cache/torch/hub/checkpoints"} \
        -v "$audio_dir:$container_audio_dir" \
        -w "$WORKSPACE" \
        $IMAGE_NAME \
        bash scripts/infer_pipeline.sh --video_id "$model_dir" --audio_path "$container_audio_path"

elif [ "$mode" == "build" ]; then
    echo "Building Docker image $IMAGE_NAME..."
    docker build -t $IMAGE_NAME -f Dockerfile .
else
    echo "Usage: $0 {train|infer|build} ..."
    exit 1
fi
