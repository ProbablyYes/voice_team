# Eval Metrics Dockerfile (CPU)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180

# System deps for video decode + image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python deps:
# - torch/torchvision: for PIQ + (optional) FID backends
# - piq: NIQE/PSNR/SSIM helpers (NIQE is required)
# - scikit-image/opencv/imageio: frame extraction and SSIM/PSNR fallback
# - torch-fidelity: (optional) FID (may download inception weights on first run)
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch torchvision \
    && pip install --no-cache-dir \
      piq==0.8.0 \
      scikit-image==0.24.0 \
      opencv-python-headless==4.10.0.84 \
      imageio==2.36.1 \
      imageio-ffmpeg==0.5.1 \
      torch-fidelity==0.3.0

# Copy only evaluation code to keep image small
COPY eval_metrics /workspace/eval_metrics

ENTRYPOINT ["python", "-m", "eval_metrics.evaluate"]

