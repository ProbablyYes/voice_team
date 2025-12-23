# TTS Server Dockerfile (CPU)
FROM python:3.10-slim

# System deps for audio and building wheels
# - Use host proxy for apt (Clash: 7897 mixed port)
# - Force IPv4 to avoid IPv6 unreachable issues
# - Optionally switch Debian mirror to Tsinghua for stability
RUN set -e; \
    printf "deb http://mirrors.tuna.tsinghua.edu.cn/debian trixie main\n" > /etc/apt/sources.list; \
    printf "deb http://mirrors.tuna.tsinghua.edu.cn/debian trixie-updates main\n" >> /etc/apt/sources.list; \
    printf "deb http://mirrors.tuna.tsinghua.edu.cn/debian-security trixie-security main\n" >> /etc/apt/sources.list; \
    apt-get -o Acquire::ForceIPv4=true \
        -o Acquire::http::Proxy=false \
        -o Acquire::https::Proxy=false update \
    && apt-get -o Acquire::ForceIPv4=true \
          -o Acquire::http::Proxy=false \
          -o Acquire::https::Proxy=false install -y --no-install-recommends \
     git \
     libsndfile1 \
     ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy server entry
COPY tts_server.py /app/tts_server.py

# Optionally include local MeloTTS source (place repo under external/MeloTTS)
COPY external/MeloTTS /app/external/MeloTTS

# Optional build args (proxies and pre-downloaded wheel)
ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG all_proxy
ARG ALL_PROXY
ARG MELOTTS_WHL

# Do not set global proxy envs to avoid pip timeouts; we'll rely on direct mirrors
ENV http_proxy="" \
    https_proxy="" \
    HTTP_PROXY="" \
    HTTPS_PROXY="" \
    all_proxy="" \
    ALL_PROXY=""

# Use faster PyPI mirror by default (can be overridden at build)
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Python deps: Flask, PyTorch (CPU)
RUN pip install --no-cache-dir --timeout 180 \
        Flask==3.0.3 \
    && pip install --no-cache-dir --timeout 300 \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        torch \
        && true

## Install MeloTTS with minimal deps to save space
## Prefer local source; otherwise install PyPI package without deps, then add only needed libs
RUN set -e; \
    if [ -n "${MELOTTS_WHL}" ] && [ -f "${MELOTTS_WHL}" ]; then \
        echo "[MeloTTS] Installing wheel ${MELOTTS_WHL} (no-deps)"; \
        pip install --no-cache-dir --no-deps "${MELOTTS_WHL}"; \
    elif [ -f /app/external/MeloTTS/setup.py ] || [ -f /app/external/MeloTTS/pyproject.toml ]; then \
        echo "[MeloTTS] Installing from local /app/external/MeloTTS (no-deps)"; \
        pip install --no-cache-dir --no-deps /app/external/MeloTTS; \
    else \
        echo "[MeloTTS] Installing from GitHub (no-deps)"; \
        pip install --no-cache-dir --no-deps git+https://github.com/myshell-ai/MeloTTS.git \
        || pip install --no-cache-dir --no-deps \
             https://mirror.ghproxy.com/https://codeload.github.com/myshell-ai/MeloTTS/zip/refs/heads/main \
        ; \
    fi \
    && pip install --no-cache-dir --timeout 300 \
        --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        transformers==4.27.4 tokenizers==0.13.3 huggingface-hub==0.36.0 \
        numpy==1.26.4 librosa==0.9.1 soundfile==0.13.1 pydub==0.25.1 \
        cn2an==0.5.22 jieba==0.42.1 pypinyin==0.50.0 cached_path==1.8.1 \
    && pip install --no-cache-dir --timeout 300 \
        --index-url https://download.pytorch.org/whl/cpu \
        torchaudio==2.9.1

# Prepare output path compatible with current server default
RUN mkdir -p /root/voice_team/static/audios

# Runtime env
ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=5003 \
    TTS_DEVICE=cpu \
    TTS_SPEAKER_ID=0 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_TRUSTED_HOST=mirrors.aliyun.com

RUN pip install --no-cache-dir --timeout 300 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    mecab-python3 unidic-lite num2words pykakasi

EXPOSE 5003

CMD ["python", "/app/tts_server.py"]
