ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:22.07-py3
FROM $BASE_IMAGE

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /GeneFace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install conda packages (Replaced with pip for speed and stability)
RUN pip install fvcore iopath -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install pytorch3d
# Limit parallel jobs to prevent OOM during compilation
ENV MAX_JOBS=1
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Copy requirements
COPY docker/requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install other pip packages
RUN pip install tensorflow==2.12.0 "opencv-python-headless<4.3" protobuf==3.20.3 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Copy the rest of the code
COPY . .

# Run install_ext.sh
RUN bash docker/install_ext.sh
