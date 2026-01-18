FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# Install system dependencies + precompiled OpenCV
RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config \
    python3 python3-pip python3-dev \
    git wget curl unzip \
    libopencv-dev \
    libtbb2 libtbb-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    nano vim \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python packages
RUN pip3 install\
    ultralytics \
    onnx \
    onnxsim 

# Set workspace
WORKDIR /app

CMD ["/bin/bash"]
