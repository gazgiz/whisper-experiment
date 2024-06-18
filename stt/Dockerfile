FROM ubuntu:24.04

ARG ARCH

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# Update and install basic dependencies including git and curl
RUN apt update && apt -y upgrade && \
    apt -y install tzdata software-properties-common git curl build-essential libssl-dev libffi-dev

# Install Miniconda
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set conda environment
ENV PATH=/opt/conda/bin:$PATH

# Create conda environment with Python 3.9
RUN conda create -n webrtc_env python=3.9 -y
RUN echo "conda activate webrtc_env" >> ~/.bashrc
SHELL ["conda", "run", "-n", "webrtc_env", "/bin/bash", "-c"]

# Copy requirements.txt and server.py into the image early to fail fast
COPY requirements.txt /root/requirements.txt
COPY server.py /root/server.py

# Install Whisper separately
RUN pip install --upgrade pip
RUN pip install git+https://github.com/openai/whisper.git

# Install remaining Python dependencies inside the conda environment
RUN apt -y install cmake
RUN pip install -r /root/requirements.txt

# Install other system dependencies
RUN apt -y install meson dpkg gzip tar libmount-dev flex bison bzip2 vim git build-essential cmake ninja-build pkg-config libssl-dev libx264-dev libopus-dev libvpx-dev nasm x264 libxml2-dev && \
    apt -y install zlib1g-dev libglib2.0-dev libffi-dev && \
    apt -y install libpango1.0-dev libpangocairo-1.0-0 libcairo2-dev libsrtp2-dev

# Add deb-src lines to sources.list if not present
RUN echo "deb-src http://archive.ubuntu.com/ubuntu/ focal main restricted" >> /etc/apt/sources.list && \
    echo "deb-src http://archive.ubuntu.com/ubuntu/ focal-updates main restricted" >> /etc/apt/sources.list && \
    echo "deb-src http://archive.ubuntu.com/ubuntu/ focal universe" >> /etc/apt/sources.list && \
    echo "deb-src http://archive.ubuntu.com/ubuntu/ focal-updates universe" >> /etc/apt/sources.list && \
    echo "deb-src http://archive.ubuntu.com/ubuntu/ focal multiverse" >> /etc/apt/sources.list && \
    echo "deb-src http://archive.ubuntu.com/ubuntu/ focal-updates multiverse" >> /etc/apt/sources.list && \
    echo "deb-src http://archive.ubuntu.com/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list

# Update package list and install build dependencies for gstreamer
RUN apt update && apt build-dep -y gstreamer1.0

RUN mkdir -p /root/src && cd /root/src && \
    git clone https://gitlab.freedesktop.org/gstreamer/gstreamer.git && \
    git clone https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs

# Install Rust and necessary components
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    /root/.cargo/bin/rustup default stable && \
    /root/.cargo/bin/cargo install cargo-c

# Build and install gstreamer
RUN cd /root/src/gstreamer && \
    git checkout -b 1.24.4 refs/tags/1.24.4 && \
    meson build -Dgpl=enabled && \
    ninja -C build && \
    ninja -C build install

# Build and install gst-plugins-rs
RUN cd /root/src/gst-plugins-rs && \
    /root/.cargo/bin/cargo cbuild -p gst-plugin-webrtc && \
    /root/.cargo/bin/cargo cinstall -p gst-plugin-webrtc && \
    /root/.cargo/bin/cargo cbuild -p gst-plugin-rtp && \
    /root/.cargo/bin/cargo cinstall -p gst-plugin-rtp

# Set environment variable for GStreamer plugin path
ENV LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:/usr/local/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
ENV GST_PLUGIN_PATH=/usr/local/lib/x86_64-linux-gnu/gstreamer-1.0/:/usr/local/lib/gstreamer-1.0:$GST_PLUGIN_PATH
ENV LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:$LIBRARY_PATH
ENV PKG_CONFIG_PATH=/usr/local/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH

# Set working directory (optional)
WORKDIR /root

# Expose the server port
EXPOSE 8000

# Command to run the server
CMD ["conda", "run", "--no-capture-output", "-n", "webrtc_env", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

