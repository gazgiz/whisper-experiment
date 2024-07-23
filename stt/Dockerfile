FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG ARCH

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# Update and install basic dependencies including git and curl
RUN apt update && apt -y upgrade && \
    apt -y install tzdata software-properties-common git curl build-essential libssl-dev libffi-dev portaudio19-dev

# Install Miniconda
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set conda environment
ENV PATH=/opt/conda/bin:$PATH

# Initialize conda and create the environment
RUN conda init bash && \
    conda create -n stt_tts python=3.9 -y

# Copy requirements.txt and server.py into the image
COPY requirements.txt /root/requirements.txt
COPY server_stt.py /root/server_stt.py
COPY config_stt.json /root/config_stt.json
COPY server_tts.py /root/server_tts.py
COPY config_tts.json /root/config_tts.json

# Ensure the conda environment is activated and install dependencies
RUN /opt/conda/bin/conda run -n stt_tts pip install --upgrade pip && \
    /opt/conda/bin/conda run -n stt_tts pip install -r /root/requirements.txt

# Pre-download Faster Whisper
RUN /opt/conda/bin/conda run -n stt_tts python -c "from faster_whisper import WhisperModel; model = WhisperModel('medium')"

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

# Ensure pip is installed
RUN apt install -y python3-pip

# Upgrade Meson to the latest version
RUN pip3 install --upgrade meson

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

# Install GStreamer Python bindings
RUN apt install -y python3-gi python3-gi-cairo gobject-introspection libgirepository1.0-dev gir1.2-gstreamer-1.0 gir1.2-gtk-3.0


# Install PyGObject via pip
RUN pip install pygobject

# Set environment variable for GStreamer plugin path
ENV LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:/usr/local/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
ENV GST_PLUGIN_PATH=/usr/local/lib/x86_64-linux-gnu/gstreamer-1.0/:/usr/local/lib/gstreamer-1.0:$GST_PLUGIN_PATH
ENV LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:$LIBRARY_PATH
ENV PKG_CONFIG_PATH=/usr/local/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH

# Set working directory (optional)
WORKDIR /root

CMD ["conda", "activate", "stt_tts"]

