FROM ubuntu:24.04

ARG ARCH

# Set environment variables to non-interactive (this will prevent some prompts)
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt update && apt -y upgrade

# Initial setup and installing dependencies
RUN apt -y install tzdata software-properties-common && \
    apt -y install meson dpkg gzip tar curl libmount-dev flex bison bzip2 curl vim git build-essential cmake ninja-build python3-pip pkg-config libssl-dev libx264-dev libopus-dev libvpx-dev nasm x264 python3-pip libxml2-dev && \
    apt install -y zlib1g-dev libglib2.0-dev libffi-dev && \
    apt install -y libpango1.0-dev libpangocairo-1.0-0 libcairo2-dev libsrtp2-dev

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
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/x86_64-linux-gnu:/usr/local/lib/aarch64-linux-gnu/
ENV GST_PLUGIN_PATH=$GST_PLUGIN_PATH:/usr/local/lib/x86_64-linux-gnu/gstreamer-1.0/:/usr/local/lib/gstreamer-1.0/
ENV LIBRARY_PATH="${LIBRARY_PATH}:/usr/local/lib/x86_64-linux-gnu"
ENV PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:/usr/local/lib/x86_64-linux-gnu/pkgconfig"
ENV PYTHONPATH=/usr/local/lib/python3.12/site-packages
ENV GI_TYPELIB_PATH=/usr/local/lib/x86_64-linux-gnu/girepository-1.0/

# Set working directory (optional)
WORKDIR /root

# This will run when the container starts (can be overridden)
CMD ["bash"]

