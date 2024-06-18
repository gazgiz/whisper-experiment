FROM ubuntu:22.04

ARG ARCH

# Set environment variables to non-interactive (this will prevent some prompts)
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
ENV PYLON_ROOT=/opt/pylon

RUN apt update && apt -y upgrade

# Copy the tar.gz file into the Docker image
COPY pylon_*_linux-*_debs.tar.gz /tmp/

# Extract the tar.gz file and install the Debian package
RUN cd /tmp && tar -xvzf pylon_*_linux-*_debs.tar.gz \
    && dpkg -i pylon_*_*.deb


# Initial setup and installing dependencies
RUN apt -y install tzdata software-properties-common && \
    apt -y install dpkg gzip tar curl libmount-dev flex bison bzip2 curl vim git build-essential cmake ninja-build python3-pip pkg-config libssl-dev libx264-dev libopus-dev libvpx-dev nasm x264 python3-pip libxml2-dev && \
    apt install -y zlib1g-dev libglib2.0-dev libffi-dev && \
    apt install -y libpango1.0-dev libpangocairo-1.0-0 libcairo2-dev libsrtp2-dev && \
    pip3 install meson

RUN mkdir -p /root/src && cd /root/src && \
    git clone https://gitlab.freedesktop.org/gstreamer/gstreamer.git && \
    git clone https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs && \
    # Install Rust and necessary components
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    /root/.cargo/bin/rustup default stable && \
    /root/.cargo/bin/cargo install cargo-c && \
    # build and install gstreamer
    # Backup the sources.list file
    cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    # Uncomment all the deb-src lines
    sed -i 's/^# deb-src/deb-src/' /etc/apt/sources.list && \
    # Update the package list
    apt update && \
    apt build-dep -y gstreamer1.0 && \
    cd ~/src/gstreamer && \
    git checkout -b 1.22.5 refs/tags/1.22.5 && \
    cd subprojects && git clone https://github.com/basler/gst-plugin-pylon && cd .. && \
    export PYLON_ROOT=/opt/pylon && \
    meson build -Dgpl=enabled  -Dcustom_subprojects=gst-plugin-pylon && \
    ninja -C build && \
    ninja -C build install && \
    # build and install gst-plugins-rs
    cd ~/src/gst-plugins-rs && \
    /root/.cargo/bin/cargo cbuild -p gst-plugin-webrtc && \
    /root/.cargo/bin/cargo cinstall -p gst-plugin-webrtc && \
    /root/.cargo/bin/cargo cbuild -p gst-plugin-rtp && \
    /root/.cargo/bin/cargo cinstall -p gst-plugin-rtp

COPY pylon_stream.sh /root
COPY v4l2_stream.sh /root

# Set environment variable for GStreamer plugin path
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/x86_64-linux-gnu::/usr/local/lib/aarch64-linux-gnu/:/opt/pylon/lib
ENV GST_PLUGIN_PATH=$GST_PLUGIN_PATH:/usr/local/lib/x86_64-linux-gnu/gstreamer-1.0/:/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0/:/usr/local/lib/gstreamer-1.0/
ENV LIBRARY_PATH="${LIBRARY_PATH}:/usr/local/lib/x86_64-linux-gnu:/usr/local/lib/aarch64-linux-gnu/"
ENV PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:/usr/local/lib/x86_64-linux-gnu/pkgconfig:/usr/local/lib/aarch64-linux-gnu/pkgconfig"
ENV PYTHONPATH=/usr/local/lib/python3.9/site-packages
ENV GI_TYPELIB_PATH=/usr/local/lib/x86_64-linux-gnu/girepository-1.0/:/usr/local/lib/aarch64-linux-gnu/girepository-1.0

# Set working directory (optional)
WORKDIR /root

# This will run when the container starts (can be overridden)
CMD ["bash"]

