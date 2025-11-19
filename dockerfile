FROM debian:trixie-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        libeigen3-dev \
        libomp-dev \
        ca-certificates \
        wget && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CC=gcc \
    CXX=g++

RUN python3 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir \
        numpy "laspy[laszip]" \
        pybind11 \
        scikit-build-core \
        ninja plyfile lazrs viser

ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace

ENV VISER_HOST=0.0.0.0 \
    VISER_PORT=8080

EXPOSE 8080

CMD ["/bin/bash"]
