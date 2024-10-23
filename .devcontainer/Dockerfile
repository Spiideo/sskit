ARG TORCH_VERSION=2.4.1-cuda12.1-cudnn9
ARG DEVEL=devel
# ARG DEVEL=runtime

FROM pytorch/pytorch:${TORCH_VERSION}-${DEVEL} as build
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked --mount=type=cache,target=/var/lib/apt,sharing=locked apt-get update && apt-get autoclean && DEBIAN_FRONTEND=noninteractive TZ=CEST \
    apt-get --yes install git

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"
ENV CUDA_HOME=/usr/local/cuda/

COPY requirements*.txt /workspace/
RUN --mount=type=cache,target=/root/.cache/pip pip wheel --wheel-dir /wheels --no-deps -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip wheel --wheel-dir /wheels-devel --no-deps -r requirements-devel.txt

FROM pytorch/pytorch:${TORCH_VERSION}-runtime as base
RUN rm -f /etc/apt/apt.conf.d/docker-clean
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked --mount=type=cache,target=/var/lib/apt,sharing=locked apt-get update && apt-get autoclean && DEBIAN_FRONTEND=noninteractive TZ=CEST \
    apt-get --yes install git libglib2.0-0 curl wget libosmesa6 freeglut3 libegl1 libgbm1 libgles1 libgles2 libice6 libsm6
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels --mount=type=cache,target=/root/.cache/pip pip install /wheels/*

# RUN --mount=type=cache,target=/root/.cache/pip mim install mmengine "mmcv==2.1.0" "mmdet>=3.1.0" "mmpose>=1.1.0"

FROM base as vscode
RUN curl -L 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' | tar -C /usr/local/bin/ -x -v -z -f -

FROM base as devel
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt apt-get update && apt-get autoclean && DEBIAN_FRONTEND=noninteractive TZ=CEST \
    apt-get --yes install libx11-6 libgl1 libglu1 fonts-freefont-ttf sudo fontconfig vim
RUN --mount=type=bind,from=build,source=/wheels-devel,target=/wheels --mount=type=cache,target=/root/.cache/pip pip install /wheels/*
ENV LOCAL_PYTHON=/opt/conda/bin/python

ARG USER=user
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $USER
RUN useradd -m -u $UID -g $GID -s /bin/bash -d /workspace/home $USER
RUN echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER && chmod 0440 /etc/sudoers.d/$USER
USER $USER


