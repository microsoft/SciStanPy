FROM mcr.microsoft.com/mirror/nvcr/nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# Use bash
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install the necessary packages
# hadolint ignore=DL3008
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        build-essential \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Micromamba package manager.
RUN wget -q -P /tmp \
    "https://github.com/conda-forge/miniforge/releases/download/23.3.1-0/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniforge3-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:$PATH"

# Install mamba packages.
RUN mamba install -y -c conda-forge \
        bokeh::jupyter_bokeh \
        conda-forge::arviz>=0.21 \
        conda-forge::biopython \
        conda-forge::cmdstan \
        conda-forge::cmdstanpy \
        conda-forge::hvplot \
        conda-forge::seaborn \
        conda-forge::typeguard \
        dask \
        jupyter \
        panel \
        pocl \
        pip \
        python=3.12 \
        pyviz::datashader \
        watchfiles \
    && mamba clean --all --force-pkgs-dirs --yes

# Set the environment variables for conda
ENV LD_LIBRARY_PATH="opt/conda/lib:$LD_LIBRARY_PATH"
ENV CMDSTAN="/opt/conda/bin/cmdstan"

# Pip install packages.
# hadolint ignore=DL3013
RUN python3 -m pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio

# Give full permissions to the cmdstan conda directory
# hadolint ignore=DL3059
RUN chmod -R 777 /opt/conda/bin/cmdstan

# Install scistanpy
COPY pyproject.toml setup.py /opt/SciStanPy/
COPY scistanpy/ /opt/SciStanPy/scistanpy/
COPY flipv3/ /opt/SciStanPy/flipv3/
WORKDIR /opt/SciStanPy
RUN python3 -m pip install --no-cache-dir -e .
