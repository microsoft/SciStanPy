FROM mcr.microsoft.com/mirror/nvcr/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Use bash
SHELL ["/bin/bash", "-c"]

# Install the necessary packages
# hadolint ignore=DL3008
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        build-essential \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Micromamba package manager.
RUN "${SHELL}" <(curl -L micro.mamba.pm/install.sh) && mv ~/.local/bin/micromamba /usr/bin

# Install micromamba packages.
RUN micromamba config set channel_priority flexible \
    && micromamba install -y -c conda-forge \
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
    && micromamba clean --all --force-pkgs-dirs --yes

# Image breaks with the default micromamba install. Move to /usr/local
RUN mv /root/micromamba /usr/local/

# Set the environment variables for micromamba
ENV PATH="/usr/local/micromamba/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/micromamba/lib:$LD_LIBRARY_PATH"

# Pip install packages.
# hadolint ignore=DL3013
RUN python3 -m pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Install dms_stan
COPY pyproject.toml setup.py /opt/DMSStan/
COPY dms_stan/ /opt/DMSStan/dms_stan/
WORKDIR /opt/DMSStan
RUN python3 -m pip install --no-cache-dir -e .
