FROM mcr.microsoft.com/mirror/nvcr/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Use bash
SHELL ["/bin/bash", "-c"]

# Install the necessary packages
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        build-essential \
        curl

# Install Micromamba package manager.
RUN "${SHELL}" <(curl -L micro.mamba.pm/install.sh) && mv ~/.local/bin/micromamba /usr/bin

# Install micromamba packages.
COPY pipless-conda.yml /opt/DMSStan/
WORKDIR /opt/DMSStan
RUN micromamba config set channel_priority flexible && \
    micromamba env update -f pipless-conda.yml -y --name base --prune \
    && micromamba clean --all --force-pkgs-dirs --yes

# Install dms_stan
COPY pyproject.toml setup.py dms_stan /opt/DMSStan/
RUN pip install --no-cache-dir -e .
