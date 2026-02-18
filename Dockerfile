FROM mcr.microsoft.com/mirror/nvcr/nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# Use bash
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install the necessary packages including Python 3.12
# hadolint ignore=DL3008
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        binutils \
        build-essential \
        cpp-11 \
        dpkg \
        libc-bin \
        libcap2 \
        libgnutls30 \
        libgssapi-krb5-2 \
        libpam-modules \
        libsqlite3-0 \
        libssl3 \
        libsystemd0 \
        libtasn1-6 \
        linux-libc-dev \
        pocl-opencl-icd \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        wget \
    && apt-get upgrade -y \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default and ensure pip is available
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && python3 -m ensurepip --upgrade

# Install Python packages via pip
# hadolint ignore=DL3013
RUN python3 -m pip install --no-cache-dir \
    "arviz>=0.21" \
    biopython \
    "cmdstanpy[all]==1.2.5" \
    "dask[complete]" \
    datashader \
    hvplot \
    "idna>=3.7" \
    jupyter \
    jupyter-bokeh \
    "jupyter-core>=5.8.1" \
    panel \
    "pillow>=11.3.0" \
    "requests>=2.32.0" \
    seaborn \
    torch \
    torchaudio \
    torchvision \
    "tqdm>=4.66.3" \
    typeguard \
    "urllib3>2.0.6" \
    watchfiles

# Install cmdstan via cmdstanpy
RUN install_cmdstan --dir /opt/ --version 2.38.0
ENV CMDSTAN=/opt/cmdstan-2.38.0
RUN chmod -R 777 $CMDSTAN

# Install scistanpy
COPY pyproject.toml setup.py /opt/SciStanPy/
COPY scistanpy/ /opt/SciStanPy/scistanpy/
COPY flipv3/ /opt/SciStanPy/flipv3/
WORKDIR /opt/SciStanPy
RUN python3 -m pip install --no-cache-dir -e .
