FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

SHELL ["/bin/bash","-o","pipefail","-c"]
ENV DEBIAN_FRONTEND=noninteractive

# 1) system deps (poche, cambiano raramente)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget ffmpeg libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2) Miniconda
ARG CONDA_VER=py311_24.5.0-0
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-$CONDA_VER-Linux-x86_64.sh -O /tmp/conda.sh && \
    bash /tmp/conda.sh -b -p /opt/conda && rm /tmp/conda.sh
ENV PATH=/opt/conda/bin:$PATH

# 3) ambiente Conda (layer stabile finché non tocchi environment.yml)
COPY environment.yml /tmp/environment.yml
RUN conda env create -n genesis -f /tmp/environment.yml && conda clean -afy

# 4) TUTTI gli altri file del progetto **tranne src/** (grazie al .dockerignore)
WORKDIR /workspace
COPY . /workspace
#  - Copia doc/, LICENSE, setup.py, genesis/, asset vari, ecc.
#  - Non copia src/ perché è ignorato

# 5) Ora copiamo solo il codice “volatile”
COPY src/ /workspace/src/

# PYTHONPATH serve per importare i moduli senza installarli come package
ENV CONDA_DEFAULT_ENV=genesis \
    PATH=/opt/conda/envs/genesis/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONPATH=/workspace/src

RUN useradd -m appuser && chown -R appuser /workspace
USER appuser

ENTRYPOINT ["conda","run","--no-capture-output","-n","genesis","python","src/winged_drone/evolution.py"]
