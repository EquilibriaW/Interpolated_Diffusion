FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git curl ca-certificates ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# Torch 2.10 is expected from nightly at the time of writing.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu128
RUN python3 -m pip install --pre torch torchvision torchaudio --index-url ${TORCH_INDEX_URL}

WORKDIR /workspace/Interpolated_Diffusion
COPY . /workspace/Interpolated_Diffusion

RUN python3 -m pip install -r requirements.txt && \
    python3 -m pip install \
      diffusers transformers accelerate safetensors datasets webdataset tensorboard scikit-learn && \
    python3 -m pip install git+https://github.com/KellerJordan/Muon.git

ENV PYTHONPATH=/workspace/Interpolated_Diffusion
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV NCCL_DEBUG=warn
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1

RUN chmod +x docker/runpod/entrypoint.sh
ENTRYPOINT ["/workspace/Interpolated_Diffusion/docker/runpod/entrypoint.sh"]

