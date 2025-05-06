# ComfyUI Setup Guide

## Quick Start

1. Clone and install:
```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install -r requirements.txt
python main.py
```

2. Access UI at: http://localhost:8188

## Model Directories

Place your models in these folders:
- `models/checkpoints/`: Model checkpoints (`.safetensors`, `.ckpt`)
- `models/vae/`: VAE models
- `models/loras/`: LoRA models
- `models/embeddings/`: Embeddings
- `models/controlnet/`: ControlNet models
- `models/upscale_models/`: Upscale models

## Docker Quick Start

```bash
# Using Docker Compose (Recommended)
docker compose up -d

# Access Points
ComfyUI: http://localhost:8188
API: http://localhost:8787
```

## System Requirements
- Python 3.10+
- NVIDIA GPU (recommended) or CPU
- ~10GB disk space

## Common Issues

### NVIDIA CUDA Setup
```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Deploy in GCP
```bash
gcloud auth configure-docker
docker login gcr.io

docker build -t comfy-image-generation .
docker tag comfy-image-generation gcr.io/citric-lead-450721-v2/comfy-image-generation:1.0.6
docker push gcr.io/citric-lead-450721-v2/comfy-image-generation:1.0.6

```