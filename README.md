# ComfyUI Setup Guide

## Basic Setup

1. Clone the ComfyUI repository:
```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run ComfyUI:
```bash
python main.py
```

Access the interface at: http://localhost:8188

## Model Management

ComfyUI automatically creates necessary folders. Here's where to place your models:

- Checkpoints (`.safetensors`, `.ckpt`): `models/checkpoints/`
- VAE models: `models/vae/`
- LoRA models: `models/loras/`
- Embeddings: `models/embeddings/`
- Controlnet models: `models/controlnet/`
- Upscale models: `models/upscale_models/`

## Working with Workflows

### Saving Workflows
1. Create your workflow in the ComfyUI interface
2. Click "Save" in the workflow menu (top-right)
3. Workflows are saved as JSON files in the `workflows` directory

### Getting API JSON
1. Create and test your workflow in the UI
2. Click "Save (API Format)" in the workflow menu
3. This generates API-compatible JSON that can be used with the ComfyUI API

### Loading Workflows
1. Click "Load" in the workflow menu
2. Select your saved workflow file
3. Or drag and drop the workflow JSON file into the UI

## Docker Support

### Using Docker Compose (Recommended)

1. Start both ComfyUI and server:
```bash
docker compose up -d
```

2. View logs:
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f comfyui
docker compose logs -f server
```

3. Stop services:
```bash
docker compose down
```

### Using Dockerfile Directly

1. Build ComfyUI image:
```bash
# Build ComfyUI
docker build -t comfyui -f Dockerfile.comfyui .

# Build Server
docker build -t comfyui-server -f Dockerfile.server .
```

2. Run ComfyUI container:
```bash
docker run -d \
  --name comfyui \
  --gpus all \
  -p 8188:8188 \
  -v ./ComfyUI/models:/app/models \
  -v ./ComfyUI/output:/app/output \
  -v ./ComfyUI/custom_nodes:/app/custom_nodes \
  comfyui
```

3. Run Server container:
```bash
docker run -d \
  --name comfyui-server \
  -p 5000:5000 \
  -v ./workflows:/app/workflows \
  --env COMFYUI_SERVER=http://comfyui:8188 \
  --env WS_SERVER=ws://comfyui:8188 \
  --link comfyui \
  comfyui-server
```

### Access Points
- ComfyUI interface: http://localhost:8188
- Server API: http://localhost:5000

### Volume Mounts Explained
- `./ComfyUI/models`: Store all model files
- `./ComfyUI/output`: Generated images and outputs
- `./ComfyUI/custom_nodes`: Custom node extensions
- `./workflows`: Workflow JSON files

### Docker Troubleshooting

1. GPU not detected:
```bash
# Verify NVIDIA drivers
nvidia-smi

# Check NVIDIA Docker support
docker run --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

2. Container access issues:
```bash
# Check container logs
docker logs comfyui
docker logs comfyui-server

# Check container status
docker ps -a
```

3. Volume mount issues:
```bash
# Verify directory permissions
chmod 777 ComfyUI/output ComfyUI/models ComfyUI/custom_nodes workflows
```

4. Cleanup:
```bash
# Remove containers
docker rm -f comfyui comfyui-server

# Remove images
docker rmi comfyui comfyui-server

# Remove unused volumes
docker volume prune
```

## System Requirements

- Python 3.10 or newer
- NVIDIA GPU (recommended) or CPU
- Required disk space: ~10GB (including models)

## Troubleshooting

1. CUDA issues:
```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2. AMD GPU support (Linux):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

3. Model loading errors:
- Verify file permissions
- Check model file integrity
- Ensure correct model placement in directories
