FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential \
    python3 python3-pip supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ComfyUI files
COPY ComfyUI/ /app/ComfyUI/

# Copy application files
COPY app.py .
COPY requirements.txt .
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy workflows
COPY workflows/ /app/workflows/

# Install Python dependencies
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir -r requirements.txt
RUN cd ComfyUI && pip3 install --no-cache-dir -r requirements.txt

# Expose both ports
EXPOSE 8188 8787

# Run supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
