# Use an official Python runtime as a parent image
FROM nvidia/cuda:12.2.0-base-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    # CUDA and GPU settings
    CUDA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/bin:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH} \
    PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH} \
    # Python settings
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.9 \
    python3-pip \
    python3.9-dev \
    default-mysql-client \
    libmariadb-dev \
    libmariadb-dev-compat \
    iputils-ping \
    net-tools \
    dnsutils \
    netcat-openbsd \
    curl \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

# Install base requirements first
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 12.1 support
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cu121 \
    torchvision==0.15.2+cu121 \
    torchaudio==2.0.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install additional Python packages for MariaDB and other requirements
RUN pip3 install --no-cache-dir \
    pymysql \
    mysql-connector-python==8.0.33 \
    alembic

# Install additional Python packages for MariaDB and other requirements
RUN pip install --no-cache-dir \
    pymysql \
    mysql-connector-python==8.0.33 \
    alembic

# Copy the current directory contents into the container at /app
COPY . .

# Ensure scripts have proper permissions and line endings
RUN chmod +x /app/startup.sh /app/init-db.sh && \
    find /app -name "*.sh" -exec dos2unix {} + 2>/dev/null || true && \
    find /app -name "*.sh" -exec chmod 755 {} +

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Verify GPU is available
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); [torch.cuda.is_available()] and [print(f'Device {i}: {torch.cuda.get_device_properties(i)}') for i in range(torch.cuda.device_count())]"

# Command to run the application with uvicorn
CMD ["sh", "-c", "export PYTHONUNBUFFERED=1 && uvicorn main:app --host 0.0.0.0 --port 8000"]
