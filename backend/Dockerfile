# Use an official Python runtime as a parent image
# Base image is selected based on whether GPU support is needed
ARG FORCE_GPU=0

# Set base image
FROM python:3.9-slim as base

# Set up environment for GPU support if enabled
FROM base as gpu-runtime
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/bin:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install NVIDIA runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-3 \
    && rm -rf /var/lib/apt/lists/*

# Final stage - use base as parent
FROM base

# Copy GPU-specific files if GPU is enabled
ARG FORCE_GPU=0
# Only copy CUDA if GPU is enabled
RUN if [ "$FORCE_GPU" = "1" ]; then \
        echo "Setting up GPU support"; \
        COPY --from=gpu-runtime /usr/local/cuda /usr/local/cuda; \
    fi

# Set CUDA environment variables if GPU is enabled
ENV PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Set the working directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

# Install system dependencies and network debugging tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-mysql-client \
    libmariadb-dev \
    libmariadb-dev-compat \
    iputils-ping \
    net-tools \
    dnsutils \
    netcat-openbsd \
    curl \
    vim \
    && if [ "$GPU_ENABLED" = "true" ]; then \
        # Install CUDA toolkit and cuDNN for GPU support
        apt-get install -y --no-install-recommends \
        cuda-toolkit-11-3 \
        libcudnn8=8.2.1.32-1+cuda11.3 \
        libcudnn8-dev=8.2.1.32-1+cuda11.3; \
    fi \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Install base requirements first
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch based on GPU support
RUN if [ "$FORCE_GPU" = "1" ]; then \
        echo "Installing PyTorch with CUDA support"; \
        pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113; \
    else \
        echo "Installing CPU-only PyTorch"; \
        pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install additional Python packages for MariaDB and other requirements
RUN pip install --no-cache-dir \
    pymysql \
    mysql-connector-python==8.0.33 \
    alembic

# Copy the current directory contents into the container at /app
COPY . .

# Make scripts executable
RUN chmod +x /app/startup.sh /app/init-db.sh

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Verify GPU is available if enabled
RUN if [ "$GPU_ENABLED" = "true" ]; then \
        python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"; \
    fi

# Command to run the application
CMD ["./startup.sh"]
