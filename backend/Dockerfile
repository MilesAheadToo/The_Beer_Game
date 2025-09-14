# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    default-mysql-client \
    libmariadb-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    # Ensure PyTorch uses CPU
    CUDA_VISIBLE_DEVICES= \
    NO_CUDA=1 \
    USE_CUDA=0

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
RUN if [ "$GPU_ENABLED" = "true" ]; then \
        echo "Installing PyTorch with CUDA support..."; \
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

# Ensure scripts have proper permissions and line endings
RUN chmod +x /app/startup.sh /app/init-db.sh && \
    find /app -name "*.sh" -exec dos2unix {} + 2>/dev/null || true && \
    find /app -name "*.sh" -exec chmod 755 {} +

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Verify GPU is available if enabled
RUN if [ "$GPU_ENABLED" = "true" ]; then \
        python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"; \
    fi

# Command to run the application with uvicorn
CMD ["sh", "-c", "export PYTHONUNBUFFERED=1 && uvicorn main:app --host 0.0.0.0 --port 8000"]
