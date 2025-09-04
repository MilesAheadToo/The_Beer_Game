# Use an official Python runtime as a parent image
FROM python:3.9-slim

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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional Python packages for MariaDB
RUN pip install --no-cache-dir pymysql mysql-connector-python==8.0.33 alembic

# Copy the current directory contents into the container at /app
COPY . .

# Make scripts executable
RUN chmod +x /app/startup.sh /app/init-db.sh

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Command to run the application
CMD ["./startup.sh"]
