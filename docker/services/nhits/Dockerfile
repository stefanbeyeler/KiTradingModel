# NHITS Service Dockerfile
# Use Python base image - CUDA libraries are mounted from host via nvidia runtime
FROM python:3.11-slim

LABEL maintainer="KI Trading Model"
LABEL service="nhits"

# Build argument for port
ARG SERVICE_PORT=3002

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support (uses host CUDA via nvidia runtime)
# PyTorch 2.5+ supports sm_110 (Thor) when using CUDA 13.x from host
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY .env /app/.env

# Create directories for models and logs
RUN mkdir -p /app/data/models /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV SERVICE_NAME=nhits
ENV PORT=${SERVICE_PORT}
# Enable CUDA to use host libraries
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose port
EXPOSE ${SERVICE_PORT}

# Health check - uses shell form to expand variable
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run service - uses shell form to expand variable
CMD python -m uvicorn src.services.nhits_app.main:app --host 0.0.0.0 --port ${PORT}
