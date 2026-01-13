# TCN Training Service Dockerfile
# Port: 3013
#
# Dedicated training container - runs separately from inference service.
# This prevents training from blocking pattern detection API requests.
#
# Uses PyTorch Nightly with CUDA 13.0 for NVIDIA Thor GPU (sm_110) support.

FROM python:3.11-slim
ARG SERVICE_PORT=3013

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY docker/services/tcn-train/requirements.txt .

# Install PyTorch Nightly with CUDA 13.0 support for NVIDIA Thor (sm_110)
RUN pip install --no-cache-dir --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the TCN training app source code
# ARCHITEKTUR: TCN-Train greift auf Daten NUR über Data Service (Port 3001) zu
# twelvedata_service.py wird NICHT kopiert - alle Datenzugriffe über HTTP
COPY src/services/tcn_train_app/ ./src/services/tcn_train_app/
COPY src/services/data_gateway_service.py ./src/services/
COPY src/services/cache_service.py ./src/services/
COPY src/config/ ./src/config/
COPY src/utils/ ./src/utils/
COPY src/shared/ ./src/shared/

# Create empty __init__.py files to avoid import issues
RUN touch ./src/__init__.py ./src/services/__init__.py

# Create directories for data (will be mounted as volumes)
RUN mkdir -p /app/data/models/tcn /app/data/symbols /app/logs

ENV PYTHONPATH=/app
ENV SERVICE_NAME=tcn-train
ENV PORT=${SERVICE_PORT}
# Enable CUDA to use host libraries
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Low priority for training - doesn't compete with inference
ENV NICE_PRIORITY=19

EXPOSE ${SERVICE_PORT}

# Health check - must respond quickly even during training
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

# Run with nice priority to yield CPU to inference services
CMD ["sh", "-c", "nice -n ${NICE_PRIORITY:-19} python -m uvicorn src.services.tcn_train_app.main:app --host 0.0.0.0 --port ${PORT:-3013}"]
