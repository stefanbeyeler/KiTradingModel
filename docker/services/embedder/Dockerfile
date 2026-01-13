# Embedder Service Dockerfile
# Port: 3005

FROM python:3.11-slim
ARG SERVICE_PORT=3005

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY docker/services/embedder/requirements.txt .

# Install PyTorch Nightly with CUDA 13.0 support for NVIDIA Thor (sm_110)
RUN pip install --no-cache-dir --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the Embedder app source code (avoid importing other services)
# ARCHITEKTUR: Embedder greift auf Daten NUR über Data Service (Port 3001) zu
# twelvedata_service.py wird NICHT kopiert - alle Datenzugriffe über HTTP
COPY src/services/embedder_app/ ./src/services/embedder_app/
COPY src/services/data_gateway_service.py ./src/services/
COPY src/services/cache_service.py ./src/services/
COPY src/config/ ./src/config/
COPY src/utils/ ./src/utils/
COPY src/shared/ ./src/shared/

# Create empty __init__.py files to avoid import issues
RUN touch ./src/__init__.py ./src/services/__init__.py

# Create directories for data (will be mounted as volumes)
RUN mkdir -p /app/data/models/embedder /app/logs /app/cache

ENV PYTHONPATH=/app
ENV SERVICE_NAME=embedder
ENV PORT=${SERVICE_PORT}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE ${SERVICE_PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

# Dynamischer Port via Umgebungsvariable
CMD ["sh", "-c", "python -m uvicorn src.services.embedder_app.main:app --host 0.0.0.0 --port ${PORT:-3005}"]
