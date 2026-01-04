# HMM Training Service Dockerfile
# Port: 3014
#
# Dedicated training container for HMM and LightGBM Scorer models.
# Runs separately from inference service to prevent blocking API requests.
#
# Model Types:
# - HMM: Gaussian Hidden Markov Model for market regime detection
# - Scorer: LightGBM signal scoring model

FROM python:3.11-slim
ARG SERVICE_PORT=3014

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY docker/services/hmm-train/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the HMM training app source code
COPY src/services/hmm_train_app/ ./src/services/hmm_train_app/
COPY src/services/data_gateway_service.py ./src/services/
COPY src/services/cache_service.py ./src/services/
COPY src/services/twelvedata_service.py ./src/services/
COPY src/config/ ./src/config/
COPY src/utils/ ./src/utils/
COPY src/shared/ ./src/shared/

# Create empty __init__.py files to avoid import issues
RUN touch ./src/__init__.py ./src/services/__init__.py

# Create directories for data (will be mounted as volumes)
RUN mkdir -p /app/data/models/hmm /app/data/symbols /app/logs

ENV PYTHONPATH=/app
ENV SERVICE_NAME=hmm-train
ENV PORT=${SERVICE_PORT}

# Low priority for training - doesn't compete with inference
ENV NICE_PRIORITY=19

EXPOSE ${SERVICE_PORT}

# Health check - must respond quickly even during training
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

# Run with nice priority to yield CPU to inference services
CMD ["sh", "-c", "nice -n ${NICE_PRIORITY:-19} python -m uvicorn src.services.hmm_train_app.main:app --host 0.0.0.0 --port ${PORT:-3014}"]
