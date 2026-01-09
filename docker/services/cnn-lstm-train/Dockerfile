# CNN-LSTM Multi-Task Training Service
# Port: 3017

FROM python:3.11-slim

ARG SERVICE_PORT=3017

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY docker/services/cnn-lstm-train/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code - Training App
COPY src/services/cnn_lstm_train_app/ ./src/services/cnn_lstm_train_app/

# Copy source code - Inference App (for model architecture)
COPY src/services/cnn_lstm_app/ ./src/services/cnn_lstm_app/

# Copy shared config
COPY src/config/ ./src/config/
COPY src/utils/ ./src/utils/

# Create __init__.py files
RUN touch ./src/__init__.py ./src/services/__init__.py

# Create directories
RUN mkdir -p /app/data/models/cnn-lstm /app/logs

# Environment
ENV PYTHONPATH=/app
ENV SERVICE_NAME=cnn-lstm-train
ENV PORT=${SERVICE_PORT}
ENV PYTHONUNBUFFERED=1
ENV NICE_PRIORITY=19

# Limit threads for CPU training
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV NUMEXPR_MAX_THREADS=4

EXPOSE ${SERVICE_PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

# Run with low CPU priority (nice -n 19)
CMD ["sh", "-c", "nice -n ${NICE_PRIORITY:-19} python -m uvicorn src.services.cnn_lstm_train_app.main:app --host 0.0.0.0 --port ${PORT:-3017}"]
