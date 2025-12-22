# HMM-Regime Service Dockerfile
# Port: 3004

FROM python:3.11-slim
ARG SERVICE_PORT=3004

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY docker/services/hmm/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the HMM app source code (avoid importing other services)
COPY src/services/hmm_app/ ./src/services/hmm_app/
COPY src/services/data_gateway_service.py ./src/services/
COPY src/config/ ./src/config/
COPY src/utils/ ./src/utils/
COPY src/shared/ ./src/shared/

# Create empty __init__.py files to avoid import issues
RUN touch ./src/__init__.py ./src/services/__init__.py

# Create directories
RUN mkdir -p /app/data/models/hmm /app/logs

ENV PYTHONPATH=/app
ENV SERVICE_NAME=hmm
ENV PORT=${SERVICE_PORT}

EXPOSE ${SERVICE_PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

# Dynamischer Port via Umgebungsvariable
CMD ["sh", "-c", "python -m uvicorn src.services.hmm_app.main:app --host 0.0.0.0 --port ${PORT:-3004}"]
