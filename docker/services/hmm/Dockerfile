# HMM-Regime Service Dockerfile
# Port: 3006

FROM python:3.11-slim
ARG SERVICE_PORT=3006

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY docker/services/hmm/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create directories
RUN mkdir -p /app/data/models/hmm /app/logs

ENV PYTHONPATH=/app
ENV SERVICE_NAME=hmm
ENV PORT=${SERVICE_PORT}

EXPOSE ${SERVICE_PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

CMD ["python", "-m", "uvicorn", "src.services.hmm_app.main:app", "--host", "0.0.0.0", "--port", "3006"]
