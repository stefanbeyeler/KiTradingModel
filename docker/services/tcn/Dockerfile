# TCN-Pattern Service Dockerfile
# Port: 3005

FROM python:3.11-slim
ARG SERVICE_PORT=3005

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY docker/services/tcn/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create directories
RUN mkdir -p /app/data/models/tcn /app/logs

ENV PYTHONPATH=/app
ENV SERVICE_NAME=tcn
ENV PORT=${SERVICE_PORT}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE ${SERVICE_PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

CMD ["python", "-m", "uvicorn", "src.services.tcn_app.main:app", "--host", "0.0.0.0", "--port", "3005"]
