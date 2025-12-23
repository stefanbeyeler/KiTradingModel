# Data Service Dockerfile
FROM python:3.11-slim

LABEL maintainer="KI Trading Model"
LABEL service="data"

# Build argument for port
ARG SERVICE_PORT=3001

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install pytest and test dependencies for testing API
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-timeout httpx

# Copy source code
COPY src/ /app/src/
COPY .env /app/.env

# Create directories for data (will be mounted as volumes)
RUN mkdir -p /app/data /app/data/symbols /app/data/faiss /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV SERVICE_NAME=data
ENV PORT=${SERVICE_PORT}

# Expose port
EXPOSE ${SERVICE_PORT}

# Health check - uses shell form to expand variable
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run service - uses shell form to expand variable
CMD python -m uvicorn src.services.data_app.main:app --host 0.0.0.0 --port ${PORT}
