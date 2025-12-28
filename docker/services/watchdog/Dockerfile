# Watchdog Service Dockerfile
FROM python:3.11-slim

LABEL maintainer="KI Trading Model"
LABEL service="watchdog"

# Build argument for port
ARG SERVICE_PORT=3010

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for watchdog
RUN pip install --no-cache-dir httpx python-telegram-bot

# Copy source code
COPY src/ /app/src/
COPY .env /app/.env

# Create directories for data (will be mounted as volumes)
RUN mkdir -p /app/data /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV SERVICE_NAME=watchdog
ENV PORT=${SERVICE_PORT}

# Expose port
EXPOSE ${SERVICE_PORT}

# Health check - uses shell form to expand variable
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run service - uses shell form to expand variable
CMD python -m uvicorn src.services.watchdog_app.main:app --host 0.0.0.0 --port ${PORT}
