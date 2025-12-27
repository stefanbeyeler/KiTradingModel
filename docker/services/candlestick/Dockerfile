# Candlestick Pattern Service - Inference
# Port: 3006

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY docker/services/candlestick/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/

# Create directories
RUN mkdir -p /app/logs /app/data

# Environment variables
ENV PYTHONPATH=/app
ENV PORT=3006
ENV LOG_LEVEL=INFO
ENV DATA_SERVICE_URL=http://trading-data:3001
ENV AUTO_SCAN_ENABLED=true
ENV SCAN_INTERVAL_SECONDS=300
ENV MAX_HISTORY_ENTRIES=1000
ENV MAX_HISTORY_AGE_HOURS=24
ENV DATA_DIR=/app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3006/health')" || exit 1

# Expose port
EXPOSE 3006

# Run service
CMD ["python", "-m", "uvicorn", "src.services.candlestick_app.main:app", "--host", "0.0.0.0", "--port", "3006"]
