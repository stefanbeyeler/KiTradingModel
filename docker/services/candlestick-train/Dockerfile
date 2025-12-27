# Candlestick Pattern Training Service
# Port: 3016

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY docker/services/candlestick-train/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/

# Create directories
RUN mkdir -p /app/logs /app/models

# Environment variables
ENV PYTHONPATH=/app
ENV PORT=3016
ENV LOG_LEVEL=INFO
ENV DATA_SERVICE_URL=http://trading-data:3001
ENV CANDLESTICK_SERVICE_URL=http://trading-candlestick:3006
ENV MODEL_DIR=/app/models
ENV SCHEDULED_TRAINING_ENABLED=false
ENV TRAINING_INTERVAL_HOURS=24
ENV TRAINING_HOUR_UTC=2
ENV SCHEDULED_TRAINING_EPOCHS=50

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3016/health')" || exit 1

# Expose port
EXPOSE 3016

# Run service
CMD ["python", "-m", "uvicorn", "src.services.candlestick_train_app.main:app", "--host", "0.0.0.0", "--port", "3016"]
