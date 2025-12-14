# NHITS Service Dockerfile
FROM nvcr.io/nvidia/pytorch:23.08-py3

LABEL maintainer="KI Trading Model"
LABEL service="nhits"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY .env /app/.env

# Create directories for models and logs
RUN mkdir -p /app/data/models /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV SERVICE_NAME=nhits
ENV PORT=3001

# Expose port
EXPOSE 3001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3001/health || exit 1

# Run service
CMD ["python", "-m", "uvicorn", "src.services.nhits_app.main:app", "--host", "0.0.0.0", "--port", "3001"]
