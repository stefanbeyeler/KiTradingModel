# LLM Service Dockerfile
FROM nvcr.io/nvidia/pytorch:23.08-py3

LABEL maintainer="KI Trading Model"
LABEL service="llm"

# Build argument for port
ARG SERVICE_PORT=3004

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

# Create directories for RAG and logs
RUN mkdir -p /app/data/rag /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV SERVICE_NAME=llm
ENV PORT=${SERVICE_PORT}

# Expose port
EXPOSE ${SERVICE_PORT}

# Health check - uses shell form to expand variable
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run service - uses shell form to expand variable
CMD python -m uvicorn src.services.llm_app.main:app --host 0.0.0.0 --port ${PORT}
