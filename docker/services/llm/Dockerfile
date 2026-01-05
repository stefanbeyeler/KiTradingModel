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

# Copy only the LLM app source code (avoid importing other services)
# ARCHITEKTUR: LLM greift auf Daten NUR über Data Service (Port 3001) zu
# twelvedata_service.py und yfinance_service.py werden NICHT kopiert - alle Datenzugriffe über HTTP
COPY src/services/llm_app/ ./src/services/llm_app/
COPY src/services/llm_service.py ./src/services/
COPY src/services/rag_service.py ./src/services/
COPY src/services/data_gateway_service.py ./src/services/
COPY src/services/cache_service.py ./src/services/
COPY src/api/ ./src/api/
COPY src/models/ ./src/models/
COPY src/config/ ./src/config/
COPY src/utils/ ./src/utils/
COPY src/shared/ ./src/shared/
COPY src/version.py ./src/

# Create empty __init__.py files to avoid import issues
RUN touch ./src/__init__.py ./src/services/__init__.py

# Copy .env for runtime configuration
COPY .env /app/.env

# Create directories for data (will be mounted as volumes)
RUN mkdir -p /app/data/rag /app/data/symbols /app/logs

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
