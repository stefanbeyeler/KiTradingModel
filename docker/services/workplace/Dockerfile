# Trading Workplace Service Dockerfile
# Aggregiert alle ML-Vorhersagen mit Multi-Signal-Scoring

FROM python:3.11-slim

LABEL maintainer="KI Trading Model"
LABEL service="workplace"
LABEL description="Trading-Setup-Aggregation mit Multi-Signal-Scoring"

ARG SERVICE_PORT=3020

WORKDIR /app

# System-Dependencies installieren
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python-Dependencies installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Zusätzliche Dependencies für den Workplace Service
RUN pip install --no-cache-dir httpx pydantic-settings

# Source-Code kopieren
COPY src/services/workplace_app/ ./src/services/workplace_app/
COPY src/config/ ./src/config/
COPY src/shared/ ./src/shared/
COPY src/utils/ ./src/utils/
COPY src/version.py ./src/

# Leere __init__.py Dateien erstellen
RUN touch ./src/__init__.py ./src/services/__init__.py

# .env kopieren
COPY .env /app/.env

# Datenverzeichnisse erstellen
RUN mkdir -p /app/data /app/logs

# Environment
ENV PYTHONPATH=/app
ENV SERVICE_NAME=workplace
ENV PORT=${SERVICE_PORT}

EXPOSE ${SERVICE_PORT}

# Health-Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Service starten
CMD python -m uvicorn src.services.workplace_app.main:app --host 0.0.0.0 --port ${PORT}
