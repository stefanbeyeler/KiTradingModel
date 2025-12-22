#!/bin/bash
# ==============================================================================
# Migration Script: Host-Daten zu Docker Volumes
# ==============================================================================
# Dieses Script migriert die lokalen Daten aus ./data/ in Docker Volumes.
# Es muss nur einmalig ausgeführt werden, um bestehende Daten zu übernehmen.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"

# Docker Compose Projekt-Name (kleinbuchstaben vom Verzeichnisnamen)
PROJECT_NAME=$(basename "$PROJECT_ROOT" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]')

echo "=============================================="
echo "KI Trading Model - Data Migration to Volumes"
echo "=============================================="
echo ""
echo "Projekt-Root: $PROJECT_ROOT"
echo "Projekt-Name: $PROJECT_NAME"
echo "Daten-Verzeichnis: $DATA_DIR"
echo ""

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Prüfe ob Docker läuft
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Fehler: Docker ist nicht gestartet oder nicht erreichbar.${NC}"
    exit 1
fi

# Prüfe ob Datenverzeichnis existiert
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${YELLOW}Warnung: Kein data/ Verzeichnis gefunden. Nichts zu migrieren.${NC}"
    exit 0
fi

# Funktion zum Erstellen eines Volumes und Kopieren von Daten
migrate_to_volume() {
    local source_path="$1"
    local volume_name="$2"
    local description="$3"

    echo -n "Migriere $description... "

    if [ ! -d "$source_path" ] && [ ! -f "$source_path" ]; then
        echo -e "${YELLOW}Übersprungen (nicht vorhanden)${NC}"
        return 0
    fi

    # Erstelle Volume falls nicht vorhanden
    docker volume create "$volume_name" > /dev/null 2>&1 || true

    # Temporären Container verwenden um Daten zu kopieren
    docker run --rm \
        -v "$source_path":/source:ro \
        -v "$volume_name":/dest \
        alpine sh -c "cp -a /source/. /dest/" 2>/dev/null

    echo -e "${GREEN}OK${NC}"
}

# Funktion für einzelne Dateien
migrate_file_to_volume() {
    local source_file="$1"
    local volume_name="$2"
    local dest_filename="$3"
    local description="$4"

    echo -n "Migriere $description... "

    if [ ! -f "$source_file" ]; then
        echo -e "${YELLOW}Übersprungen (nicht vorhanden)${NC}"
        return 0
    fi

    # Erstelle Volume falls nicht vorhanden
    docker volume create "$volume_name" > /dev/null 2>&1 || true

    # Temporären Container verwenden um Datei zu kopieren
    docker run --rm \
        -v "$(dirname "$source_file")":/source:ro \
        -v "$volume_name":/dest \
        alpine sh -c "cp /source/$(basename "$source_file") /dest/$dest_filename" 2>/dev/null

    echo -e "${GREEN}OK${NC}"
}

echo "Starte Migration..."
echo ""

# ============================================================================
# NHITS Service Volumes
# ============================================================================
echo "--- NHITS Service ---"
migrate_to_volume "$DATA_DIR/models/nhits" "${PROJECT_NAME}_nhits-models" "NHITS Modelle"
migrate_to_volume "$DATA_DIR/model_feedback" "${PROJECT_NAME}_nhits-feedback" "Model Feedback"
migrate_to_volume "$DATA_DIR/model_metrics" "${PROJECT_NAME}_nhits-metrics" "Model Metrics"

# ============================================================================
# Data Service Volumes
# ============================================================================
echo ""
echo "--- Data Service ---"
migrate_file_to_volume "$DATA_DIR/symbols.json" "${PROJECT_NAME}_symbols-data" "symbols.json" "Symbols JSON"
migrate_to_volume "$DATA_DIR/faiss" "${PROJECT_NAME}_faiss-data" "FAISS Index"

# ============================================================================
# RAG Service Volumes
# ============================================================================
echo ""
echo "--- RAG Service ---"
migrate_to_volume "$DATA_DIR/faiss" "${PROJECT_NAME}_rag-faiss" "RAG FAISS Index"

# ============================================================================
# Zusammenfassung
# ============================================================================
echo ""
echo "=============================================="
echo -e "${GREEN}Migration abgeschlossen!${NC}"
echo "=============================================="
echo ""
echo "Die folgenden Docker Volumes wurden erstellt/aktualisiert:"
echo ""
docker volume ls --filter "name=${PROJECT_NAME}_" --format "  - {{.Name}}"
echo ""
echo "Nächste Schritte:"
echo "  1. Starte die Services mit: docker compose -f docker-compose.microservices.yml up -d"
echo "  2. Überprüfe die Logs: docker compose -f docker-compose.microservices.yml logs -f"
echo ""
echo "Optional: Nach erfolgreicher Verifizierung kannst du das lokale ./data/"
echo "Verzeichnis archivieren oder löschen:"
echo "  mv data data.backup.$(date +%Y%m%d)"
echo ""
