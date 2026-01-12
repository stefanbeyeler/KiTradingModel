"""
Config Router für Trading Workplace Service.

Verwaltet TradingView und andere Service-Konfigurationen.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from ..models.schemas import (
    TradingViewConfig,
    TradingViewTestRequest,
    TradingViewTestResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pfad für Konfigurationsdateien
CONFIG_DIR = Path("data/workplace")
TRADINGVIEW_CONFIG_FILE = CONFIG_DIR / "tradingview_config.json"


def ensure_config_dir():
    """Stellt sicher, dass das Konfigurationsverzeichnis existiert."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_tradingview_config() -> TradingViewConfig:
    """Lädt die TradingView-Konfiguration aus der Datei."""
    ensure_config_dir()

    if TRADINGVIEW_CONFIG_FILE.exists():
        try:
            with open(TRADINGVIEW_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return TradingViewConfig(**data)
        except Exception as e:
            logger.warning(f"Fehler beim Laden der TradingView-Config: {e}")

    return TradingViewConfig()


def save_tradingview_config(config: TradingViewConfig) -> bool:
    """Speichert die TradingView-Konfiguration in die Datei."""
    ensure_config_dir()

    try:
        with open(TRADINGVIEW_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config.model_dump(), f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Fehler beim Speichern der TradingView-Config: {e}")
        return False


@router.get(
    "/tradingview",
    response_model=TradingViewConfig,
    summary="TradingView Konfiguration abrufen",
    description="Gibt die aktuelle TradingView-Konfiguration zurück."
)
async def get_tradingview_config() -> TradingViewConfig:
    """Ruft die TradingView-Konfiguration ab."""
    return load_tradingview_config()


@router.post(
    "/tradingview",
    response_model=TradingViewConfig,
    summary="TradingView Konfiguration speichern",
    description="Speichert die TradingView-Konfiguration."
)
async def save_tradingview_config_endpoint(config: TradingViewConfig) -> TradingViewConfig:
    """Speichert die TradingView-Konfiguration."""
    if not save_tradingview_config(config):
        raise HTTPException(status_code=500, detail="Fehler beim Speichern der Konfiguration")

    logger.info("TradingView-Konfiguration gespeichert")
    return config


@router.post(
    "/tradingview/test",
    response_model=TradingViewTestResponse,
    summary="TradingView Verbindung testen",
    description="Testet die TradingView-Verbindung mit den angegebenen Anmeldedaten."
)
async def test_tradingview_connection(request: TradingViewTestRequest) -> TradingViewTestResponse:
    """
    Testet die TradingView-Verbindung.

    Da TradingView kein offizielles API für Authentifizierung bietet,
    validieren wir nur die Eingaben. Das Widget funktioniert auch ohne Anmeldung.
    """
    if not request.username and not request.session_id:
        return TradingViewTestResponse(
            valid=True,
            error=None
        )

    # Basis-Validierung
    if request.session_id and len(request.session_id) < 20:
        return TradingViewTestResponse(
            valid=False,
            error="Session ID zu kurz - bitte vollständige Session ID eingeben"
        )

    if request.username and len(request.username) < 3:
        return TradingViewTestResponse(
            valid=False,
            error="Benutzername zu kurz"
        )

    # TradingView-Verbindung kann nicht direkt getestet werden
    # Das Widget verwendet die Anmeldedaten automatisch wenn vorhanden
    return TradingViewTestResponse(
        valid=True,
        error=None
    )


@router.post(
    "/restart",
    summary="Service neu starten",
    description="Startet den Workplace-Service neu. Der Service wird kurzzeitig nicht erreichbar sein."
)
async def restart_service():
    """
    Startet den Workplace-Service neu.

    Der Service sendet ein SIGTERM an sich selbst, was einen sauberen Neustart auslöst.
    Docker/Supervisor wird den Service automatisch neu starten.
    """
    import os
    import signal
    import asyncio

    logger.info("Service-Neustart angefordert")

    # Verzögerten Neustart in separatem Task starten
    async def delayed_restart():
        await asyncio.sleep(1)  # Kurze Verzögerung damit Response noch gesendet wird
        logger.info("Service wird beendet für Neustart...")
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(delayed_restart())

    return {"status": "restarting", "message": "Service wird neu gestartet"}
