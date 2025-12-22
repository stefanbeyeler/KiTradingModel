"""
Standardized Logging Configuration
===================================

Einheitliche Logging-Konfiguration für alle Microservices.
"""

import sys
from loguru import logger
from typing import Optional


def setup_logging(
    service_name: str,
    log_level: str = "INFO",
    log_dir: str = "logs",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> "logger":
    """
    Konfiguriert das Logging für einen Service.

    Args:
        service_name: Name des Services (für Log-Prefix und Dateiname)
        log_level: Log-Level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Verzeichnis für Log-Dateien
        rotation: Wann Log-Dateien rotiert werden
        retention: Wie lange Log-Dateien aufbewahrt werden

    Returns:
        Konfigurierter Logger
    """
    # Bestehende Handler entfernen
    logger.remove()

    # Format-String für Konsole (mit Farben)
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        f"<cyan>{service_name.upper()}</cyan> | "
        "<level>{message}</level>"
    )

    # Format-String für Datei (ohne Farben)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        f"{service_name.upper()} | "
        "{message}"
    )

    # Console Handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=log_level,
        colorize=True,
    )

    # File Handler
    logger.add(
        f"{log_dir}/{service_name}_{{time}}.log",
        format=file_format,
        level="DEBUG",  # Datei bekommt immer DEBUG
        rotation=rotation,
        retention=retention,
        compression="gz",
    )

    # Error-spezifischer File Handler
    logger.add(
        f"{log_dir}/{service_name}_errors_{{time}}.log",
        format=file_format,
        level="ERROR",
        rotation=rotation,
        retention=retention,
        compression="gz",
    )

    return logger


def log_startup_info(
    service_name: str,
    version: str,
    port: int,
    gpu_available: bool = False,
    gpu_name: Optional[str] = None,
) -> None:
    """
    Loggt standardisierte Startup-Informationen.

    Args:
        service_name: Name des Services
        version: Service-Version
        port: Port auf dem der Service läuft
        gpu_available: GPU verfügbar
        gpu_name: Name der GPU
    """
    logger.info("=" * 60)
    logger.info(f"  {service_name.upper()} SERVICE v{version}")
    logger.info("=" * 60)
    logger.info(f"  Port: {port}")
    if gpu_available:
        logger.info(f"  GPU: {gpu_name}")
    else:
        logger.info("  GPU: Not available (using CPU)")
    logger.info("=" * 60)


def log_shutdown_info(service_name: str) -> None:
    """Loggt Shutdown-Informationen."""
    logger.info("=" * 60)
    logger.info(f"  {service_name.upper()} SERVICE SHUTDOWN")
    logger.info("=" * 60)
