"""Configuration Export/Import Service - Backup and restore symbols and strategies."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from loguru import logger
import httpx

from ..config import settings
from src.config.microservices import microservices_config

# Workplace Service URL for strategy management
WORKPLACE_URL = os.getenv("WORKPLACE_SERVICE_URL", microservices_config.workplace_service_url)


class ConfigExportMetadata(BaseModel):
    """Metadata for a configuration export."""
    filename: str
    created_at: datetime
    description: Optional[str] = None
    symbols_count: int = 0
    strategies_count: int = 0
    file_size: int = 0
    version: str = "1.0"


class ConfigExportData(BaseModel):
    """Complete configuration export data."""
    version: str = "1.0"
    exported_at: datetime = Field(default_factory=datetime.utcnow)
    description: Optional[str] = None
    symbols: list[dict] = Field(default_factory=list)
    strategies: list[dict] = Field(default_factory=list)


class ConfigExportResult(BaseModel):
    """Result of an export operation."""
    success: bool
    filename: str
    message: str
    metadata: Optional[ConfigExportMetadata] = None


class ConfigImportResult(BaseModel):
    """Result of an import operation."""
    success: bool
    message: str
    symbols_imported: int = 0
    symbols_updated: int = 0
    symbols_skipped: int = 0
    strategies_imported: int = 0
    strategies_updated: int = 0
    strategies_skipped: int = 0
    errors: list[str] = Field(default_factory=list)


class ConfigExportService:
    """Service for exporting and importing configuration data."""

    def __init__(self):
        self._export_dir = Path(os.getenv("CONFIG_EXPORT_DIR", "data/exports"))
        self._symbols_file = Path("data/symbols.json")
        # Strategies are now managed by Workplace Service (Port 3020)
        self._workplace_url = WORKPLACE_URL
        self._ensure_export_dir()

    def _ensure_export_dir(self):
        """Ensure export directory exists."""
        self._export_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self, prefix: str = "config") -> str:
        """Generate a unique filename with timestamp."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.json"

    async def export_config(
        self,
        description: Optional[str] = None,
        include_symbols: bool = True,
        include_strategies: bool = True,
        filename: Optional[str] = None,
    ) -> ConfigExportResult:
        """
        Export current configuration to a JSON file.

        Args:
            description: Optional description for the export
            include_symbols: Include symbols in export
            include_strategies: Include strategies in export
            filename: Custom filename (auto-generated if not provided)
        """
        try:
            export_data = ConfigExportData(
                description=description,
                exported_at=datetime.utcnow(),
            )

            # Load symbols
            if include_symbols and self._symbols_file.exists():
                with open(self._symbols_file, "r", encoding="utf-8") as f:
                    export_data.symbols = json.load(f)

            # Load strategies from Workplace Service
            if include_strategies:
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get(f"{self._workplace_url}/api/v1/strategies")
                        if response.status_code == 200:
                            export_data.strategies = response.json()
                        else:
                            logger.warning(f"Could not fetch strategies from Workplace: {response.status_code}")
                except Exception as e:
                    logger.warning(f"Could not fetch strategies from Workplace: {e}")

            # Generate filename
            if not filename:
                filename = self._generate_filename()

            if not filename.endswith(".json"):
                filename += ".json"

            # Write export file
            export_path = self._export_dir / filename
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data.model_dump(mode="json"), f, indent=2, default=str)

            # Create metadata
            metadata = ConfigExportMetadata(
                filename=filename,
                created_at=datetime.utcnow(),
                description=description,
                symbols_count=len(export_data.symbols),
                strategies_count=len(export_data.strategies),
                file_size=export_path.stat().st_size,
            )

            logger.info(f"Configuration exported to {filename}: {metadata.symbols_count} symbols, {metadata.strategies_count} strategies")

            return ConfigExportResult(
                success=True,
                filename=filename,
                message=f"Export erfolgreich: {metadata.symbols_count} Symbole, {metadata.strategies_count} Strategien",
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ConfigExportResult(
                success=False,
                filename=filename or "",
                message=f"Export fehlgeschlagen: {str(e)}",
            )

    async def import_config(
        self,
        data: dict,
        import_symbols: bool = True,
        import_strategies: bool = True,
        overwrite_existing: bool = False,
    ) -> ConfigImportResult:
        """
        Import configuration from uploaded data.

        Args:
            data: The configuration data to import
            import_symbols: Import symbols from data
            import_strategies: Import strategies from data
            overwrite_existing: Overwrite existing entries
        """
        result = ConfigImportResult(success=True, message="")
        errors = []

        try:
            # Import symbols
            if import_symbols and "symbols" in data:
                symbols_result = await self._import_symbols(
                    data["symbols"],
                    overwrite_existing
                )
                result.symbols_imported = symbols_result["imported"]
                result.symbols_updated = symbols_result["updated"]
                result.symbols_skipped = symbols_result["skipped"]
                errors.extend(symbols_result["errors"])

            # Import strategies
            if import_strategies and "strategies" in data:
                strategies_result = await self._import_strategies(
                    data["strategies"],
                    overwrite_existing
                )
                result.strategies_imported = strategies_result["imported"]
                result.strategies_updated = strategies_result["updated"]
                result.strategies_skipped = strategies_result["skipped"]
                errors.extend(strategies_result["errors"])

            result.errors = errors
            result.message = (
                f"Import abgeschlossen: "
                f"{result.symbols_imported} Symbole importiert, "
                f"{result.symbols_updated} aktualisiert, "
                f"{result.strategies_imported} Strategien importiert, "
                f"{result.strategies_updated} aktualisiert"
            )

            if errors:
                result.message += f" ({len(errors)} Fehler)"

            logger.info(result.message)

        except Exception as e:
            logger.error(f"Import failed: {e}")
            result.success = False
            result.message = f"Import fehlgeschlagen: {str(e)}"

        return result

    async def _import_symbols(
        self,
        symbols: list[dict],
        overwrite: bool
    ) -> dict:
        """Import symbols into the system."""
        result = {"imported": 0, "updated": 0, "skipped": 0, "errors": []}

        # Load existing symbols
        existing = {}
        if self._symbols_file.exists():
            with open(self._symbols_file, "r", encoding="utf-8") as f:
                for s in json.load(f):
                    existing[s.get("symbol")] = s

        for symbol_data in symbols:
            try:
                symbol_id = symbol_data.get("symbol")
                if not symbol_id:
                    result["errors"].append("Symbol ohne ID übersprungen")
                    result["skipped"] += 1
                    continue

                if symbol_id in existing:
                    if overwrite:
                        # Update existing
                        existing[symbol_id].update(symbol_data)
                        existing[symbol_id]["updated_at"] = datetime.utcnow().isoformat()
                        result["updated"] += 1
                    else:
                        result["skipped"] += 1
                else:
                    # Add new
                    symbol_data["created_at"] = datetime.utcnow().isoformat()
                    symbol_data["updated_at"] = datetime.utcnow().isoformat()
                    existing[symbol_id] = symbol_data
                    result["imported"] += 1

            except Exception as e:
                result["errors"].append(f"Fehler bei Symbol {symbol_data.get('symbol', '?')}: {str(e)}")
                result["skipped"] += 1

        # Save updated symbols
        self._symbols_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._symbols_file, "w", encoding="utf-8") as f:
            json.dump(list(existing.values()), f, indent=2, default=str)

        return result

    async def _import_strategies(
        self,
        strategies: list[dict],
        overwrite: bool
    ) -> dict:
        """Import strategies to Workplace Service."""
        result = {"imported": 0, "updated": 0, "skipped": 0, "errors": []}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Fetch existing strategies from Workplace
                existing = {}
                try:
                    response = await client.get(f"{self._workplace_url}/api/v1/strategies")
                    if response.status_code == 200:
                        for s in response.json():
                            existing[s.get("id")] = s
                except Exception as e:
                    logger.warning(f"Could not fetch existing strategies: {e}")

                for strategy_data in strategies:
                    try:
                        strategy_id = strategy_data.get("id")
                        if not strategy_id:
                            result["errors"].append("Strategie ohne ID übersprungen")
                            result["skipped"] += 1
                            continue

                        is_existing = strategy_id in existing

                        if is_existing and not overwrite:
                            result["skipped"] += 1
                            continue

                        # Use the import endpoint which handles both create and update
                        response = await client.post(
                            f"{self._workplace_url}/api/v1/strategies/import",
                            json=strategy_data
                        )
                        if response.status_code == 200:
                            if is_existing:
                                result["updated"] += 1
                            else:
                                result["imported"] += 1
                        else:
                            result["errors"].append(f"Import fehlgeschlagen für {strategy_id}: {response.text}")
                            result["skipped"] += 1

                    except Exception as e:
                        result["errors"].append(f"Fehler bei Strategie {strategy_data.get('id', '?')}: {str(e)}")
                        result["skipped"] += 1

        except Exception as e:
            result["errors"].append(f"Verbindung zum Workplace Service fehlgeschlagen: {str(e)}")
            logger.error(f"Could not connect to Workplace Service: {e}")

        return result

    async def list_exports(self) -> list[ConfigExportMetadata]:
        """List all available export files."""
        exports = []

        for file_path in sorted(self._export_dir.glob("*.json"), reverse=True):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                exports.append(ConfigExportMetadata(
                    filename=file_path.name,
                    created_at=datetime.fromisoformat(data.get("exported_at", datetime.utcnow().isoformat()).replace("Z", "+00:00").split("+")[0]),
                    description=data.get("description"),
                    symbols_count=len(data.get("symbols", [])),
                    strategies_count=len(data.get("strategies", [])),
                    file_size=file_path.stat().st_size,
                    version=data.get("version", "1.0"),
                ))
            except Exception as e:
                logger.warning(f"Could not read export file {file_path}: {e}")

        return exports

    async def get_export(self, filename: str) -> Optional[dict]:
        """Get export file content by filename."""
        file_path = self._export_dir / filename

        if not file_path.exists():
            return None

        # Security check - prevent path traversal
        if not file_path.resolve().is_relative_to(self._export_dir.resolve()):
            logger.warning(f"Path traversal attempt blocked: {filename}")
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def delete_export(self, filename: str) -> bool:
        """Delete an export file."""
        file_path = self._export_dir / filename

        if not file_path.exists():
            return False

        # Security check
        if not file_path.resolve().is_relative_to(self._export_dir.resolve()):
            logger.warning(f"Path traversal attempt blocked: {filename}")
            return False

        file_path.unlink()
        logger.info(f"Deleted export file: {filename}")
        return True

    async def import_from_saved(
        self,
        filename: str,
        import_symbols: bool = True,
        import_strategies: bool = True,
        overwrite_existing: bool = False,
    ) -> ConfigImportResult:
        """Import configuration from a saved export file."""
        data = await self.get_export(filename)

        if data is None:
            return ConfigImportResult(
                success=False,
                message=f"Export-Datei '{filename}' nicht gefunden",
            )

        return await self.import_config(
            data=data,
            import_symbols=import_symbols,
            import_strategies=import_strategies,
            overwrite_existing=overwrite_existing,
        )


# Global service instance
config_export_service = ConfigExportService()
