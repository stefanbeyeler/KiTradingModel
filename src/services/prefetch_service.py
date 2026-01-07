"""
Pre-Fetching Service für optimierte Cache-Bereitstellung.

Dieser Service sorgt dafür, dass häufig benötigte Daten bereits im Cache
verfügbar sind, bevor sie von nachgelagerten Services (NHITS, TCN, HMM)
angefordert werden.

Strategien:
1. Startup Pre-Fetch: Lädt kritische Daten beim Start
2. Scheduled Pre-Fetch: Aktualisiert Daten vor TTL-Ablauf
3. Priority-basiert: Favoriten und aktive Symbole zuerst
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional
from loguru import logger

from .cache_service import cache_service, CacheCategory, TIMEFRAME_TTL, DEFAULT_TTL


class PrefetchConfig:
    """Konfiguration für Pre-Fetching."""

    # Verfügbare Indikatoren für Pre-Fetching (ML-relevant)
    AVAILABLE_INDICATORS = [
        "rsi",      # Relative Strength Index - Momentum
        "macd",     # Moving Average Convergence Divergence
        "bbands",   # Bollinger Bands - Volatilität
        "atr",      # Average True Range - Volatilität
        "ema",      # Exponential Moving Average
        "sma",      # Simple Moving Average
        "adx",      # Average Directional Index - Trend Strength
        "stoch",    # Stochastic Oscillator
        "obv",      # On-Balance Volume
        "vwap",     # Volume Weighted Average Price
    ]

    def __init__(
        self,
        enabled: bool = True,
        # Welche Timeframes sollen pre-fetched werden
        timeframes: list[str] = None,
        # Maximale Anzahl Symbole für Pre-Fetch
        max_symbols: int = 50,
        # Nur Favoriten pre-fetchen
        favorites_only: bool = False,
        # Intervall für periodisches Pre-Fetching (Sekunden)
        refresh_interval: int = 300,
        # Output-Size für OHLCV-Daten
        ohlcv_limit: int = 500,
        # Verzögerung zwischen API-Aufrufen (Rate Limiting)
        api_delay: float = 0.2,
        # Welche Indikatoren sollen pre-fetched werden (leer = keine)
        indicators: list[str] = None,
        # Output-Size für Indikatoren
        indicator_limit: int = 100,
    ):
        self.enabled = enabled
        self.timeframes = timeframes or ["1h", "4h", "1day"]
        self.max_symbols = max_symbols
        self.favorites_only = favorites_only
        self.refresh_interval = refresh_interval
        self.ohlcv_limit = ohlcv_limit
        self.api_delay = api_delay
        # Indikatoren: Nur gültige aus der verfügbaren Liste akzeptieren
        self.indicators = [
            ind for ind in (indicators or [])
            if ind.lower() in self.AVAILABLE_INDICATORS
        ]
        self.indicator_limit = indicator_limit


class PrefetchService:
    """
    Service für automatisches Pre-Fetching von Marktdaten.

    Optimiert für nachgelagerte ML-Services:
    - NHITS: Benötigt H1, D1 für Forecasting
    - TCN: Benötigt M15-D1 für Pattern-Erkennung
    - HMM: Benötigt H1, H4, D1 für Regime-Detection
    """

    def __init__(self):
        self._config = PrefetchConfig()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._stats = {
            "last_run": None,
            "symbols_fetched": 0,
            "timeframes_fetched": 0,
            "indicators_fetched": 0,
            "errors": 0,
            "total_runs": 0,
            "cache_entries_created": 0,
            "indicator_entries_created": 0,
        }
        self._symbols_cache: list[dict] = []

    def configure(self, **kwargs) -> None:
        """Aktualisiert die Pre-Fetch-Konfiguration."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        logger.info(f"PrefetchService konfiguriert: {kwargs}")

    async def start(self) -> None:
        """Startet den Pre-Fetching Service."""
        if self._running:
            logger.warning("PrefetchService läuft bereits")
            return

        if not self._config.enabled:
            logger.info("PrefetchService ist deaktiviert")
            return

        self._running = True
        logger.info(
            f"PrefetchService gestartet - Timeframes: {self._config.timeframes}, "
            f"Intervall: {self._config.refresh_interval}s"
        )

        # Initial Pre-Fetch
        await self._run_prefetch()

        # Periodisches Pre-Fetch
        self._task = asyncio.create_task(self._periodic_prefetch())

    async def stop(self) -> None:
        """Stoppt den Pre-Fetching Service."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("PrefetchService gestoppt")

    async def _periodic_prefetch(self) -> None:
        """Periodisches Pre-Fetching im Hintergrund."""
        while self._running:
            try:
                await asyncio.sleep(self._config.refresh_interval)
                if self._running:
                    await self._run_prefetch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im periodischen Pre-Fetch: {e}")
                self._stats["errors"] += 1

    async def _run_prefetch(self) -> None:
        """Führt einen Pre-Fetch-Durchlauf durch."""
        start_time = datetime.now(timezone.utc)
        self._stats["total_runs"] += 1

        try:
            # 1. Symbole laden
            symbols = await self._get_symbols()
            if not symbols:
                logger.warning("Keine Symbole für Pre-Fetch gefunden")
                return

            # 2. Nach Priorität sortieren (Favoriten zuerst)
            symbols = self._prioritize_symbols(symbols)

            # 3. Limitieren
            symbols = symbols[:self._config.max_symbols]

            indicators_info = f", {len(self._config.indicators)} Indikatoren" if self._config.indicators else ""
            logger.info(
                f"Pre-Fetch gestartet für {len(symbols)} Symbole, "
                f"{len(self._config.timeframes)} Timeframes{indicators_info}"
            )

            # 4. OHLCV-Daten pre-fetchen
            fetched_count = 0
            indicator_count = 0
            for symbol_data in symbols:
                # Verwende immer display symbol (BTCUSD) - Konvertierung erfolgt im TwelveDataService
                symbol = (
                    symbol_data.get("symbol") or
                    symbol_data.get("name", "")
                )
                if not symbol:
                    continue

                for timeframe in self._config.timeframes:
                    try:
                        success = await self._prefetch_ohlcv(symbol, timeframe)
                        if success:
                            fetched_count += 1
                        # Rate Limiting
                        await asyncio.sleep(self._config.api_delay)
                    except Exception as e:
                        logger.error(f"Pre-Fetch Fehler für {symbol}/{timeframe}: {e}")
                        self._stats["errors"] += 1

                    # 5. Indikatoren pre-fetchen (falls konfiguriert)
                    for indicator in self._config.indicators:
                        try:
                            success = await self._prefetch_indicator(symbol, timeframe, indicator)
                            if success:
                                indicator_count += 1
                            # Rate Limiting
                            await asyncio.sleep(self._config.api_delay)
                        except Exception as e:
                            logger.error(f"Pre-Fetch Fehler für {indicator}/{symbol}/{timeframe}: {e}")
                            self._stats["errors"] += 1

            self._stats["last_run"] = start_time.isoformat()
            self._stats["symbols_fetched"] = len(symbols)
            self._stats["timeframes_fetched"] = len(self._config.timeframes)
            self._stats["indicators_fetched"] = len(self._config.indicators)
            self._stats["cache_entries_created"] = fetched_count
            self._stats["indicator_entries_created"] = indicator_count

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            indicator_info = f", {indicator_count} Indikatoren" if indicator_count else ""
            logger.info(
                f"Pre-Fetch abgeschlossen: {fetched_count} OHLCV{indicator_info} in {duration:.1f}s"
            )

        except Exception as e:
            logger.error(f"Pre-Fetch Durchlauf fehlgeschlagen: {e}")
            self._stats["errors"] += 1

    async def _get_symbols(self) -> list[dict]:
        """Lädt die Symbol-Liste direkt vom Symbol-Service."""
        # Versuche aus Cache
        cached = await cache_service.get(CacheCategory.SYMBOLS, "prefetch_list")
        if cached:
            return cached

        # Direkt über Symbol-Service laden (wir sind im gleichen Container)
        try:
            from .symbol_service import symbol_service

            # Alle Symbole laden
            all_symbols = await symbol_service.get_all_symbols()

            # Zu dict-Liste konvertieren
            symbols = [
                {
                    "symbol": s.symbol,
                    "display_name": s.display_name,
                    "category": s.category.value if s.category else "other",
                    "is_favorite": s.is_favorite,
                    "status": s.status.value if s.status else "active",
                    "twelvedata_symbol": s.twelvedata_symbol,
                }
                for s in all_symbols
            ]

            # Favoriten-Filter falls konfiguriert
            if self._config.favorites_only:
                symbols = [s for s in symbols if s.get("is_favorite")]

            # Cache für 1 Stunde
            if symbols:
                await cache_service.set(
                    CacheCategory.SYMBOLS, symbols, "prefetch_list",
                    ttl=3600
                )
                self._symbols_cache = symbols

            return symbols

        except Exception as e:
            logger.error(f"Fehler beim Laden der Symbole: {e}")

        return self._symbols_cache or []

    def _prioritize_symbols(self, symbols: list[dict]) -> list[dict]:
        """Sortiert Symbole nach Priorität."""
        def priority_key(s: dict) -> tuple:
            is_favorite = s.get("is_favorite", False)
            is_active = s.get("status") == "active"
            category = s.get("category", "other")

            # Priorität: Favoriten > Aktiv > Kategorie
            cat_priority = {
                "crypto": 1,
                "forex": 2,
                "index": 3,
                "commodity": 4,
                "stock": 5,
                "etf": 6,
                "other": 7,
            }

            return (
                0 if is_favorite else 1,
                0 if is_active else 1,
                cat_priority.get(category, 7),
            )

        return sorted(symbols, key=priority_key)

    async def _prefetch_ohlcv(self, symbol: str, timeframe: str) -> bool:
        """
        Pre-fetcht OHLCV-Daten für ein Symbol/Timeframe.

        Prüft zuerst, ob bereits gültige Daten im Cache sind.
        Nutzt TwelveData Service direkt (gleicher Container).
        """
        # Normalize symbol for cache key
        cache_symbol = symbol.upper().replace("/", "")
        cache_params = {"interval": timeframe, "outputsize": self._config.ohlcv_limit}

        # Prüfe ob Cache noch gültig ist
        cached = await cache_service.get(
            CacheCategory.OHLCV, cache_symbol, timeframe, params=cache_params
        )
        if cached:
            # Bereits im Cache - kein Fetch nötig
            return False

        # Daten direkt über TwelveData Service abrufen
        try:
            from .twelvedata_service import twelvedata_service

            result = await twelvedata_service.get_time_series(
                symbol=symbol,
                interval=timeframe,
                outputsize=self._config.ohlcv_limit,
            )

            if result and result.get("values"):
                values = result["values"]
                logger.debug(f"Pre-fetched {len(values)} rows for {symbol}/{timeframe}")
                return True

        except Exception as e:
            logger.warning(f"Pre-Fetch Fehler für {symbol}/{timeframe}: {e}")

        return False

    async def _prefetch_indicator(self, symbol: str, timeframe: str, indicator: str) -> bool:
        """
        Pre-fetcht einen technischen Indikator für ein Symbol/Timeframe.

        Args:
            symbol: Trading-Symbol (z.B. BTCUSD)
            timeframe: Zeitrahmen (z.B. 1h, 4h, 1day)
            indicator: Indikator-Name (z.B. rsi, macd, bbands)

        Returns:
            True wenn neue Daten geladen wurden, False wenn bereits gecacht
        """
        # Normalize symbol for cache key
        cache_symbol = symbol.upper().replace("/", "")
        cache_params = {
            "interval": timeframe,
            "outputsize": self._config.indicator_limit,
            "indicator": indicator.lower()
        }

        # Prüfe ob Cache noch gültig ist
        cached = await cache_service.get(
            CacheCategory.INDICATORS, cache_symbol, timeframe, params=cache_params
        )
        if cached:
            return False

        # Indikator über TwelveData Service abrufen
        try:
            from .twelvedata_service import twelvedata_service

            result = await twelvedata_service.get_technical_indicators(
                symbol=symbol,
                interval=timeframe,
                indicator=indicator,
                outputsize=self._config.indicator_limit,
            )

            if result and result.get("values") and not result.get("error"):
                values = result["values"]
                # Cache the result
                await cache_service.set(
                    CacheCategory.INDICATORS,
                    result,
                    cache_symbol,
                    timeframe,
                    params=cache_params,
                    ttl=300  # 5 Minuten TTL für Indikatoren
                )
                logger.debug(f"Pre-fetched {indicator.upper()} for {symbol}/{timeframe} ({len(values) if isinstance(values, list) else 1} values)")
                return True

        except Exception as e:
            logger.warning(f"Pre-Fetch Fehler für {indicator}/{symbol}/{timeframe}: {e}")

        return False

    async def prefetch_symbol(self, symbol: str, timeframes: list[str] = None) -> dict:
        """
        Pre-fetcht Daten für ein einzelnes Symbol.

        Args:
            symbol: Das Symbol
            timeframes: Optionale Liste von Timeframes (default: config)

        Returns:
            Dictionary mit Pre-Fetch-Ergebnissen
        """
        timeframes = timeframes or self._config.timeframes
        results = {"symbol": symbol, "timeframes": {}, "errors": []}

        for tf in timeframes:
            try:
                success = await self._prefetch_ohlcv(symbol, tf)
                results["timeframes"][tf] = "cached" if not success else "fetched"
            except Exception as e:
                results["timeframes"][tf] = "error"
                results["errors"].append(f"{tf}: {str(e)}")

        return results

    async def prefetch_all_active(self) -> dict:
        """
        Pre-fetcht Daten für alle aktiven Symbole.

        Returns:
            Dictionary mit Zusammenfassung
        """
        symbols = await self._get_symbols()
        active_symbols = [s for s in symbols if s.get("status") == "active"]

        results = {
            "total_symbols": len(active_symbols),
            "timeframes": self._config.timeframes,
            "fetched": 0,
            "cached": 0,
            "errors": 0,
        }

        for symbol_data in active_symbols[:self._config.max_symbols]:
            symbol = symbol_data.get("symbol", "")
            for tf in self._config.timeframes:
                try:
                    success = await self._prefetch_ohlcv(symbol, tf)
                    if success:
                        results["fetched"] += 1
                    else:
                        results["cached"] += 1
                    await asyncio.sleep(self._config.api_delay)
                except Exception:
                    results["errors"] += 1

        return results

    def get_stats(self) -> dict:
        """Gibt Pre-Fetch-Statistiken zurück."""
        return {
            **self._stats,
            "config": {
                "enabled": self._config.enabled,
                "timeframes": self._config.timeframes,
                "max_symbols": self._config.max_symbols,
                "favorites_only": self._config.favorites_only,
                "refresh_interval": self._config.refresh_interval,
                "ohlcv_limit": self._config.ohlcv_limit,
                "indicators": self._config.indicators,
                "indicator_limit": self._config.indicator_limit,
            },
            "running": self._running,
        }

    def get_config(self) -> dict:
        """Gibt die aktuelle Konfiguration zurück."""
        return {
            "enabled": self._config.enabled,
            "timeframes": self._config.timeframes,
            "max_symbols": self._config.max_symbols,
            "favorites_only": self._config.favorites_only,
            "refresh_interval": self._config.refresh_interval,
            "ohlcv_limit": self._config.ohlcv_limit,
            "api_delay": self._config.api_delay,
            "indicators": self._config.indicators,
            "indicator_limit": self._config.indicator_limit,
            "available_indicators": PrefetchConfig.AVAILABLE_INDICATORS,
        }


# Singleton-Instanz
prefetch_service = PrefetchService()
