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
        # Momentum (-> indicators_momentum)
        "rsi",          # Relative Strength Index
        "macd",         # Moving Average Convergence Divergence
        "adx",          # Average Directional Index
        "stoch",        # Stochastic Oscillator
        "cci",          # Commodity Channel Index
        "willr",        # Williams %R
        "roc",          # Rate of Change
        "mom",          # Momentum
        # Volatility (-> indicators_volatility)
        "bbands",       # Bollinger Bands
        "atr",          # Average True Range
        "natr",         # Normalized ATR
        # Moving Averages (-> indicators_ma)
        "ema",          # Exponential Moving Average
        "sma",          # Simple Moving Average
        "wma",          # Weighted Moving Average
        "dema",         # Double EMA
        "tema",         # Triple EMA
        # Volume (-> indicators_volume)
        "obv",          # On-Balance Volume
        "ad",           # Accumulation/Distribution
        "adosc",        # A/D Oscillator
        # Trend (-> indicators_trend)
        "supertrend",   # Supertrend
        "ichimoku",     # Ichimoku Cloud
        "sar",          # Parabolic SAR
        "aroon",        # Aroon Up/Down
        "aroonosc",     # Aroon Oscillator
        # Levels (-> indicators_levels): Lokal berechnet via pivot_calculator.py
        # (pivot_points_hl requires TwelveData Pro plan)
        # Other
        "vwap",         # Volume Weighted Average Price
    ]

    # Optimierte Limits pro Timeframe für 1 Jahr Abdeckung
    # Berechnung: Tage * Kerzen/Tag * (5/7 für Wochenenden bei Intraday)
    # TwelveData max: 5000 pro Request
    TIMEFRAME_LIMITS: dict[str, int] = {
        "1min": 5000,    # ~3.5 Trading-Tage (1440 * 5/7 = 1028/Tag)
        "5min": 5000,    # ~17 Trading-Tage (288 * 5/7 = 206/Tag)
        "15min": 5000,   # ~52 Trading-Tage (96 * 5/7 = 69/Tag)
        "30min": 5000,   # ~104 Trading-Tage (48 * 5/7 = 34/Tag)
        "1h": 5000,      # ~208 Trading-Tage (~10 Monate)
        "4h": 5000,      # ~833 Trading-Tage (~3.3 Jahre)
        "1day": 1000,    # 1000 Tage (~2.7 Jahre)
        "1week": 520,    # 520 Wochen (~10 Jahre)
        "1month": 240,   # 240 Monate (~20 Jahre)
    }

    def __init__(
        self,
        enabled: bool = True,
        # Welche Timeframes sollen pre-fetched werden (langsam, alle 10 Min)
        timeframes: list[str] = None,
        # Schnelle Timeframes (M1-M30, jede Minute)
        fast_timeframes: list[str] = None,
        # Intervall für schnelle Timeframes (Sekunden)
        fast_refresh_interval: int = 60,
        # Maximale Anzahl Symbole für Pre-Fetch
        max_symbols: int = 50,
        # Nur Favoriten pre-fetchen
        favorites_only: bool = False,
        # Intervall für periodisches Pre-Fetching (Sekunden)
        refresh_interval: int = 300,
        # Output-Size für OHLCV-Daten (Fallback, wird pro Timeframe überschrieben)
        ohlcv_limit: int = 5000,
        # Verzögerung zwischen API-Aufrufen (Rate Limiting)
        api_delay: float = 0.2,
        # Welche Indikatoren sollen pre-fetched werden (leer = keine)
        indicators: list[str] = None,
        # Output-Size für Indikatoren
        indicator_limit: int = 100,
        # TimescaleDB Sync aktivieren (schreibt Daten persistent in DB)
        db_sync_enabled: bool = True,
        # EasyInsight Indikatoren aktivieren (H1 only)
        easyinsight_indicators_enabled: bool = True,
        # EasyInsight Indikator-Limit
        easyinsight_limit: int = 500,
    ):
        self.enabled = enabled
        self.timeframes = timeframes or ["1h", "4h", "1day", "1week", "1month"]
        self.fast_timeframes = fast_timeframes or ["1min", "5min", "15min", "30min"]
        self.fast_refresh_interval = fast_refresh_interval
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
        self.db_sync_enabled = db_sync_enabled
        self.easyinsight_indicators_enabled = easyinsight_indicators_enabled
        self.easyinsight_limit = easyinsight_limit

    def get_limit_for_timeframe(self, timeframe: str) -> int:
        """Gibt das optimierte Limit für einen Timeframe zurück."""
        return self.TIMEFRAME_LIMITS.get(timeframe, self.ohlcv_limit)


class PrefetchService:
    """
    Service für automatisches Pre-Fetching von Marktdaten.

    Optimiert für nachgelagerte ML-Services:
    - NHITS: Benötigt H1, D1 für Forecasting
    - TCN: Benötigt M15-D1 für Pattern-Erkennung
    - HMM: Benötigt H1, H4, D1 für Regime-Detection
    """

    # Redis-Key für persistente Konfiguration
    CONFIG_CACHE_KEY = "prefetch_config"

    def __init__(self):
        self._config = PrefetchConfig()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._fast_task: Optional[asyncio.Task] = None  # Task für schnelle Timeframes
        self._stats = {
            "last_run": None,
            "last_fast_run": None,
            "symbols_fetched": 0,
            "timeframes_fetched": 0,
            "fast_timeframes_fetched": 0,
            "indicators_fetched": 0,
            "errors": 0,
            "total_runs": 0,
            "total_fast_runs": 0,
            "cache_entries_created": 0,
            "fast_cache_entries_created": 0,
            "indicator_entries_created": 0,
            "easyinsight_indicators_created": 0,
            "pivot_points_calculated": 0,
        }
        self._symbols_cache: list[dict] = []

    async def load_config(self) -> bool:
        """
        Lädt die Konfiguration aus Redis beim Start.

        Returns:
            True wenn Konfiguration geladen wurde, False sonst
        """
        try:
            cached_config = await cache_service.get(
                CacheCategory.METADATA, self.CONFIG_CACHE_KEY
            )
            if cached_config and isinstance(cached_config, dict):
                # Konfiguration anwenden
                for key, value in cached_config.items():
                    if hasattr(self._config, key) and key != "AVAILABLE_INDICATORS":
                        setattr(self._config, key, value)
                logger.info(f"PrefetchService Konfiguration aus Redis geladen: {cached_config}")
                return True
        except Exception as e:
            logger.warning(f"Fehler beim Laden der Pre-Fetch-Konfiguration: {e}")
        return False

    async def save_config(self) -> bool:
        """
        Speichert die aktuelle Konfiguration in Redis.

        Returns:
            True wenn erfolgreich gespeichert, False sonst
        """
        try:
            config_dict = {
                "enabled": self._config.enabled,
                "timeframes": self._config.timeframes,
                "fast_timeframes": self._config.fast_timeframes,
                "fast_refresh_interval": self._config.fast_refresh_interval,
                "max_symbols": self._config.max_symbols,
                "favorites_only": self._config.favorites_only,
                "refresh_interval": self._config.refresh_interval,
                "ohlcv_limit": self._config.ohlcv_limit,
                "api_delay": self._config.api_delay,
                "indicators": self._config.indicators,
                "indicator_limit": self._config.indicator_limit,
                "db_sync_enabled": self._config.db_sync_enabled,
                "easyinsight_indicators_enabled": self._config.easyinsight_indicators_enabled,
                "easyinsight_limit": self._config.easyinsight_limit,
            }
            # Lange TTL für Konfiguration (30 Tage)
            await cache_service.set(
                CacheCategory.METADATA,
                config_dict,
                self.CONFIG_CACHE_KEY,
                ttl=2592000  # 30 Tage
            )
            logger.info("PrefetchService Konfiguration in Redis gespeichert")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Pre-Fetch-Konfiguration: {e}")
            return False

    async def configure(self, **kwargs) -> None:
        """Aktualisiert die Pre-Fetch-Konfiguration und speichert sie."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        logger.info(f"PrefetchService konfiguriert: {kwargs}")
        # Konfiguration persistent speichern
        await self.save_config()

    async def start(self) -> None:
        """Startet den Pre-Fetching Service."""
        if self._running:
            logger.warning("PrefetchService läuft bereits")
            return

        if not self._config.enabled:
            logger.info("PrefetchService ist deaktiviert")
            return

        self._running = True
        indicators_info = f", Indikatoren: {self._config.indicators}" if self._config.indicators else ""
        db_sync_info = " (DB-Sync: aktiviert)" if self._config.db_sync_enabled else " (DB-Sync: deaktiviert)"
        fast_tf_info = f", Fast-TFs: {self._config.fast_timeframes} @ {self._config.fast_refresh_interval}s" if self._config.fast_timeframes else ""
        logger.info(
            f"PrefetchService gestartet - Timeframes: {self._config.timeframes}, "
            f"Intervall: {self._config.refresh_interval}s{fast_tf_info}{indicators_info}{db_sync_info}"
        )

        # Initial Pre-Fetch und periodisches Pre-Fetch als Background-Task
        # Damit der Application Startup nicht blockiert wird
        self._task = asyncio.create_task(self._initial_and_periodic_prefetch())

        # Schneller Task für M1-M30 Timeframes (jede Minute)
        if self._config.fast_timeframes:
            self._fast_task = asyncio.create_task(self._fast_periodic_prefetch())

    async def _initial_and_periodic_prefetch(self) -> None:
        """Initial Pre-Fetch und dann periodisches Pre-Fetching im Hintergrund."""
        try:
            # Initial Pre-Fetch
            await self._run_prefetch()
        except Exception as e:
            logger.error(f"Initial Pre-Fetch fehlgeschlagen: {e}")

        # Periodisches Pre-Fetch
        await self._periodic_prefetch()

    async def stop(self) -> None:
        """Stoppt den Pre-Fetching Service."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._fast_task:
            self._fast_task.cancel()
            try:
                await self._fast_task
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

    async def _fast_periodic_prefetch(self) -> None:
        """Schnelles periodisches Pre-Fetching für M1-M30 Timeframes (jede Minute)."""
        # Initial kurz warten, damit der normale Prefetch zuerst startet
        await asyncio.sleep(5)

        while self._running:
            try:
                if self._running and self._config.fast_timeframes:
                    await self._run_fast_prefetch()
                await asyncio.sleep(self._config.fast_refresh_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fehler im schnellen Pre-Fetch: {e}")
                self._stats["errors"] += 1

    async def _run_fast_prefetch(self) -> None:
        """Führt einen schnellen Pre-Fetch-Durchlauf für M1-M30 durch."""
        start_time = datetime.now(timezone.utc)
        self._stats["total_fast_runs"] += 1

        try:
            # Symbole laden (aus Cache)
            symbols = await self._get_symbols()
            if not symbols:
                return

            # Nach Priorität sortieren und limitieren
            symbols = self._prioritize_symbols(symbols)
            symbols = symbols[:self._config.max_symbols]

            fetched_count = 0
            for symbol_data in symbols:
                symbol = symbol_data.get("symbol") or symbol_data.get("name", "")
                if not symbol:
                    continue

                for timeframe in self._config.fast_timeframes:
                    try:
                        success = await self._prefetch_ohlcv(symbol, timeframe)
                        if success:
                            fetched_count += 1
                        # Kürzeres Rate Limiting für schnelle Timeframes
                        await asyncio.sleep(self._config.api_delay / 2)
                    except Exception as e:
                        logger.error(f"Fast Pre-Fetch Fehler für {symbol}/{timeframe}: {e}")
                        self._stats["errors"] += 1

            self._stats["last_fast_run"] = start_time.isoformat()
            self._stats["fast_timeframes_fetched"] = len(self._config.fast_timeframes)
            self._stats["fast_cache_entries_created"] = fetched_count

            if fetched_count > 0:
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                logger.info(
                    f"Fast Pre-Fetch abgeschlossen: {fetched_count} OHLCV für "
                    f"{self._config.fast_timeframes} in {duration:.1f}s"
                )

        except Exception as e:
            logger.error(f"Fast Pre-Fetch Durchlauf fehlgeschlagen: {e}")
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

            # 6. EasyInsight-Indikatoren pre-fetchen (falls aktiviert)
            easyinsight_count = 0
            if self._config.easyinsight_indicators_enabled and self._config.db_sync_enabled:
                logger.info(f"Pre-Fetch EasyInsight Indikatoren für {len(symbols)} Symbole...")
                for symbol_data in symbols:
                    symbol = symbol_data.get("symbol") or symbol_data.get("name", "")
                    if not symbol:
                        continue

                    try:
                        saved = await self._prefetch_easyinsight_indicators(symbol)
                        easyinsight_count += saved
                        # Rate Limiting (weniger streng, da lokaler Service)
                        await asyncio.sleep(0.1)
                    except Exception as e:
                        logger.debug(f"EasyInsight Prefetch Fehler für {symbol}: {e}")

            # 7. Pivot Points lokal berechnen und speichern (falls DB-Sync aktiviert)
            pivot_count = 0
            if self._config.db_sync_enabled:
                logger.info(f"Berechne Pivot Points für {len(symbols)} Symbole...")
                for symbol_data in symbols:
                    symbol = symbol_data.get("symbol") or symbol_data.get("name", "")
                    if not symbol:
                        continue

                    # Pivot Points für D1 und W1 berechnen (sinnvollste Timeframes)
                    for timeframe in ["1day", "1week"]:
                        try:
                            saved = await self._calculate_and_store_pivots(symbol, timeframe)
                            pivot_count += saved
                        except Exception as e:
                            logger.debug(f"Pivot-Berechnung fehlgeschlagen für {symbol}/{timeframe}: {e}")

            self._stats["last_run"] = start_time.isoformat()
            self._stats["symbols_fetched"] = len(symbols)
            self._stats["timeframes_fetched"] = len(self._config.timeframes)
            self._stats["indicators_fetched"] = len(self._config.indicators)
            self._stats["cache_entries_created"] = fetched_count
            self._stats["indicator_entries_created"] = indicator_count
            self._stats["easyinsight_indicators_created"] = easyinsight_count
            self._stats["pivot_points_calculated"] = pivot_count

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            indicator_info = f", {indicator_count} TwelveData-Indikatoren" if indicator_count else ""
            easyinsight_info = f", {easyinsight_count} EasyInsight-Indikatoren" if easyinsight_count else ""
            pivot_info = f", {pivot_count} Pivot-Points" if pivot_count else ""
            logger.info(
                f"Pre-Fetch abgeschlossen: {fetched_count} OHLCV{indicator_info}{easyinsight_info}{pivot_info} in {duration:.1f}s"
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
        Wenn db_sync_enabled: Nutzt DataGateway (speichert automatisch in TimescaleDB)
        Sonst: Nutzt TwelveData Service direkt (nur Cache).
        """
        # Normalize symbol for cache key
        cache_symbol = symbol.upper().replace("/", "")

        # Dynamisches Limit pro Timeframe für optimale Abdeckung
        limit = self._config.get_limit_for_timeframe(timeframe)
        cache_params = {"interval": timeframe, "outputsize": limit}

        # Prüfe ob Cache noch gültig ist
        cached = await cache_service.get(
            CacheCategory.OHLCV, cache_symbol, timeframe, params=cache_params
        )
        if cached:
            # Bereits im Cache - kein Fetch nötig
            return False

        try:
            if self._config.db_sync_enabled:
                # DataGateway verwenden - speichert automatisch in TimescaleDB
                from .data_gateway_service import data_gateway

                data, source = await data_gateway.get_historical_data_with_fallback(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    force_refresh=False,  # Cache nutzen wenn vorhanden
                )

                if data:
                    logger.debug(
                        f"Pre-fetched {len(data)} rows for {symbol}/{timeframe} "
                        f"from {source} (DB-Sync: enabled)"
                    )
                    return True
            else:
                # Direkt über TwelveData Service abrufen (nur Cache)
                from .twelvedata_service import twelvedata_service

                result = await twelvedata_service.get_time_series(
                    symbol=symbol,
                    interval=timeframe,
                    outputsize=limit,
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

        Wenn db_sync_enabled: Speichert Indikatoren auch in TimescaleDB.

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

                # In TimescaleDB speichern wenn aktiviert
                if self._config.db_sync_enabled:
                    try:
                        from .data_repository import data_repository

                        # Indikator-spezifische Parameter extrahieren
                        params = {}
                        for key in ["time_period", "fast_period", "slow_period", "signal_period", "sd"]:
                            if key in result:
                                params[key] = result[key]

                        saved_count = await data_repository.save_indicators(
                            symbol=cache_symbol,
                            timeframe=timeframe,
                            indicator_name=indicator.lower(),
                            data=values if isinstance(values, list) else [values],
                            parameters=params,
                            source="twelvedata",
                        )
                        if saved_count > 0:
                            logger.debug(
                                f"Saved {saved_count} {indicator.upper()} values to TimescaleDB "
                                f"for {symbol}/{timeframe}"
                            )
                    except Exception as db_err:
                        logger.warning(f"Failed to save {indicator} to TimescaleDB: {db_err}")

                logger.debug(
                    f"Pre-fetched {indicator.upper()} for {symbol}/{timeframe} "
                    f"({len(values) if isinstance(values, list) else 1} values)"
                )
                return True

        except Exception as e:
            logger.warning(f"Pre-Fetch Fehler für {indicator}/{symbol}/{timeframe}: {e}")

        return False

    async def _prefetch_easyinsight_indicators(self, symbol: str) -> int:
        """
        Pre-fetcht EasyInsight-Indikatoren für ein Symbol und speichert sie in TimescaleDB.

        EasyInsight liefert H1-Daten mit vorberechneten Indikatoren wie:
        - rsi, macd_main, macd_signal, cci, sto_main, sto_signal
        - adx_main, adx_plusdi, adx_minusdi, atr_d1
        - bb_base, bb_lower, bb_upper
        - ichimoku_* (alle 5 Linien)
        - ma_10, strength_1d, strength_1w, strength_4h

        Diese werden automatisch in die optimierten Tabellen geroutet:
        - indicators_momentum: rsi, macd, cci, stoch, adx
        - indicators_volatility: atr, bb_*
        - indicators_trend: ichimoku_*
        - indicators_ma: ma_10

        Args:
            symbol: Trading-Symbol (z.B. BTCUSD)

        Returns:
            Anzahl der gespeicherten Indikator-Datensätze
        """
        if not self._config.db_sync_enabled:
            return 0

        cache_symbol = symbol.upper().replace("/", "")

        try:
            from .data_gateway_service import data_gateway
            from .data_repository import data_repository

            # EasyInsight OHLCV mit Indikatoren abrufen über Data Gateway
            data = await data_gateway.get_easyinsight_historical(
                symbol=symbol,
                timeframe="H1",
                limit=self._config.easyinsight_limit
            )

            if not data:
                return 0

            # Indikator-Felder aus den Daten extrahieren
            # EasyInsight liefert diese im gleichen Response wie OHLCV
            indicator_fields = [
                "rsi", "macd_main", "macd_signal", "cci",
                "adx_main", "adx_plusdi", "adx_minusdi",
                "atr_d1", "atr_pct_d1",
                "bb_base", "bb_lower", "bb_upper",
                "sto_main", "sto_signal",
                "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkoua",
                "ichimoku_senkoub", "ichimoku_chikou",
                "ma_10", "strength_1d", "strength_1w", "strength_4h",
            ]

            # Gruppieren der Indikatoren für Batch-Speicherung
            indicators_batch: dict[str, list[dict]] = {}

            for indicator_name in indicator_fields:
                indicator_data = []
                for row in data:
                    if indicator_name in row and row[indicator_name] is not None:
                        timestamp = row.get("snapshot_time") or row.get("timestamp")
                        if timestamp:
                            indicator_data.append({
                                "timestamp": timestamp,
                                indicator_name: row[indicator_name]
                            })

                if indicator_data:
                    indicators_batch[indicator_name] = indicator_data

            if not indicators_batch:
                return 0

            # Batch-Speicherung - automatisch in optimierte Tabellen geroutet
            try:
                results = await data_repository.save_indicators_batch(
                    symbol=cache_symbol,
                    timeframe="H1",
                    indicators_data=indicators_batch,
                    source="easyinsight",
                )

                saved_total = sum(results.values())

                if saved_total > 0:
                    logger.debug(
                        f"Saved {saved_total} EasyInsight indicator values for {symbol}/H1 "
                        f"(batches: {results})"
                    )

                return saved_total

            except Exception as e:
                logger.warning(f"Batch save failed for {symbol}, falling back to individual: {e}")

                # Fallback: Jeden Indikator einzeln speichern
                saved_total = 0
                for indicator_name, indicator_data in indicators_batch.items():
                    try:
                        saved_count = await data_repository.save_indicators(
                            symbol=cache_symbol,
                            timeframe="H1",
                            indicator_name=indicator_name,
                            data=indicator_data,
                            parameters={},
                            source="easyinsight",
                        )
                        saved_total += saved_count
                    except Exception as ind_err:
                        logger.debug(f"Failed to save EasyInsight {indicator_name}: {ind_err}")

                return saved_total

        except Exception as e:
            logger.warning(f"EasyInsight indicator prefetch failed for {symbol}: {e}")
            return 0

    async def _calculate_and_store_pivots(self, symbol: str, timeframe: str) -> int:
        """
        Berechnet Pivot Points lokal aus OHLCV-Daten und speichert sie in TimescaleDB.

        Diese Methode ersetzt den TwelveData pivot_points_hl Indikator, der nur im
        Pro-Plan verfügbar ist. Die Berechnung erfolgt lokal aus bereits gecachten
        OHLCV-Daten.

        Berechnet:
        - Standard (Floor Trader) Pivot Points: P, R1-R3, S1-S3
        - Fibonacci Pivot Points: P, fib_R1-R3, fib_S1-S3
        - Camarilla Pivot Points: cam_R1-R4, cam_S1-S4

        Args:
            symbol: Trading-Symbol (z.B. BTCUSD)
            timeframe: Zeitrahmen (z.B. 1day, 1week)

        Returns:
            Anzahl der gespeicherten Pivot Point Datensätze
        """
        if not self._config.db_sync_enabled:
            return 0

        cache_symbol = symbol.upper().replace("/", "")

        try:
            from .data_gateway_service import data_gateway
            from .pivot_calculator import pivot_calculator
            from .timescaledb_service import timescaledb_service
            from src.config.timeframes import normalize_timeframe

            # OHLCV-Daten abrufen (sollten bereits im Cache sein)
            ohlcv_data, source = await data_gateway.get_historical_data_with_fallback(
                symbol=symbol,
                timeframe=timeframe,
                limit=100,  # Letzte 100 Kerzen für Pivot-Berechnung
                force_refresh=False,
            )

            if not ohlcv_data or len(ohlcv_data) < 2:
                return 0

            # Normalisierter Timeframe für DB
            normalized_tf = normalize_timeframe(timeframe).value

            # Pivot Points berechnen und speichern
            saved_count = await pivot_calculator.calculate_and_store_pivots(
                symbol=cache_symbol,
                timeframe=normalized_tf,
                ohlcv_data=ohlcv_data,
                timescaledb_service=timescaledb_service,
            )

            if saved_count > 0:
                logger.debug(
                    f"Calculated and stored {saved_count} pivot points for {symbol}/{timeframe}"
                )

            return saved_count

        except Exception as e:
            logger.warning(f"Pivot calculation failed for {symbol}/{timeframe}: {e}")
            return 0

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
                "fast_timeframes": self._config.fast_timeframes,
                "fast_refresh_interval": self._config.fast_refresh_interval,
                "max_symbols": self._config.max_symbols,
                "favorites_only": self._config.favorites_only,
                "refresh_interval": self._config.refresh_interval,
                "ohlcv_limit": self._config.ohlcv_limit,
                "indicators": self._config.indicators,
                "indicator_limit": self._config.indicator_limit,
                "db_sync_enabled": self._config.db_sync_enabled,
            },
            "running": self._running,
            "fast_task_running": self._fast_task is not None and not self._fast_task.done() if self._fast_task else False,
        }

    def get_config(self) -> dict:
        """Gibt die aktuelle Konfiguration zurück."""
        return {
            "enabled": self._config.enabled,
            "timeframes": self._config.timeframes,
            "fast_timeframes": self._config.fast_timeframes,
            "fast_refresh_interval": self._config.fast_refresh_interval,
            "max_symbols": self._config.max_symbols,
            "favorites_only": self._config.favorites_only,
            "refresh_interval": self._config.refresh_interval,
            "ohlcv_limit": self._config.ohlcv_limit,
            "timeframe_limits": PrefetchConfig.TIMEFRAME_LIMITS,
            "api_delay": self._config.api_delay,
            "indicators": self._config.indicators,
            "indicator_limit": self._config.indicator_limit,
            "db_sync_enabled": self._config.db_sync_enabled,
            "easyinsight_indicators_enabled": self._config.easyinsight_indicators_enabled,
            "easyinsight_limit": self._config.easyinsight_limit,
            "available_indicators": PrefetchConfig.AVAILABLE_INDICATORS,
        }


# Singleton-Instanz
prefetch_service = PrefetchService()
