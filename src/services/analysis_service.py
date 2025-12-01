"""Analysis Service - Main pipeline for generating trading recommendations."""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
import asyncpg
from loguru import logger

from ..config import settings
from ..models.trading_data import (
    TimeSeriesData,
    TechnicalIndicators,
    TradingSignal,
    MarketAnalysis,
    TradingRecommendation,
    AnalysisRequest,
    AnalysisResponse,
    TradeDirection,
    SignalType,
    ConfidenceLevel,
    TradingStrategy,
    RiskLevel,
)
from .llm_service import LLMService
from .rag_service import RAGService
from .query_log_service import TimescaleDBDataLog, RAGContextLog


class AnalysisService:
    """Main service for generating trading recommendations."""

    def __init__(self):
        self.llm_service = LLMService()
        self.rag_service = RAGService()
        self._pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create database connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=settings.timescaledb_host,
                port=settings.timescaledb_port,
                database=settings.timescaledb_database,
                user=settings.timescaledb_user,
                password=settings.timescaledb_password,
                min_size=2,
                max_size=10
            )
            logger.info(f"Connected to TimescaleDB at {settings.timescaledb_host}:{settings.timescaledb_port}")
        return self._pool

    async def _fetch_time_series(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> list[TimeSeriesData]:
        """Fetch time series data directly from TimescaleDB."""
        result = await self._fetch_time_series_with_details(symbol, start_date, end_date)
        return result[0]

    async def _fetch_time_series_with_details(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> tuple[list[TimeSeriesData], TimescaleDBDataLog]:
        """
        Fetch time series data with detailed logging information.
        Returns: (time_series, timescaledb_data_log)
        """
        pool = await self._get_pool()

        # SQL Query für Logging
        query = """
            SELECT
                data_timestamp,
                symbol,
                d1_open,
                d1_high,
                d1_low,
                d1_close,
                h1_open,
                h1_high,
                h1_low,
                h1_close,
                m15_open,
                m15_high,
                m15_low,
                m15_close,
                bid,
                ask,
                spread,
                rsi14price_close,
                macd12269price_close_main_line,
                macd12269price_close_signal_line,
                adx14_main_line,
                adx14_plusdi_line,
                adx14_minusdi_line,
                bb200200price_close_base_line,
                bb200200price_close_upper_band,
                bb200200price_close_lower_band,
                atr_d1,
                cci14price_typical,
                sto533mode_smasto_lowhigh_main_line,
                sto533mode_smasto_lowhigh_signal_line,
                ichimoku92652_tenkansen_line,
                ichimoku92652_kijunsen_line,
                ma100mode_smaprice_close,
                strength_4h,
                strength_1d,
                strength_1w
            FROM symbol
            WHERE symbol = $1
            AND data_timestamp >= $2
            AND data_timestamp <= $3
            ORDER BY data_timestamp ASC
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, start_date, end_date)

            time_series = []
            raw_data_sample = []
            indicators_fetched = {}
            ohlc_data = {
                "d1": {"open": None, "high": None, "low": None, "close": None},
                "h1": {"open": None, "high": None, "low": None, "close": None},
                "m15": {"open": None, "high": None, "low": None, "close": None},
            }

            for i, row in enumerate(rows):
                ts = TimeSeriesData(
                    timestamp=row["data_timestamp"],
                    symbol=symbol,
                    open=float(row["d1_open"]) if row["d1_open"] else 0,
                    high=float(row["d1_high"]) if row["d1_high"] else 0,
                    low=float(row["d1_low"]) if row["d1_low"] else 0,
                    close=float(row["d1_close"]) if row["d1_close"] else 0,
                    volume=0,  # Volume not available in this schema
                    additional_data={
                        "bid": float(row["bid"]) if row["bid"] else None,
                        "ask": float(row["ask"]) if row["ask"] else None
                    }
                )
                time_series.append(ts)

                # Sammle Rohdaten-Sample (erste 3 Zeilen)
                if i < 3:
                    sample = {
                        "timestamp": row["data_timestamp"].isoformat() if row["data_timestamp"] else None,
                        "symbol": row["symbol"],
                        "d1_ohlc": {
                            "open": float(row["d1_open"]) if row["d1_open"] else None,
                            "high": float(row["d1_high"]) if row["d1_high"] else None,
                            "low": float(row["d1_low"]) if row["d1_low"] else None,
                            "close": float(row["d1_close"]) if row["d1_close"] else None,
                        },
                        "bid": float(row["bid"]) if row["bid"] else None,
                        "ask": float(row["ask"]) if row["ask"] else None,
                        "spread": float(row["spread"]) if row["spread"] else None,
                    }
                    raw_data_sample.append(sample)

            # Letzte Zeile für aktuelle Indikatoren verwenden
            if rows:
                last_row = rows[-1]

                # OHLC-Daten
                ohlc_data["d1"] = {
                    "open": float(last_row["d1_open"]) if last_row["d1_open"] else None,
                    "high": float(last_row["d1_high"]) if last_row["d1_high"] else None,
                    "low": float(last_row["d1_low"]) if last_row["d1_low"] else None,
                    "close": float(last_row["d1_close"]) if last_row["d1_close"] else None,
                }
                ohlc_data["h1"] = {
                    "open": float(last_row["h1_open"]) if last_row["h1_open"] else None,
                    "high": float(last_row["h1_high"]) if last_row["h1_high"] else None,
                    "low": float(last_row["h1_low"]) if last_row["h1_low"] else None,
                    "close": float(last_row["h1_close"]) if last_row["h1_close"] else None,
                }
                ohlc_data["m15"] = {
                    "open": float(last_row["m15_open"]) if last_row["m15_open"] else None,
                    "high": float(last_row["m15_high"]) if last_row["m15_high"] else None,
                    "low": float(last_row["m15_low"]) if last_row["m15_low"] else None,
                    "close": float(last_row["m15_close"]) if last_row["m15_close"] else None,
                }

                # Technische Indikatoren
                if last_row["rsi14price_close"]:
                    indicators_fetched["rsi14"] = float(last_row["rsi14price_close"])
                if last_row["macd12269price_close_main_line"]:
                    indicators_fetched["macd_main"] = float(last_row["macd12269price_close_main_line"])
                if last_row["macd12269price_close_signal_line"]:
                    indicators_fetched["macd_signal"] = float(last_row["macd12269price_close_signal_line"])
                if last_row["adx14_main_line"]:
                    indicators_fetched["adx"] = float(last_row["adx14_main_line"])
                if last_row["adx14_plusdi_line"]:
                    indicators_fetched["adx_plus_di"] = float(last_row["adx14_plusdi_line"])
                if last_row["adx14_minusdi_line"]:
                    indicators_fetched["adx_minus_di"] = float(last_row["adx14_minusdi_line"])
                if last_row["bb200200price_close_base_line"]:
                    indicators_fetched["bb_middle"] = float(last_row["bb200200price_close_base_line"])
                if last_row["bb200200price_close_upper_band"]:
                    indicators_fetched["bb_upper"] = float(last_row["bb200200price_close_upper_band"])
                if last_row["bb200200price_close_lower_band"]:
                    indicators_fetched["bb_lower"] = float(last_row["bb200200price_close_lower_band"])
                if last_row["atr_d1"]:
                    indicators_fetched["atr_d1"] = float(last_row["atr_d1"])
                if last_row["cci14price_typical"]:
                    indicators_fetched["cci14"] = float(last_row["cci14price_typical"])
                if last_row["sto533mode_smasto_lowhigh_main_line"]:
                    indicators_fetched["stoch_k"] = float(last_row["sto533mode_smasto_lowhigh_main_line"])
                if last_row["sto533mode_smasto_lowhigh_signal_line"]:
                    indicators_fetched["stoch_d"] = float(last_row["sto533mode_smasto_lowhigh_signal_line"])
                if last_row["ichimoku92652_tenkansen_line"]:
                    indicators_fetched["ichimoku_tenkan"] = float(last_row["ichimoku92652_tenkansen_line"])
                if last_row["ichimoku92652_kijunsen_line"]:
                    indicators_fetched["ichimoku_kijun"] = float(last_row["ichimoku92652_kijunsen_line"])
                if last_row["ma100mode_smaprice_close"]:
                    indicators_fetched["ma100"] = float(last_row["ma100mode_smaprice_close"])
                if last_row["strength_4h"]:
                    indicators_fetched["strength_4h"] = float(last_row["strength_4h"])
                if last_row["strength_1d"]:
                    indicators_fetched["strength_1d"] = float(last_row["strength_1d"])
                if last_row["strength_1w"]:
                    indicators_fetched["strength_1w"] = float(last_row["strength_1w"])

            # Erstelle TimescaleDB Log
            tsdb_log = TimescaleDBDataLog(
                query_timestamp=datetime.now(),
                tables_queried=["symbol"],
                symbols_queried=[symbol],
                time_range_start=start_date,
                time_range_end=end_date,
                rows_fetched=len(rows),
                ohlc_data=ohlc_data,
                indicators_fetched=indicators_fetched,
                raw_data_sample=raw_data_sample,
                sql_queries=[query.strip()],
            )

            logger.info(
                f"TimescaleDB: Fetched {len(time_series)} rows for {symbol}, "
                f"{len(indicators_fetched)} indicators available"
            )
            return time_series, tsdb_log

    async def get_available_symbols(self) -> list[str]:
        """Get list of available trading symbols from TimescaleDB."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT symbol FROM symbol WHERE symbol IS NOT NULL ORDER BY symbol LIMIT 100"
            )
            return [row["symbol"] for row in rows if row["symbol"]]

    async def check_timescaledb_connection(self) -> bool:
        """Check if TimescaleDB is accessible."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.warning(f"TimescaleDB connection check failed: {e}")
            return False

    async def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Generate a complete trading analysis and recommendation.

        Args:
            request: Analysis request parameters

        Returns:
            Complete analysis response with recommendation
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Load strategy - either specified or default
        strategy = None
        from .strategy_service import StrategyService
        strategy_service = StrategyService()

        if request.strategy_id:
            strategy = await strategy_service.get_strategy(request.strategy_id)
            if not strategy:
                # Fallback to default strategy if specified strategy not found
                strategy = await strategy_service.get_default_strategy()
        else:
            # No strategy specified - use default strategy
            strategy = await strategy_service.get_default_strategy()

        strategy_name = strategy.name if strategy else "Standard"
        logger.info(f"Starting analysis for {request.symbol} (request_id: {request_id}, strategy: {strategy_name})")

        try:
            # Use strategy's lookback_days if available
            effective_lookback = strategy.lookback_days if strategy else request.lookback_days

            # 1. Fetch time series data from TimescaleDB MIT DETAILLIERTER PROTOKOLLIERUNG
            time_series, tsdb_log = await self._fetch_time_series_with_details(
                symbol=request.symbol,
                start_date=datetime.now() - timedelta(days=effective_lookback),
                end_date=datetime.now()
            )

            if not time_series:
                raise ValueError(f"No data available for symbol {request.symbol}")

            # 2. Calculate technical indicators
            market_analysis = await self._create_market_analysis(
                symbol=request.symbol,
                time_series=time_series
            )

            # 3. Query relevant historical context from RAG MIT DETAILLIERTER PROTOKOLLIERUNG
            rag_context = []
            rag_log = None
            if strategy is None or strategy.use_rag_context:
                rag_context, rag_log = await self._get_rag_context_with_details(market_analysis)

            # 4. Build custom prompt from strategy
            custom_prompt = request.custom_prompt
            if strategy and not custom_prompt:
                from .strategy_service import StrategyService
                strategy_service = StrategyService()
                custom_prompt = strategy_service.get_strategy_prompt(strategy)

            # 5. Generate recommendation using LLM MIT DETAILLIERTER PROTOKOLLIERUNG
            recommendation = await self.llm_service.generate_analysis(
                market_data=market_analysis,
                rag_context=rag_context,
                custom_prompt=custom_prompt,
                strategy_id=strategy.id if strategy else None,
                strategy_name=strategy.name if strategy else "Standard",
                # Übergebe detaillierte Protokollierungsdaten
                timescaledb_data=tsdb_log,
                rag_context_details=rag_log,
            )

            # Add strategy name to reasoning if used
            if strategy and recommendation.reasoning:
                recommendation.reasoning = f"[{strategy.name}] {recommendation.reasoning}"

            # 6. Store analysis in RAG for future reference
            await self.rag_service.add_analysis(market_analysis, recommendation)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            response = AnalysisResponse(
                request_id=request_id,
                symbol=request.symbol,
                timestamp=datetime.utcnow(),
                analysis=market_analysis,
                recommendation=recommendation,
                rag_context=rag_context[:3],  # Include top 3 contexts in response
                model_used=settings.ollama_model,
                processing_time_ms=processing_time
            )

            logger.info(
                f"Analysis complete for {request.symbol}: {recommendation.signal} "
                f"({processing_time:.2f}ms, strategy: {strategy_name})"
            )

            return response

        except Exception as e:
            logger.error(f"Analysis failed for {request.symbol}: {e}")
            raise

    async def _create_market_analysis(
        self,
        symbol: str,
        time_series: list[TimeSeriesData]
    ) -> MarketAnalysis:
        """Create market analysis from time series data."""

        # Convert to pandas DataFrame for technical analysis
        df = pd.DataFrame([
            {
                "timestamp": ts.timestamp,
                "open": ts.open,
                "high": ts.high,
                "low": ts.low,
                "close": ts.close,
                "volume": ts.volume
            }
            for ts in time_series
        ])

        df = df.sort_values("timestamp").reset_index(drop=True)

        # Calculate technical indicators
        indicators = self._calculate_indicators(df)

        # Generate trading signals
        signals = self._generate_signals(df, indicators)

        # Calculate price changes
        current_price = df["close"].iloc[-1]
        price_24h_ago = df["close"].iloc[-2] if len(df) > 1 else current_price
        price_7d_ago = df["close"].iloc[-7] if len(df) > 7 else current_price

        price_change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
        price_change_7d = ((current_price - price_7d_ago) / price_7d_ago) * 100

        # Determine trend
        trend_direction = self._determine_trend(df, indicators)

        # Calculate volatility level
        volatility_level = self._calculate_volatility_level(df)

        # Find support and resistance levels
        support_levels, resistance_levels = self._find_support_resistance(df)

        return MarketAnalysis(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            current_price=float(current_price),
            price_change_24h=round(price_change_24h, 2),
            price_change_7d=round(price_change_7d, 2),
            technical_indicators=indicators,
            signals=signals,
            trend=trend_direction,
            volatility=volatility_level,
            support_levels=support_levels,
            resistance_levels=resistance_levels
        )

    def _calculate_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Calculate all technical indicators."""

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume_data = df["volume"]

        # Moving Averages (SMA)
        sma_20 = close.rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
        sma_50 = close.rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
        sma_200 = close.rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None

        # Exponential Moving Averages (EMA)
        ema_12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = close.ewm(span=26, adjust=False).mean().iloc[-1]

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        # MACD
        ema_12_series = close.ewm(span=12, adjust=False).mean()
        ema_26_series = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12_series - ema_26_series
        macd_signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_value = macd_line.iloc[-1]
        macd_signal = macd_signal_line.iloc[-1]
        macd_histogram = (macd_line - macd_signal_line).iloc[-1]

        # Bollinger Bands
        sma_20_series = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        bollinger_upper = (sma_20_series + (std_20 * 2)).iloc[-1]
        bollinger_middle = sma_20_series.iloc[-1]
        bollinger_lower = (sma_20_series - (std_20 * 2)).iloc[-1]

        # ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]

        # OBV (On-Balance Volume)
        obv_values = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv_values.append(obv_values[-1] + volume_data.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv_values.append(obv_values[-1] - volume_data.iloc[i])
            else:
                obv_values.append(obv_values[-1])
        obv = obv_values[-1]

        return TechnicalIndicators(
            sma_20=round(sma_20, 4) if sma_20 and not pd.isna(sma_20) else None,
            sma_50=round(sma_50, 4) if sma_50 and not pd.isna(sma_50) else None,
            sma_200=round(sma_200, 4) if sma_200 and not pd.isna(sma_200) else None,
            ema_12=round(ema_12, 4) if not pd.isna(ema_12) else None,
            ema_26=round(ema_26, 4) if not pd.isna(ema_26) else None,
            rsi=round(rsi, 2) if not pd.isna(rsi) else None,
            macd=round(macd_value, 4) if not pd.isna(macd_value) else None,
            macd_signal=round(macd_signal, 4) if not pd.isna(macd_signal) else None,
            macd_histogram=round(macd_histogram, 4) if not pd.isna(macd_histogram) else None,
            bollinger_upper=round(bollinger_upper, 4) if not pd.isna(bollinger_upper) else None,
            bollinger_middle=round(bollinger_middle, 4) if not pd.isna(bollinger_middle) else None,
            bollinger_lower=round(bollinger_lower, 4) if not pd.isna(bollinger_lower) else None,
            atr=round(atr, 4) if not pd.isna(atr) else None,
            obv=round(obv, 2) if not pd.isna(obv) else None
        )

    def _generate_signals(
        self,
        df: pd.DataFrame,
        indicators: TechnicalIndicators
    ) -> list[TradingSignal]:
        """Generate trading signals from indicators."""
        signals = []
        current_price = df["close"].iloc[-1]

        # RSI Signals
        if indicators.rsi:
            if indicators.rsi < 30:
                signals.append(TradingSignal(
                    signal_type=SignalType.BUY,
                    indicator="RSI",
                    value=indicators.rsi,
                    description="RSI unter 30 - Überverkauft"
                ))
            elif indicators.rsi > 70:
                signals.append(TradingSignal(
                    signal_type=SignalType.SELL,
                    indicator="RSI",
                    value=indicators.rsi,
                    description="RSI über 70 - Überkauft"
                ))

        # MACD Signals
        if indicators.macd and indicators.macd_signal:
            if indicators.macd > indicators.macd_signal:
                signals.append(TradingSignal(
                    signal_type=SignalType.BUY,
                    indicator="MACD",
                    value=indicators.macd,
                    description="MACD über Signal-Linie - Bullish Crossover"
                ))
            else:
                signals.append(TradingSignal(
                    signal_type=SignalType.SELL,
                    indicator="MACD",
                    value=indicators.macd,
                    description="MACD unter Signal-Linie - Bearish Crossover"
                ))

        # Bollinger Band Signals
        if indicators.bollinger_lower and indicators.bollinger_upper:
            if current_price <= indicators.bollinger_lower:
                signals.append(TradingSignal(
                    signal_type=SignalType.BUY,
                    indicator="Bollinger Bands",
                    value=current_price,
                    description="Preis am unteren Bollinger Band"
                ))
            elif current_price >= indicators.bollinger_upper:
                signals.append(TradingSignal(
                    signal_type=SignalType.SELL,
                    indicator="Bollinger Bands",
                    value=current_price,
                    description="Preis am oberen Bollinger Band"
                ))

        # SMA Crossover Signals
        if indicators.sma_20 and indicators.sma_50:
            if indicators.sma_20 > indicators.sma_50:
                signals.append(TradingSignal(
                    signal_type=SignalType.BUY,
                    indicator="SMA Crossover",
                    value=indicators.sma_20,
                    description="SMA 20 über SMA 50 - Golden Cross"
                ))
            else:
                signals.append(TradingSignal(
                    signal_type=SignalType.SELL,
                    indicator="SMA Crossover",
                    value=indicators.sma_20,
                    description="SMA 20 unter SMA 50 - Death Cross"
                ))

        # Price vs SMA 200
        if indicators.sma_200:
            if current_price > indicators.sma_200:
                signals.append(TradingSignal(
                    signal_type=SignalType.BUY,
                    indicator="SMA 200",
                    value=current_price,
                    description="Preis über SMA 200 - Langfristiger Aufwärtstrend"
                ))
            else:
                signals.append(TradingSignal(
                    signal_type=SignalType.SELL,
                    indicator="SMA 200",
                    value=current_price,
                    description="Preis unter SMA 200 - Langfristiger Abwärtstrend"
                ))

        return signals

    def _determine_trend(
        self,
        df: pd.DataFrame,
        indicators: TechnicalIndicators
    ) -> str:
        """Determine overall market trend."""
        current_price = df["close"].iloc[-1]

        bullish_signals = 0
        bearish_signals = 0

        # Check SMAs
        if indicators.sma_20 and current_price > indicators.sma_20:
            bullish_signals += 1
        elif indicators.sma_20:
            bearish_signals += 1

        if indicators.sma_50 and current_price > indicators.sma_50:
            bullish_signals += 1
        elif indicators.sma_50:
            bearish_signals += 1

        if indicators.sma_200 and current_price > indicators.sma_200:
            bullish_signals += 2  # More weight for 200 SMA
        elif indicators.sma_200:
            bearish_signals += 2

        # Check MACD
        if indicators.macd and indicators.macd_signal:
            if indicators.macd > indicators.macd_signal:
                bullish_signals += 1
            else:
                bearish_signals += 1

        # Determine trend
        if bullish_signals > bearish_signals + 2:
            return "strong_uptrend"
        elif bullish_signals > bearish_signals:
            return "uptrend"
        elif bearish_signals > bullish_signals + 2:
            return "strong_downtrend"
        elif bearish_signals > bullish_signals:
            return "downtrend"
        else:
            return "sideways"

    def _calculate_volatility_level(self, df: pd.DataFrame) -> str:
        """Calculate volatility level."""
        returns = df["close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        if volatility < 0.15:
            return "low"
        elif volatility < 0.30:
            return "medium"
        elif volatility < 0.50:
            return "high"
        else:
            return "very_high"

    def _find_support_resistance(
        self,
        df: pd.DataFrame,
        num_levels: int = 3
    ) -> tuple[list[float], list[float]]:
        """Find support and resistance levels."""
        highs = df["high"].values
        lows = df["low"].values
        current_price = df["close"].iloc[-1]

        # Simple pivot point based levels
        pivot = (df["high"].iloc[-1] + df["low"].iloc[-1] + df["close"].iloc[-1]) / 3

        # Calculate support and resistance
        r1 = 2 * pivot - df["low"].iloc[-1]
        r2 = pivot + (df["high"].iloc[-1] - df["low"].iloc[-1])
        s1 = 2 * pivot - df["high"].iloc[-1]
        s2 = pivot - (df["high"].iloc[-1] - df["low"].iloc[-1])

        # Find local highs and lows
        window = 5
        resistance_levels = []
        support_levels = []

        for i in range(window, len(df) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                if highs[i] > current_price:
                    resistance_levels.append(highs[i])

            if lows[i] == min(lows[i-window:i+window+1]):
                if lows[i] < current_price:
                    support_levels.append(lows[i])

        # Combine with pivot levels
        resistance_levels.extend([r1, r2])
        support_levels.extend([s1, s2])

        # Sort and get unique levels
        resistance_levels = sorted(set([round(r, 4) for r in resistance_levels if r > current_price]))[:num_levels]
        support_levels = sorted(set([round(s, 4) for s in support_levels if s < current_price]), reverse=True)[:num_levels]

        return support_levels, resistance_levels

    async def _get_rag_context(self, analysis: MarketAnalysis) -> list[str]:
        """Get relevant context from RAG system (parallelized)."""
        result = await self._get_rag_context_with_details(analysis)
        return result[0]

    async def _get_rag_context_with_details(
        self, analysis: MarketAnalysis
    ) -> tuple[list[str], RAGContextLog]:
        """
        Get relevant context from RAG system with detailed logging.
        Returns: (documents, rag_context_log)
        """

        # Build query for patterns
        query = f"""
{analysis.symbol} mit RSI {analysis.technical_indicators.rsi}
und {analysis.trend} Trend bei {analysis.volatility} Volatilität
"""

        # Run both RAG queries in parallel for better performance
        # Note: document_types filter includes "market_data" which is the primary type in the database
        similar_task = self.rag_service.query_relevant_context_with_details(
            query=f"Symbol: {analysis.symbol} Preis: {analysis.current_price} Trend: {analysis.trend}",
            symbol=analysis.symbol,
            n_results=settings.max_context_documents // 2,
            document_types=["analysis", "market_data"]  # Include market_data for historical context
        )

        patterns_task = self.rag_service.query_relevant_context_with_details(
            query=query,
            symbol=analysis.symbol,
            n_results=settings.max_context_documents // 2,
            document_types=["pattern", "market_data"]
        )

        (similar_docs, similar_details), (pattern_docs, pattern_details) = await asyncio.gather(
            similar_task, patterns_task
        )

        # Kombiniere Dokumente
        all_docs = similar_docs + pattern_docs

        # Kombiniere Details für Logging
        all_document_details = similar_details.get("document_details", []) + pattern_details.get("document_details", [])

        rag_log = RAGContextLog(
            query_text=query.strip(),
            documents_retrieved=similar_details.get("documents_retrieved", 0) + pattern_details.get("documents_retrieved", 0),
            documents_used=len(all_docs),
            document_details=all_document_details,
            filter_symbol=analysis.symbol,
            filter_document_types=["analysis", "pattern", "market_data"],
            embedding_model=similar_details.get("embedding_model", ""),
            embedding_dimension=similar_details.get("embedding_dimension", 0),
            search_k=similar_details.get("search_k", 0) + pattern_details.get("search_k", 0),
            embedding_time_ms=similar_details.get("embedding_time_ms", 0) + pattern_details.get("embedding_time_ms", 0),
            search_time_ms=similar_details.get("search_time_ms", 0) + pattern_details.get("search_time_ms", 0),
        )

        return all_docs, rag_log

    async def quick_recommendation(
        self,
        symbol: str,
        lookback_days: int = 30,
        use_llm: bool = False,
        strategy: Optional[TradingStrategy] = None
    ) -> TradingRecommendation:
        """
        Generate a fast trading recommendation based on technical indicators only.

        This is much faster than full analysis as it skips LLM inference by default.

        Args:
            symbol: Trading symbol
            lookback_days: Days of historical data
            use_llm: If True, use LLM for analysis (slower but more detailed)
            strategy: Optional trading strategy to use

        Returns:
            TradingRecommendation
        """
        start_time = time.time()
        strategy_name = strategy.name if strategy else "Standard"
        logger.info(f"Quick recommendation for {symbol} (use_llm={use_llm}, strategy={strategy_name})")

        # Use strategy's lookback_days if available
        effective_lookback = strategy.lookback_days if strategy else lookback_days

        # Fetch time series data MIT DETAILLIERTER PROTOKOLLIERUNG
        time_series, tsdb_log = await self._fetch_time_series_with_details(
            symbol=symbol,
            start_date=datetime.now() - timedelta(days=effective_lookback),
            end_date=datetime.now()
        )

        if not time_series:
            raise ValueError(f"No data available for symbol {symbol}")

        # Calculate technical indicators
        market_analysis = await self._create_market_analysis(
            symbol=symbol,
            time_series=time_series
        )

        if use_llm:
            # Use LLM for detailed analysis (slower) MIT DETAILLIERTER PROTOKOLLIERUNG
            rag_context = []
            rag_log = None
            if strategy is None or strategy.use_rag_context:
                rag_context, rag_log = await self._get_rag_context_with_details(market_analysis)

            # Build custom prompt from strategy
            custom_prompt = None
            if strategy:
                from .strategy_service import StrategyService
                temp_service = StrategyService()
                custom_prompt = temp_service.get_strategy_prompt(strategy)

            recommendation = await self.llm_service.generate_analysis(
                market_data=market_analysis,
                rag_context=rag_context,
                custom_prompt=custom_prompt,
                strategy_id=strategy.id if strategy else None,
                strategy_name=strategy.name if strategy else "Standard",
                # Detaillierte Protokollierungsdaten
                timescaledb_data=tsdb_log,
                rag_context_details=rag_log,
            )
        else:
            # Fast rule-based recommendation (no LLM)
            recommendation = self._generate_rule_based_recommendation(market_analysis, strategy)

        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Quick recommendation for {symbol}: {recommendation.signal} ({processing_time:.0f}ms)")

        return recommendation

    def _generate_rule_based_recommendation(
        self,
        analysis: MarketAnalysis,
        strategy: Optional[TradingStrategy] = None
    ) -> TradingRecommendation:
        """Generate a recommendation based on technical indicators without LLM."""

        buy_signals = 0
        sell_signals = 0
        key_factors = []
        risks = []

        indicators = analysis.technical_indicators
        current_price = analysis.current_price

        # Get strategy-specific thresholds
        rsi_buy_threshold = 30
        rsi_sell_threshold = 70
        min_buy_signals = 3
        min_sell_signals = 3
        stop_loss_multiplier = 2.0
        take_profit_multiplier = 3.0
        timeframe = "short_term"

        # Extract strategy-specific settings
        if strategy:
            min_buy_signals = strategy.min_buy_signals
            min_sell_signals = strategy.min_sell_signals
            stop_loss_multiplier = strategy.stop_loss_atr_multiplier
            take_profit_multiplier = strategy.take_profit_atr_multiplier
            timeframe = strategy.preferred_timeframe

            # Get RSI thresholds from strategy indicators
            for ind in strategy.indicators:
                if ind.name == "RSI" and ind.enabled:
                    if ind.buy_threshold:
                        rsi_buy_threshold = ind.buy_threshold
                    if ind.sell_threshold:
                        rsi_sell_threshold = ind.sell_threshold

        # RSI Analysis
        if indicators.rsi:
            if indicators.rsi < rsi_buy_threshold:
                buy_signals += 2
                key_factors.append(f"RSI überverkauft ({indicators.rsi:.1f})")
            elif indicators.rsi < rsi_buy_threshold + 10:
                buy_signals += 1
                key_factors.append(f"RSI niedrig ({indicators.rsi:.1f})")
            elif indicators.rsi > rsi_sell_threshold:
                sell_signals += 2
                key_factors.append(f"RSI überkauft ({indicators.rsi:.1f})")
            elif indicators.rsi > rsi_sell_threshold - 10:
                sell_signals += 1
                key_factors.append(f"RSI hoch ({indicators.rsi:.1f})")

        # MACD Analysis
        if indicators.macd and indicators.macd_signal:
            if indicators.macd > indicators.macd_signal:
                buy_signals += 1
                if indicators.macd_histogram and indicators.macd_histogram > 0:
                    buy_signals += 1
                    key_factors.append("MACD bullish mit positivem Histogram")
            else:
                sell_signals += 1
                if indicators.macd_histogram and indicators.macd_histogram < 0:
                    sell_signals += 1
                    key_factors.append("MACD bearish mit negativem Histogram")

        # Moving Average Analysis
        if indicators.sma_20 and indicators.sma_50:
            if indicators.sma_20 > indicators.sma_50:
                buy_signals += 1
                key_factors.append("Golden Cross (SMA20 > SMA50)")
            else:
                sell_signals += 1
                key_factors.append("Death Cross (SMA20 < SMA50)")

        if indicators.sma_200:
            if current_price > indicators.sma_200:
                buy_signals += 1
                key_factors.append("Preis über SMA200 (Aufwärtstrend)")
            else:
                sell_signals += 1
                risks.append("Preis unter SMA200 (Abwärtstrend)")

        # Bollinger Band Analysis
        if indicators.bollinger_lower and indicators.bollinger_upper:
            bb_position = (current_price - indicators.bollinger_lower) / (indicators.bollinger_upper - indicators.bollinger_lower)
            if bb_position < 0.2:
                buy_signals += 1
                key_factors.append("Preis nahe unterem Bollinger Band")
            elif bb_position > 0.8:
                sell_signals += 1
                key_factors.append("Preis nahe oberem Bollinger Band")

        # Trend Analysis
        if analysis.trend in ["strong_uptrend", "uptrend"]:
            buy_signals += 1
        elif analysis.trend in ["strong_downtrend", "downtrend"]:
            sell_signals += 1

        # Volatility Risk
        if analysis.volatility in ["high", "very_high"]:
            risks.append(f"Hohe Volatilität ({analysis.volatility})")

        # Determine signal and confidence based on strategy thresholds
        signal_diff = buy_signals - sell_signals

        # Adjust thresholds based on risk level
        high_threshold = min_buy_signals + 1
        medium_threshold = min_buy_signals - 1

        if strategy and strategy.risk_level == RiskLevel.CONSERVATIVE:
            high_threshold += 1
            medium_threshold += 1
        elif strategy and strategy.risk_level == RiskLevel.AGGRESSIVE:
            high_threshold -= 1
            medium_threshold = max(1, medium_threshold - 1)

        # Determine direction, signal and confidence
        if signal_diff >= high_threshold:
            direction = TradeDirection.LONG
            signal = SignalType.BUY
            confidence = ConfidenceLevel.HIGH
            confidence_score = 85
        elif signal_diff >= medium_threshold:
            direction = TradeDirection.LONG
            signal = SignalType.BUY
            confidence = ConfidenceLevel.MEDIUM
            confidence_score = 65
        elif signal_diff >= 1:
            direction = TradeDirection.LONG
            signal = SignalType.BUY
            confidence = ConfidenceLevel.LOW
            confidence_score = 45
        elif signal_diff <= -high_threshold:
            direction = TradeDirection.SHORT
            signal = SignalType.SELL
            confidence = ConfidenceLevel.HIGH
            confidence_score = 85
        elif signal_diff <= -medium_threshold:
            direction = TradeDirection.SHORT
            signal = SignalType.SELL
            confidence = ConfidenceLevel.MEDIUM
            confidence_score = 65
        elif signal_diff <= -1:
            direction = TradeDirection.SHORT
            signal = SignalType.SELL
            confidence = ConfidenceLevel.LOW
            confidence_score = 45
        else:
            direction = TradeDirection.NEUTRAL
            signal = SignalType.HOLD
            confidence = ConfidenceLevel.MEDIUM
            confidence_score = 50

        # Calculate entry, stop-loss, take-profit using strategy multipliers
        entry_price = current_price
        atr = indicators.atr if indicators.atr else current_price * 0.02

        # Calculate price levels based on direction
        if direction == TradeDirection.LONG:
            stop_loss = current_price - (stop_loss_multiplier * atr)
            take_profit_1 = current_price + (take_profit_multiplier * atr * 0.5)
            take_profit_2 = current_price + (take_profit_multiplier * atr)
            take_profit_3 = current_price + (take_profit_multiplier * atr * 1.5)
            setup_recommendation = f"Bullisches Setup: {buy_signals} Kaufsignale vs {sell_signals} Verkaufssignale identifiziert"
        elif direction == TradeDirection.SHORT:
            stop_loss = current_price + (stop_loss_multiplier * atr)
            take_profit_1 = current_price - (take_profit_multiplier * atr * 0.5)
            take_profit_2 = current_price - (take_profit_multiplier * atr)
            take_profit_3 = current_price - (take_profit_multiplier * atr * 1.5)
            setup_recommendation = f"Bärisches Setup: {sell_signals} Verkaufssignale vs {buy_signals} Kaufsignale identifiziert"
        else:
            stop_loss = None
            take_profit_1 = None
            take_profit_2 = None
            take_profit_3 = None
            setup_recommendation = "Keine klare Richtung - abwarten empfohlen"

        # Calculate risk-reward ratio
        risk_reward_ratio = None
        if stop_loss and take_profit_2:
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit_2 - current_price)
            if risk > 0:
                risk_reward_ratio = round(reward / risk, 2)

        # Add strategy name to setup recommendation if used
        if strategy:
            setup_recommendation = f"[{strategy.name}] {setup_recommendation}"

        # Add default factors if none found
        if not key_factors:
            key_factors = ["Keine starken Signale identifiziert"]
        if not risks:
            risks = ["Marktrisiko beachten"]

        # Add risk level warning if aggressive
        if strategy and strategy.risk_level == RiskLevel.AGGRESSIVE:
            risks.insert(0, "Aggressive Strategie - erhöhtes Risiko")

        # Build trend analysis
        trend_analysis = f"Trend: {analysis.trend}. "
        if indicators.sma_20 and indicators.sma_50:
            if indicators.sma_20 > indicators.sma_50:
                trend_analysis += "SMA20 über SMA50 (bullish). "
            else:
                trend_analysis += "SMA20 unter SMA50 (bearish). "
        if indicators.rsi:
            trend_analysis += f"RSI bei {indicators.rsi:.1f}."

        # Build support/resistance string
        support_resistance = ""
        if analysis.support_levels:
            support_resistance += f"Support: {', '.join([str(round(s, 5)) for s in analysis.support_levels[:3]])}. "
        if analysis.resistance_levels:
            support_resistance += f"Resistance: {', '.join([str(round(r, 5)) for r in analysis.resistance_levels[:3]])}."

        # Build key levels string
        key_levels_str = ""
        if indicators.bollinger_upper and indicators.bollinger_lower:
            key_levels_str = f"BB Upper: {indicators.bollinger_upper:.5f}, BB Lower: {indicators.bollinger_lower:.5f}"
        if indicators.sma_200:
            key_levels_str += f", SMA200: {indicators.sma_200:.5f}"

        # Build risk factors string
        risk_factors_str = "; ".join(risks[:3]) if risks else "Standardmäßiges Marktrisiko"

        # Build trade rationale
        trade_rationale = f"Basierend auf {len(key_factors)} identifizierten Faktoren: " + "; ".join(key_factors[:3])

        return TradingRecommendation(
            symbol=analysis.symbol,
            timestamp=datetime.utcnow(),
            # New schema fields
            direction=direction,
            confidence_score=confidence_score,
            setup_recommendation=setup_recommendation,
            entry_price=round(entry_price, 5) if entry_price else None,
            stop_loss=round(stop_loss, 5) if stop_loss else None,
            take_profit_1=round(take_profit_1, 5) if take_profit_1 else None,
            take_profit_2=round(take_profit_2, 5) if take_profit_2 else None,
            take_profit_3=round(take_profit_3, 5) if take_profit_3 else None,
            risk_reward_ratio=risk_reward_ratio,
            recommended_position_size=0.01,  # Default micro lot
            max_risk_percent=1.0,  # Default 1% risk
            trend_analysis=trend_analysis.strip(),
            support_resistance=support_resistance.strip(),
            key_levels=key_levels_str.strip(", "),
            risk_factors=risk_factors_str,
            trade_rationale=trade_rationale,
            # Legacy fields for backward compatibility
            signal=signal,
            confidence=confidence,
            take_profit=round(take_profit_1, 5) if take_profit_1 else None,
            reasoning=setup_recommendation,
            key_factors=key_factors[:5],
            risks=risks[:3],
            timeframe=timeframe
        )

    async def health_check(self) -> dict:
        """Check health of all services."""
        results = {}

        # Check LLM service
        results["llm_service"] = await self.llm_service.check_model_available()

        # Check RAG service
        try:
            stats = await self.rag_service.get_collection_stats()
            results["rag_service"] = True
            results["rag_documents"] = stats["document_count"]
        except Exception:
            results["rag_service"] = False

        results["all_healthy"] = all([
            results.get("llm_service", False),
            results.get("rag_service", False)
        ])

        return results
