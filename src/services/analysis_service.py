"""Analysis Service - Main pipeline for generating trading recommendations."""

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
    SignalType,
)
from .llm_service import LLMService
from .rag_service import RAGService


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
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            query = """
                SELECT
                    data_timestamp,
                    symbol,
                    d1_open,
                    d1_high,
                    d1_low,
                    d1_close,
                    bid,
                    ask
                FROM symbol
                WHERE symbol = $1
                AND data_timestamp >= $2
                AND data_timestamp <= $3
                ORDER BY data_timestamp ASC
            """

            rows = await conn.fetch(query, symbol, start_date, end_date)

            time_series = []
            for row in rows:
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

            logger.info(f"Fetched {len(time_series)} data points for {symbol} from TimescaleDB")
            return time_series

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

        logger.info(f"Starting analysis for {request.symbol} (request_id: {request_id})")

        try:
            # 1. Fetch time series data from TimescaleDB
            time_series = await self._fetch_time_series(
                symbol=request.symbol,
                start_date=datetime.now() - timedelta(days=request.lookback_days),
                end_date=datetime.now()
            )

            if not time_series:
                raise ValueError(f"No data available for symbol {request.symbol}")

            # 2. Calculate technical indicators
            market_analysis = await self._create_market_analysis(
                symbol=request.symbol,
                time_series=time_series
            )

            # 3. Query relevant historical context from RAG
            rag_context = await self._get_rag_context(market_analysis)

            # 4. Generate recommendation using LLM
            recommendation = await self.llm_service.generate_analysis(
                market_data=market_analysis,
                rag_context=rag_context,
                custom_prompt=request.custom_prompt
            )

            # 5. Store analysis in RAG for future reference
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
                f"({processing_time:.2f}ms)"
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
        """Get relevant context from RAG system."""

        # Get similar historical market conditions
        similar_conditions = await self.rag_service.get_similar_market_conditions(
            analysis=analysis,
            n_results=settings.max_context_documents // 2
        )

        # Query for patterns related to current indicators
        query = f"""
{analysis.symbol} mit RSI {analysis.technical_indicators.rsi}
und {analysis.trend} Trend bei {analysis.volatility} Volatilität
"""
        patterns = await self.rag_service.query_relevant_context(
            query=query,
            symbol=analysis.symbol,
            n_results=settings.max_context_documents // 2,
            document_types=["pattern", "analysis"]
        )

        return similar_conditions + patterns

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
