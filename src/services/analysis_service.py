"""Analysis Service - Main pipeline for generating trading recommendations.

WICHTIG: Dieser Service verwendet den DataGatewayService für alle externen
Datenzugriffe. Direkte API-Aufrufe zu EasyInsight oder anderen externen
Datenquellen sind NICHT erlaubt.

Siehe: DEVELOPMENT_GUIDELINES.md - Datenzugriff-Architektur
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
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
    NHITSForecast,
    # New market data models
    MarketDataSnapshot,
    OHLCData,
    PriceData,
    RSIIndicator,
    MACDIndicator,
    StochasticIndicator,
    CCIIndicator,
    ADXIndicator,
    MAIndicator,
    IchimokuIndicator,
    BollingerBandsIndicator,
    ATRIndicator,
    RangeIndicator,
    PivotPoints,
    StrengthIndicators,
    AllIndicators,
)
from .llm_service import LLMService
from .rag_service import RAGService
from .query_log_service import TimescaleDBDataLog, RAGContextLog
from .data_gateway_service import data_gateway
from ..config.settings import settings as app_settings


class AnalysisService:
    """Main service for generating trading recommendations."""

    def __init__(self):
        self.llm_service = LLMService()
        self.rag_service = RAGService()
        self.data_gateway = data_gateway

    async def fetch_latest_market_data(self, symbol: str) -> Optional[MarketDataSnapshot]:
        """
        Fetch the latest market data snapshot via Data Gateway.

        This includes all OHLC data, technical indicators, and market conditions
        for comprehensive LLM analysis.

        Verwendet: DataGatewayService (siehe DEVELOPMENT_GUIDELINES.md)
        """
        try:
            # Verwende Data Gateway anstelle von direktem API-Zugriff
            data = await self.data_gateway.get_latest_market_data(symbol)

            if not data:
                logger.warning(f"No data returned from Data Gateway for {symbol}")
                return None

            # Parse snapshot_time
            snapshot_time = datetime.fromisoformat(
                data['snapshot_time'].replace('Z', '+00:00')
            )

            # Create comprehensive market data snapshot
            snapshot = MarketDataSnapshot(
                symbol=symbol,
                snapshot_time=snapshot_time,
                category=data.get('category', 'Unknown'),
                price=PriceData(
                    bid=data.get('bid'),
                    ask=data.get('ask'),
                    spread=data.get('spread'),
                    spread_pct=data.get('spread_pct')
                ),
                ohlc=OHLCData(
                    m15_open=data.get('m15_open'),
                    m15_high=data.get('m15_high'),
                    m15_low=data.get('m15_low'),
                    m15_close=data.get('m15_close'),
                    h1_open=data.get('h1_open'),
                    h1_high=data.get('h1_high'),
                    h1_low=data.get('h1_low'),
                    h1_close=data.get('h1_close'),
                    d1_open=data.get('d1_open'),
                    d1_high=data.get('d1_high'),
                    d1_low=data.get('d1_low'),
                    d1_close=data.get('d1_close')
                ),
                indicators=AllIndicators(
                    rsi=RSIIndicator(value=data.get('rsi')),
                    macd=MACDIndicator(
                        main=data.get('macd_main'),
                        signal=data.get('macd_signal')
                    ),
                    stochastic=StochasticIndicator(
                        main=data.get('sto_main'),
                        signal=data.get('sto_signal')
                    ),
                    cci=CCIIndicator(value=data.get('cci')),
                    adx=ADXIndicator(
                        main=data.get('adx_main'),
                        plus_di=data.get('adx_plusdi'),
                        minus_di=data.get('adx_minusdi')
                    ),
                    ma=MAIndicator(ma_10=data.get('ma_10')),
                    ichimoku=IchimokuIndicator(
                        tenkan=data.get('ichimoku_tenkan'),
                        kijun=data.get('ichimoku_kijun'),
                        senkou_a=data.get('ichimoku_senkoua'),
                        senkou_b=data.get('ichimoku_senkoub'),
                        chikou=data.get('ichimoku_chikou')
                    ),
                    bollinger=BollingerBandsIndicator(
                        upper=data.get('bb_upper'),
                        base=data.get('bb_base'),
                        lower=data.get('bb_lower')
                    ),
                    atr=ATRIndicator(
                        d1=data.get('atr_d1'),
                        d1_pct=data.get('atr_pct_d1')
                    ),
                    range=RangeIndicator(d1=data.get('range_d1')),
                    pivot_points=PivotPoints(
                        s1_m5=data.get('s1_level_m5'),
                        r1_m5=data.get('r1_level_m5')
                    ),
                    strength=StrengthIndicators(
                        h4=data.get('strength_4h'),
                        d1=data.get('strength_1d'),
                        w1=data.get('strength_1w')
                    )
                )
            )

            logger.info(f"Fetched latest market data for {symbol} via Data Gateway")
            return snapshot

        except Exception as e:
            logger.error(f"Failed to fetch latest market data for {symbol}: {e}")
            return None

    async def _fetch_time_series(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "h1"
    ) -> list[TimeSeriesData]:
        """Fetch time series data from EasyInsight API.

        Args:
            symbol: Trading symbol
            start_date: Start date for data range
            end_date: End date for data range
            interval: Data interval - 'm15' (15-min), 'h1' (hourly), 'd1' (daily)
        """
        result = await self._fetch_time_series_with_details(symbol, start_date, end_date, interval)
        return result[0]

    async def _fetch_time_series_with_details(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "h1"
    ) -> tuple[list[TimeSeriesData], TimescaleDBDataLog, dict]:
        """
        Fetch time series data via Data Gateway with detailed logging.

        Verwendet: DataGatewayService (siehe DEVELOPMENT_GUIDELINES.md)

        Args:
            symbol: Trading symbol
            start_date: Start date for data range
            end_date: End date for data range
            interval: Data interval - 'm15' (15-min), 'h1' (hourly), 'd1' (daily)

        Returns:
            tuple: (time_series_data, tsdb_log, db_indicators)
        """
        # Normalize interval
        interval = interval.lower()
        if interval not in ["m15", "h1", "d1"]:
            interval = "h1"

        # Calculate number of data points needed based on interval
        days = (end_date - start_date).days
        if interval == "m15":
            # 15-minute data: 96 candles per day
            limit = max(days * 96, 192)  # At least 2 days of M15 data
        elif interval == "d1":
            # Daily data: 1 candle per day
            limit = max(days, 45)  # At least 45 days of D1 data
        else:  # h1
            # Hourly data: 24 candles per day
            limit = max(days * 24, 168)  # At least 1 week of H1 data

        time_series = []
        db_indicators = {}
        raw_data_sample = []

        try:
            # Verwende Data Gateway anstelle von direktem API-Zugriff
            rows = await self.data_gateway.get_historical_data(
                symbol=symbol,
                limit=limit,
                timeframe=interval.upper()
            )

            for i, row in enumerate(rows):
                try:
                    # Parse timestamp
                    timestamp_str = row.get('snapshot_time')
                    if not timestamp_str:
                        continue

                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                    # Select OHLC fields based on interval
                    if interval == "m15":
                        open_price = float(row.get('m15_open', 0))
                        high_price = float(row.get('m15_high', 0))
                        low_price = float(row.get('m15_low', 0))
                        close_price = float(row.get('m15_close', 0))
                    elif interval == "d1":
                        open_price = float(row.get('d1_open', 0))
                        high_price = float(row.get('d1_high', 0))
                        low_price = float(row.get('d1_low', 0))
                        close_price = float(row.get('d1_close', 0))
                    else:  # h1
                        open_price = float(row.get('h1_open', 0))
                        high_price = float(row.get('h1_high', 0))
                        low_price = float(row.get('h1_low', 0))
                        close_price = float(row.get('h1_close', 0))

                    # Use interval-specific OHLC data with all available indicators
                    ts_data = TimeSeriesData(
                        timestamp=timestamp,
                        symbol=symbol,
                        open=open_price,
                        high=high_price,
                        low=low_price,
                        close=close_price,
                        volume=0.0,
                        # Momentum Indicators
                        rsi=row.get('rsi'),
                        macd_main=row.get('macd_main'),
                        macd_signal=row.get('macd_signal'),
                        cci=row.get('cci'),
                        stoch_k=row.get('sto_main'),
                        stoch_d=row.get('sto_signal'),
                        # Trend Indicators
                        adx=row.get('adx_main'),
                        adx_plus_di=row.get('adx_plusdi'),
                        adx_minus_di=row.get('adx_minusdi'),
                        ma100=row.get('ma_10'),
                        # Volatility Indicators
                        atr=row.get('atr_d1'),
                        atr_pct=row.get('atr_pct_d1'),
                        bb_upper=row.get('bb_upper'),
                        bb_middle=row.get('bb_base'),
                        bb_lower=row.get('bb_lower'),
                        range_d1=row.get('range_d1'),
                        # Ichimoku Cloud (complete)
                        ichimoku_tenkan=row.get('ichimoku_tenkan'),
                        ichimoku_kijun=row.get('ichimoku_kijun'),
                        ichimoku_senkou_a=row.get('ichimoku_senkoua'),
                        ichimoku_senkou_b=row.get('ichimoku_senkoub'),
                        ichimoku_chikou=row.get('ichimoku_chikou'),
                        # Strength Indicators
                        strength_4h=row.get('strength_4h'),
                        strength_1d=row.get('strength_1d'),
                        strength_1w=row.get('strength_1w'),
                        # Support/Resistance Pivot Points
                        s1_level=row.get('s1_level_m5'),
                        r1_level=row.get('r1_level_m5'),
                        additional_data={
                            'bid': row.get('bid'),
                            'ask': row.get('ask'),
                            'spread': row.get('spread'),
                            'spread_pct': row.get('spread_pct'),
                            'category': row.get('category'),
                            'd1_open': row.get('d1_open'),
                            'd1_high': row.get('d1_high'),
                            'd1_low': row.get('d1_low'),
                            'd1_close': row.get('d1_close'),
                            'm15_open': row.get('m15_open'),
                            'm15_high': row.get('m15_high'),
                            'm15_low': row.get('m15_low'),
                            'm15_close': row.get('m15_close'),
                        }
                    )
                    time_series.append(ts_data)

                    # Store raw data sample (first 3 rows)
                    if i < 3:
                        raw_data_sample.append({
                            'timestamp': timestamp_str,
                            'h1_close': row.get('h1_close'),
                            'rsi': row.get('rsi'),
                            'macd_main': row.get('macd_main'),
                        })

                except Exception as row_error:
                    logger.warning(f"Failed to parse row for {symbol}: {row_error}")
                    continue

            # Extract latest indicators for db_indicators dict
            if rows:
                latest = rows[0]
                db_indicators = {
                    'rsi14': latest.get('rsi'),
                    'macd_main': latest.get('macd_main'),
                    'macd_signal': latest.get('macd_signal'),
                    'bb_upper': latest.get('bb_upper'),
                    'bb_middle': latest.get('bb_base'),
                    'bb_lower': latest.get('bb_lower'),
                    'atr_d1': latest.get('atr_d1'),
                    'ma100': latest.get('ma_10'),
                    'adx_main': latest.get('adx_main'),
                    'adx_plusdi': latest.get('adx_plusdi'),
                    'adx_minusdi': latest.get('adx_minusdi'),
                }

            logger.info(f"Fetched {len(time_series)} {interval.upper()} data points for {symbol} via Data Gateway")

        except Exception as e:
            logger.error(f"Failed to fetch time series for {symbol}: {e}")

        # Sort by timestamp (oldest first for analysis)
        time_series.sort(key=lambda x: x.timestamp)

        # Create detailed log
        tsdb_log = TimescaleDBDataLog(
            query_timestamp=datetime.now(),
            tables_queried=["easyinsight_api"],
            symbols_queried=[symbol],
            time_range_start=start_date,
            time_range_end=end_date,
            rows_fetched=len(time_series),
            ohlc_data={
                interval: {
                    'open': time_series[-1].open if time_series else None,
                    'high': time_series[-1].high if time_series else None,
                    'low': time_series[-1].low if time_series else None,
                    'close': time_series[-1].close if time_series else None,
                } if time_series else None
            },
            indicators_fetched=db_indicators,
            raw_data_sample=raw_data_sample,
            sql_queries=[f"GET /symbol-data-full/{symbol}?limit={limit}&interval={interval}"]
        )

        return time_series, tsdb_log, db_indicators

    async def get_available_symbols(self) -> list[str]:
        """
        Get list of available trading symbols via Data Gateway.

        Verwendet: DataGatewayService (siehe DEVELOPMENT_GUIDELINES.md)
        HINWEIS: Direkter DB-Zugriff wurde entfernt - nur noch API via Data Gateway.
        """
        return await self.data_gateway.get_symbol_names()

    async def fetch_all_latest_market_data(self) -> list[MarketDataSnapshot]:
        """
        Fetch latest market data for all available symbols via Data Gateway.

        Returns a list of MarketDataSnapshot objects with complete indicator data.

        Verwendet: DataGatewayService (siehe DEVELOPMENT_GUIDELINES.md)
        """
        try:
            # Verwende Data Gateway anstelle von direktem API-Zugriff
            data_rows = await self.data_gateway.get_all_latest_market_data()
            snapshots = []

            for row in data_rows:
                try:
                    snapshot_time = datetime.fromisoformat(
                        row['snapshot_time'].replace('Z', '+00:00')
                    )

                    snapshot = MarketDataSnapshot(
                        symbol=row['symbol'],
                        snapshot_time=snapshot_time,
                        category=row.get('category', 'Unknown'),
                        price=PriceData(
                            bid=row.get('bid'),
                            ask=row.get('ask'),
                            spread=row.get('spread'),
                            spread_pct=row.get('spread_pct')
                        ),
                        ohlc=OHLCData(
                            m15_open=row.get('m15_open'),
                            m15_high=row.get('m15_high'),
                            m15_low=row.get('m15_low'),
                            m15_close=row.get('m15_close'),
                            h1_open=row.get('h1_open'),
                            h1_high=row.get('h1_high'),
                            h1_low=row.get('h1_low'),
                            h1_close=row.get('h1_close'),
                            d1_open=row.get('d1_open'),
                            d1_high=row.get('d1_high'),
                            d1_low=row.get('d1_low'),
                            d1_close=row.get('d1_close')
                        ),
                        indicators=AllIndicators(
                            rsi=RSIIndicator(value=row.get('rsi')),
                            macd=MACDIndicator(
                                main=row.get('macd_main'),
                                signal=row.get('macd_signal')
                            ),
                            stochastic=StochasticIndicator(
                                main=row.get('sto_main'),
                                signal=row.get('sto_signal')
                            ),
                            cci=CCIIndicator(value=row.get('cci')),
                            adx=ADXIndicator(
                                main=row.get('adx_main'),
                                plus_di=row.get('adx_plusdi'),
                                minus_di=row.get('adx_minusdi')
                            ),
                            ma=MAIndicator(ma_10=row.get('ma_10')),
                            ichimoku=IchimokuIndicator(
                                tenkan=row.get('ichimoku_tenkan'),
                                kijun=row.get('ichimoku_kijun'),
                                senkou_a=row.get('ichimoku_senkoua'),
                                senkou_b=row.get('ichimoku_senkoub'),
                                chikou=row.get('ichimoku_chikou')
                            ),
                            bollinger=BollingerBandsIndicator(
                                upper=row.get('bb_upper'),
                                base=row.get('bb_base'),
                                lower=row.get('bb_lower')
                            ),
                            atr=ATRIndicator(
                                d1=row.get('atr_d1'),
                                d1_pct=row.get('atr_pct_d1')
                            ),
                            range=RangeIndicator(d1=row.get('range_d1')),
                            pivot_points=PivotPoints(
                                s1_m5=row.get('s1_level_m5'),
                                r1_m5=row.get('r1_level_m5')
                            ),
                            strength=StrengthIndicators(
                                h4=row.get('strength_4h'),
                                d1=row.get('strength_1d'),
                                w1=row.get('strength_1w')
                            )
                        )
                    )
                    snapshots.append(snapshot)

                except Exception as row_error:
                    logger.warning(f"Failed to parse market data for {row.get('symbol')}: {row_error}")
                    continue

            logger.info(f"Fetched latest market data for {len(snapshots)} symbols via Data Gateway")
            return snapshots

        except Exception as e:
            logger.error(f"Failed to fetch all latest market data: {e}")
            return []

    async def get_symbol_info(self, symbol: str) -> dict:
        """Get detailed information about a symbol from EasyInsight API including all available indicators."""
        try:
            # Fetch latest market data from EasyInsight API
            snapshot = await self.fetch_latest_market_data(symbol)

            if not snapshot:
                return {"error": f"Symbol {symbol} not found"}

            # Get symbol list to check availability and metadata
            symbols_list = await self.get_available_symbols()
            symbol_meta = None
            for s in symbols_list:
                if isinstance(s, dict) and s.get('symbol') == symbol:
                    symbol_meta = s
                    break

            # Helper function to safely convert to float
            def safe_float(value):
                return float(value) if value is not None else None

            # Calculate derived indicators from snapshot
            macd_main = snapshot.indicators.macd.main if snapshot.indicators and snapshot.indicators.macd else None
            macd_signal = snapshot.indicators.macd.signal if snapshot.indicators and snapshot.indicators.macd else None
            macd_histogram = (macd_main - macd_signal) if (macd_main is not None and macd_signal is not None) else None

            # Determine indicator signals
            rsi_value = snapshot.indicators.rsi.value if snapshot.indicators and snapshot.indicators.rsi else None
            rsi_signal = None
            if rsi_value is not None:
                if rsi_value > 70:
                    rsi_signal = "overbought"
                elif rsi_value < 30:
                    rsi_signal = "oversold"
                else:
                    rsi_signal = "neutral"

            macd_trend = None
            if macd_main is not None and macd_signal is not None:
                macd_trend = "bullish" if macd_main > macd_signal else "bearish"

            stoch_k = snapshot.indicators.stochastic.k if snapshot.indicators and snapshot.indicators.stochastic else None
            stoch_d = snapshot.indicators.stochastic.d if snapshot.indicators and snapshot.indicators.stochastic else None
            stoch_signal = None
            if stoch_k is not None:
                if stoch_k > 80:
                    stoch_signal = "overbought"
                elif stoch_k < 20:
                    stoch_signal = "oversold"
                else:
                    stoch_signal = "neutral"

            adx_main = snapshot.indicators.adx.main if snapshot.indicators and snapshot.indicators.adx else None
            adx_plus_di = snapshot.indicators.adx.plus_di if snapshot.indicators and snapshot.indicators.adx else None
            adx_minus_di = snapshot.indicators.adx.minus_di if snapshot.indicators and snapshot.indicators.adx else None
            trend_strength = None
            trend_direction = None
            if adx_main is not None:
                if adx_main > 25:
                    trend_strength = "strong"
                elif adx_main > 20:
                    trend_strength = "moderate"
                else:
                    trend_strength = "weak"
            if adx_plus_di is not None and adx_minus_di is not None:
                trend_direction = "bullish" if adx_plus_di > adx_minus_di else "bearish"

            ichimoku_tenkan = snapshot.indicators.ichimoku.tenkan if snapshot.indicators and snapshot.indicators.ichimoku else None
            ichimoku_kijun = snapshot.indicators.ichimoku.kijun if snapshot.indicators and snapshot.indicators.ichimoku else None
            ichimoku_senkou_a = snapshot.indicators.ichimoku.senkou_a if snapshot.indicators and snapshot.indicators.ichimoku else None
            ichimoku_senkou_b = snapshot.indicators.ichimoku.senkou_b if snapshot.indicators and snapshot.indicators.ichimoku else None
            ichimoku_tk_signal = None
            ichimoku_cloud_signal = None
            if ichimoku_tenkan is not None and ichimoku_kijun is not None:
                ichimoku_tk_signal = "bullish" if ichimoku_tenkan > ichimoku_kijun else "bearish"
            if ichimoku_senkou_a is not None and ichimoku_senkou_b is not None:
                ichimoku_cloud_signal = "bullish" if ichimoku_senkou_a > ichimoku_senkou_b else "bearish"

            bb_upper = snapshot.indicators.bollinger_bands.upper if snapshot.indicators and snapshot.indicators.bollinger_bands else None
            bb_lower = snapshot.indicators.bollinger_bands.lower if snapshot.indicators and snapshot.indicators.bollinger_bands else None
            bb_middle = snapshot.indicators.bollinger_bands.middle if snapshot.indicators and snapshot.indicators.bollinger_bands else None
            current_close = snapshot.ohlc.h1_close if snapshot.ohlc else None
            bb_position = None
            if current_close is not None and bb_upper is not None and bb_lower is not None:
                if current_close >= bb_upper:
                    bb_position = "at_upper_band"
                elif current_close <= bb_lower:
                    bb_position = "at_lower_band"
                else:
                    bb_position = "within_bands"

            return {
                "symbol": symbol,
                "last_timestamp": snapshot.snapshot_time.isoformat(),
                "first_timestamp": symbol_meta.get('earliest') if symbol_meta else None,
                "total_records": symbol_meta.get('count') if symbol_meta else None,
                "data_timestamp": snapshot.snapshot_time.isoformat(),
                "category": snapshot.category,

                # OHLC Data - Daily (D1)
                "ohlc_d1": {
                    "open": snapshot.ohlc.d1_open if snapshot.ohlc else None,
                    "high": snapshot.ohlc.d1_high if snapshot.ohlc else None,
                    "low": snapshot.ohlc.d1_low if snapshot.ohlc else None,
                    "close": snapshot.ohlc.d1_close if snapshot.ohlc else None
                },
                # OHLC Data - Hourly (H1)
                "ohlc_h1": {
                    "open": snapshot.ohlc.h1_open if snapshot.ohlc else None,
                    "high": snapshot.ohlc.h1_high if snapshot.ohlc else None,
                    "low": snapshot.ohlc.h1_low if snapshot.ohlc else None,
                    "close": snapshot.ohlc.h1_close if snapshot.ohlc else None
                },
                # OHLC Data - 15 Minutes (M15)
                "ohlc_m15": {
                    "open": snapshot.ohlc.m15_open if snapshot.ohlc else None,
                    "high": snapshot.ohlc.m15_high if snapshot.ohlc else None,
                    "low": snapshot.ohlc.m15_low if snapshot.ohlc else None,
                    "close": snapshot.ohlc.m15_close if snapshot.ohlc else None
                },

                # Price Data
                "price": {
                    "bid": snapshot.price.bid if snapshot.price else None,
                    "ask": snapshot.price.ask if snapshot.price else None,
                    "spread": snapshot.price.spread if snapshot.price else None
                },

                # Momentum Indicators
                "indicators": {
                    "rsi": {
                        "value": rsi_value,
                        "period": 14,
                        "signal": rsi_signal
                    },
                    "macd": {
                        "main_line": macd_main,
                        "signal_line": macd_signal,
                        "histogram": macd_histogram,
                        "parameters": "12,26,9",
                        "trend": macd_trend
                    },
                    "stochastic": {
                        "k_line": stoch_k,
                        "d_line": stoch_d,
                        "parameters": "5,3,3",
                        "signal": stoch_signal
                    },
                    "cci": {
                        "value": snapshot.indicators.cci.value if snapshot.indicators and snapshot.indicators.cci else None,
                        "period": 14
                    },
                    "adx": {
                        "main_line": adx_main,
                        "plus_di": adx_plus_di,
                        "minus_di": adx_minus_di,
                        "period": 14,
                        "trend_strength": trend_strength,
                        "trend_direction": trend_direction
                    },
                    "ma100": {
                        "value": snapshot.indicators.ma.value if snapshot.indicators and snapshot.indicators.ma else None,
                        "type": "SMA",
                        "period": 10
                    },
                    "ichimoku": {
                        "tenkan_sen": ichimoku_tenkan,
                        "kijun_sen": ichimoku_kijun,
                        "senkou_span_a": ichimoku_senkou_a,
                        "senkou_span_b": ichimoku_senkou_b,
                        "chikou_span": snapshot.indicators.ichimoku.chikou if snapshot.indicators and snapshot.indicators.ichimoku else None,
                        "parameters": "9,26,52",
                        "tk_signal": ichimoku_tk_signal,
                        "cloud_signal": ichimoku_cloud_signal
                    },
                    "bollinger_bands": {
                        "upper_band": bb_upper,
                        "middle_band": bb_middle,
                        "lower_band": bb_lower,
                        "period": 20,
                        "std_dev": 2,
                        "price_position": bb_position
                    },
                    "atr": {
                        "value": snapshot.indicators.atr.value if snapshot.indicators and snapshot.indicators.atr else None,
                        "timeframe": "D1"
                    },
                    "range": {
                        "value": snapshot.indicators.range_indicator.value if snapshot.indicators and snapshot.indicators.range_indicator else None,
                        "timeframe": "D1"
                    }
                },

                # Support/Resistance (Pivot Points)
                "pivot_points": {
                    "r1": snapshot.indicators.pivot_points.r1 if snapshot.indicators and snapshot.indicators.pivot_points else None,
                    "s1": snapshot.indicators.pivot_points.s1 if snapshot.indicators and snapshot.indicators.pivot_points else None,
                    "timeframe": "M5"
                },

                # Strength Indicators (Multi-Timeframe)
                "strength": {
                    "h4": snapshot.indicators.strength.h4 if snapshot.indicators and snapshot.indicators.strength else None,
                    "d1": snapshot.indicators.strength.d1 if snapshot.indicators and snapshot.indicators.strength else None,
                    "w1": snapshot.indicators.strength.w1 if snapshot.indicators and snapshot.indicators.strength else None
                }
            }

        except Exception as e:
            logger.error(f"Error fetching symbol info for {symbol}: {e}")
            return {"error": str(e)}

    async def check_easyinsight_api(self) -> bool:
        """
        Check if EasyInsight API is accessible via Data Gateway.

        Verwendet: DataGatewayService (siehe DEVELOPMENT_GUIDELINES.md)
        """
        return await self.data_gateway.check_easyinsight_health()

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
            time_series, tsdb_log, db_indicators = await self._fetch_time_series_with_details(
                symbol=request.symbol,
                start_date=datetime.now() - timedelta(days=effective_lookback),
                end_date=datetime.now()
            )

            if not time_series:
                raise ValueError(f"No data available for symbol {request.symbol}")

            # 2. Calculate technical indicators (use DB indicators where available)
            market_analysis = await self._create_market_analysis(
                symbol=request.symbol,
                time_series=time_series,
                db_indicators=db_indicators
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
        time_series: list[TimeSeriesData],
        db_indicators: dict = None
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

        # Calculate technical indicators (use DB values where available)
        indicators = self._calculate_indicators(df, db_indicators)

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

        # Generate NHITS forecast if enabled
        nhits_forecast = None
        if app_settings.nhits_enabled:
            nhits_forecast = await self._get_nhits_forecast(symbol, time_series)

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
            resistance_levels=resistance_levels,
            nhits_forecast=nhits_forecast
        )

    async def _get_nhits_forecast(
        self,
        symbol: str,
        time_series: list[TimeSeriesData]
    ) -> NHITSForecast | None:
        """
        Generate NHITS neural network forecast for the symbol.

        Args:
            symbol: Trading symbol
            time_series: Historical time series data

        Returns:
            NHITSForecast or None if forecast fails
        """
        # Timeout for NHITS forecast (120 seconds to allow for initial training)
        # First-time training for a new symbol can take 60-90 seconds
        NHITS_TIMEOUT_SECONDS = 120

        try:
            from .forecast_service import forecast_service

            # Check if we have enough data for NHITS
            min_required = app_settings.nhits_input_size + app_settings.nhits_horizon
            if len(time_series) < min_required:
                logger.warning(
                    f"NHITS: Insufficient data for {symbol}: "
                    f"{len(time_series)} < {min_required} required"
                )
                return None

            # Generate forecast with timeout to prevent blocking
            try:
                forecast_result = await asyncio.wait_for(
                    forecast_service.forecast(
                        time_series=time_series,
                        symbol=symbol
                    ),
                    timeout=NHITS_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"NHITS: Forecast timed out for {symbol} "
                    f"after {NHITS_TIMEOUT_SECONDS}s"
                )
                return None

            # Check if forecast was successful
            if not forecast_result.predicted_prices:
                logger.warning(f"NHITS: Empty forecast for {symbol}")
                return None

            # Convert to NHITSForecast model
            nhits_forecast = NHITSForecast(
                predicted_price_1h=forecast_result.predicted_price_1h,
                predicted_price_4h=forecast_result.predicted_price_4h,
                predicted_price_24h=forecast_result.predicted_price_24h,
                predicted_change_percent_1h=forecast_result.predicted_change_percent_1h,
                predicted_change_percent_4h=forecast_result.predicted_change_percent_4h,
                predicted_change_percent_24h=forecast_result.predicted_change_percent_24h,
                confidence_low_24h=forecast_result.confidence_low[-1] if forecast_result.confidence_low else None,
                confidence_high_24h=forecast_result.confidence_high[-1] if forecast_result.confidence_high else None,
                trend_up_probability=forecast_result.trend_up_probability,
                trend_down_probability=forecast_result.trend_down_probability,
                model_confidence=forecast_result.model_confidence,
                predicted_volatility=forecast_result.predicted_volatility,
            )

            logger.info(
                f"NHITS forecast for {symbol}: "
                f"24h change {forecast_result.predicted_change_percent_24h:+.2f}%, "
                f"confidence {forecast_result.model_confidence:.1%}"
            )

            return nhits_forecast

        except ImportError as e:
            logger.warning(f"NHITS: NeuralForecast not installed: {e}")
            return None
        except Exception as e:
            logger.error(f"NHITS forecast failed for {symbol}: {e}")
            return None

    def _calculate_indicators(self, df: pd.DataFrame, db_indicators: dict = None) -> TechnicalIndicators:
        """Calculate all technical indicators. Use DB values where available."""

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume_data = df["volume"]

        # Use DB indicators if available, otherwise calculate
        db = db_indicators or {}

        # Moving Averages (SMA)
        sma_20 = close.rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
        sma_50 = close.rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
        sma_200 = close.rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None

        # Use MA100 from DB if available
        if db.get("ma100"):
            sma_200 = db["ma100"]  # Use MA100 as approximation for SMA200

        # Exponential Moving Averages (EMA)
        ema_12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = close.ewm(span=26, adjust=False).mean().iloc[-1]

        # RSI - prefer DB value (from MT5)
        if db.get("rsi14") is not None:
            rsi = db["rsi14"]
            logger.info(f"Using DB RSI value: {rsi}")
        else:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            logger.info(f"Calculated RSI value: {rsi} (no DB value available)")

        # MACD - prefer DB values
        if db.get("macd_main") is not None and db.get("macd_signal") is not None:
            macd_value = db["macd_main"]
            macd_signal = db["macd_signal"]
            macd_histogram = macd_value - macd_signal
        else:
            ema_12_series = close.ewm(span=12, adjust=False).mean()
            ema_26_series = close.ewm(span=26, adjust=False).mean()
            macd_line = ema_12_series - ema_26_series
            macd_signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_value = macd_line.iloc[-1]
            macd_signal = macd_signal_line.iloc[-1]
            macd_histogram = (macd_line - macd_signal_line).iloc[-1]

        # Bollinger Bands - prefer DB values
        if db.get("bb_upper") is not None and db.get("bb_middle") is not None and db.get("bb_lower") is not None:
            bollinger_upper = db["bb_upper"]
            bollinger_middle = db["bb_middle"]
            bollinger_lower = db["bb_lower"]
        else:
            sma_20_series = close.rolling(window=20).mean()
            std_20 = close.rolling(window=20).std()
            bollinger_upper = (sma_20_series + (std_20 * 2)).iloc[-1]
            bollinger_middle = sma_20_series.iloc[-1]
            bollinger_lower = (sma_20_series - (std_20 * 2)).iloc[-1]

        # ATR - prefer DB value
        if db.get("atr_d1") is not None:
            atr = db["atr_d1"]
        else:
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
        time_series, tsdb_log, db_indicators = await self._fetch_time_series_with_details(
            symbol=symbol,
            start_date=datetime.now() - timedelta(days=effective_lookback),
            end_date=datetime.now()
        )

        if not time_series:
            raise ValueError(f"No data available for symbol {symbol}")

        # Calculate technical indicators (use DB indicators where available)
        market_analysis = await self._create_market_analysis(
            symbol=symbol,
            time_series=time_series,
            db_indicators=db_indicators
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

        # Fetch and attach complete market data snapshot
        market_data_snapshot = await self.fetch_latest_market_data(symbol)
        recommendation.market_data = market_data_snapshot

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
