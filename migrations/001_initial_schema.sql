-- TimescaleDB Initial Schema Migration
-- Version: 001
-- Date: 2024-12-XX
-- Description: Creates all tables for the Trading Data Service

-- =====================================================
-- PREREQUISITE: TimescaleDB Extension
-- =====================================================
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =====================================================
-- OHLCV TABLES (One per Timeframe)
-- =====================================================

-- M1 (1 Minute) - High frequency, short retention
CREATE TABLE IF NOT EXISTS ohlcv_m1 (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    open            DECIMAL(20, 8) NOT NULL,
    high            DECIMAL(20, 8) NOT NULL,
    low             DECIMAL(20, 8) NOT NULL,
    close           DECIMAL(20, 8) NOT NULL,
    volume          DECIMAL(30, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('ohlcv_m1', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_m1_symbol ON ohlcv_m1 (symbol, timestamp DESC);

-- M5 (5 Minutes)
CREATE TABLE IF NOT EXISTS ohlcv_m5 (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    open            DECIMAL(20, 8) NOT NULL,
    high            DECIMAL(20, 8) NOT NULL,
    low             DECIMAL(20, 8) NOT NULL,
    close           DECIMAL(20, 8) NOT NULL,
    volume          DECIMAL(30, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('ohlcv_m5', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_m5_symbol ON ohlcv_m5 (symbol, timestamp DESC);

-- M15 (15 Minutes)
CREATE TABLE IF NOT EXISTS ohlcv_m15 (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    open            DECIMAL(20, 8) NOT NULL,
    high            DECIMAL(20, 8) NOT NULL,
    low             DECIMAL(20, 8) NOT NULL,
    close           DECIMAL(20, 8) NOT NULL,
    volume          DECIMAL(30, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('ohlcv_m15', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_m15_symbol ON ohlcv_m15 (symbol, timestamp DESC);

-- M30 (30 Minutes)
CREATE TABLE IF NOT EXISTS ohlcv_m30 (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    open            DECIMAL(20, 8) NOT NULL,
    high            DECIMAL(20, 8) NOT NULL,
    low             DECIMAL(20, 8) NOT NULL,
    close           DECIMAL(20, 8) NOT NULL,
    volume          DECIMAL(30, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('ohlcv_m30', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_m30_symbol ON ohlcv_m30 (symbol, timestamp DESC);

-- M45 (45 Minutes)
CREATE TABLE IF NOT EXISTS ohlcv_m45 (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    open            DECIMAL(20, 8) NOT NULL,
    high            DECIMAL(20, 8) NOT NULL,
    low             DECIMAL(20, 8) NOT NULL,
    close           DECIMAL(20, 8) NOT NULL,
    volume          DECIMAL(30, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('ohlcv_m45', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_m45_symbol ON ohlcv_m45 (symbol, timestamp DESC);

-- H1 (1 Hour)
CREATE TABLE IF NOT EXISTS ohlcv_h1 (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    open            DECIMAL(20, 8) NOT NULL,
    high            DECIMAL(20, 8) NOT NULL,
    low             DECIMAL(20, 8) NOT NULL,
    close           DECIMAL(20, 8) NOT NULL,
    volume          DECIMAL(30, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('ohlcv_h1', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_h1_symbol ON ohlcv_h1 (symbol, timestamp DESC);

-- H2 (2 Hours)
CREATE TABLE IF NOT EXISTS ohlcv_h2 (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    open            DECIMAL(20, 8) NOT NULL,
    high            DECIMAL(20, 8) NOT NULL,
    low             DECIMAL(20, 8) NOT NULL,
    close           DECIMAL(20, 8) NOT NULL,
    volume          DECIMAL(30, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('ohlcv_h2', 'timestamp',
    chunk_time_interval => INTERVAL '14 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_h2_symbol ON ohlcv_h2 (symbol, timestamp DESC);

-- H4 (4 Hours)
CREATE TABLE IF NOT EXISTS ohlcv_h4 (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    open            DECIMAL(20, 8) NOT NULL,
    high            DECIMAL(20, 8) NOT NULL,
    low             DECIMAL(20, 8) NOT NULL,
    close           DECIMAL(20, 8) NOT NULL,
    volume          DECIMAL(30, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('ohlcv_h4', 'timestamp',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_h4_symbol ON ohlcv_h4 (symbol, timestamp DESC);

-- D1 (Daily)
CREATE TABLE IF NOT EXISTS ohlcv_d1 (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    open            DECIMAL(20, 8) NOT NULL,
    high            DECIMAL(20, 8) NOT NULL,
    low             DECIMAL(20, 8) NOT NULL,
    close           DECIMAL(20, 8) NOT NULL,
    volume          DECIMAL(30, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('ohlcv_d1', 'timestamp',
    chunk_time_interval => INTERVAL '365 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_d1_symbol ON ohlcv_d1 (symbol, timestamp DESC);

-- W1 (Weekly)
CREATE TABLE IF NOT EXISTS ohlcv_w1 (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    open            DECIMAL(20, 8) NOT NULL,
    high            DECIMAL(20, 8) NOT NULL,
    low             DECIMAL(20, 8) NOT NULL,
    close           DECIMAL(20, 8) NOT NULL,
    volume          DECIMAL(30, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('ohlcv_w1', 'timestamp',
    chunk_time_interval => INTERVAL '365 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_w1_symbol ON ohlcv_w1 (symbol, timestamp DESC);

-- MN (Monthly)
CREATE TABLE IF NOT EXISTS ohlcv_mn (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    open            DECIMAL(20, 8) NOT NULL,
    high            DECIMAL(20, 8) NOT NULL,
    low             DECIMAL(20, 8) NOT NULL,
    close           DECIMAL(20, 8) NOT NULL,
    volume          DECIMAL(30, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('ohlcv_mn', 'timestamp',
    chunk_time_interval => INTERVAL '365 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_mn_symbol ON ohlcv_mn (symbol, timestamp DESC);

-- =====================================================
-- INDICATORS TABLES
-- =====================================================

-- Flexible JSONB table for rare/custom indicators
CREATE TABLE IF NOT EXISTS indicators (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    indicator_name  VARCHAR(50) NOT NULL,
    values          JSONB NOT NULL,
    parameters      JSONB,
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, timeframe, indicator_name)
);

SELECT create_hypertable('indicators', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_indicators_lookup
    ON indicators (symbol, timeframe, indicator_name, timestamp DESC);

-- Partial indexes for frequently queried indicators
CREATE INDEX IF NOT EXISTS idx_indicators_rsi
    ON indicators (symbol, timeframe, timestamp DESC)
    WHERE indicator_name = 'RSI';

CREATE INDEX IF NOT EXISTS idx_indicators_macd
    ON indicators (symbol, timeframe, timestamp DESC)
    WHERE indicator_name = 'MACD';

-- Moving Averages (optimized table)
CREATE TABLE IF NOT EXISTS indicators_ma (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    sma_20          DECIMAL(20, 8),
    sma_50          DECIMAL(20, 8),
    sma_200         DECIMAL(20, 8),
    ema_12          DECIMAL(20, 8),
    ema_26          DECIMAL(20, 8),
    ema_50          DECIMAL(20, 8),
    ema_200         DECIMAL(20, 8),
    wma_20          DECIMAL(20, 8),
    dema_20         DECIMAL(20, 8),
    tema_20         DECIMAL(20, 8),
    vwap            DECIMAL(20, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_ma', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_indicators_ma_lookup
    ON indicators_ma (symbol, timeframe, timestamp DESC);

-- Momentum Indicators (optimized table)
CREATE TABLE IF NOT EXISTS indicators_momentum (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    rsi_14          DECIMAL(10, 4),
    rsi_7           DECIMAL(10, 4),
    rsi_21          DECIMAL(10, 4),
    stoch_rsi       DECIMAL(10, 4),
    connors_rsi     DECIMAL(10, 4),
    stoch_k         DECIMAL(10, 4),
    stoch_d         DECIMAL(10, 4),
    macd_line       DECIMAL(20, 8),
    macd_signal     DECIMAL(20, 8),
    macd_histogram  DECIMAL(20, 8),
    cci             DECIMAL(10, 4),
    williams_r      DECIMAL(10, 4),
    roc             DECIMAL(10, 4),
    momentum        DECIMAL(20, 8),
    adx             DECIMAL(10, 4),
    plus_di         DECIMAL(10, 4),
    minus_di        DECIMAL(10, 4),
    mfi             DECIMAL(10, 4),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_momentum', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_indicators_momentum_lookup
    ON indicators_momentum (symbol, timeframe, timestamp DESC);

-- Volatility Indicators (optimized table)
CREATE TABLE IF NOT EXISTS indicators_volatility (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    bb_upper        DECIMAL(20, 8),
    bb_middle       DECIMAL(20, 8),
    bb_lower        DECIMAL(20, 8),
    bb_width        DECIMAL(10, 6),
    bb_percent_b    DECIMAL(10, 6),
    atr_14          DECIMAL(20, 8),
    atr_7           DECIMAL(20, 8),
    natr            DECIMAL(10, 6),
    true_range      DECIMAL(20, 8),
    kc_upper        DECIMAL(20, 8),
    kc_middle       DECIMAL(20, 8),
    kc_lower        DECIMAL(20, 8),
    dc_upper        DECIMAL(20, 8),
    dc_middle       DECIMAL(20, 8),
    dc_lower        DECIMAL(20, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_volatility', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_indicators_volatility_lookup
    ON indicators_volatility (symbol, timeframe, timestamp DESC);

-- Trend Indicators (optimized table)
CREATE TABLE IF NOT EXISTS indicators_trend (
    timestamp           TIMESTAMPTZ NOT NULL,
    symbol              VARCHAR(20) NOT NULL,
    timeframe           VARCHAR(10) NOT NULL,
    ichimoku_tenkan     DECIMAL(20, 8),
    ichimoku_kijun      DECIMAL(20, 8),
    ichimoku_senkou_a   DECIMAL(20, 8),
    ichimoku_senkou_b   DECIMAL(20, 8),
    ichimoku_chikou     DECIMAL(20, 8),
    supertrend          DECIMAL(20, 8),
    supertrend_direction INTEGER,
    psar                DECIMAL(20, 8),
    psar_direction      INTEGER,
    aroon_up            DECIMAL(10, 4),
    aroon_down          DECIMAL(10, 4),
    aroon_oscillator    DECIMAL(10, 4),
    linreg_slope        DECIMAL(20, 8),
    linreg_intercept    DECIMAL(20, 8),
    linreg_r_squared    DECIMAL(10, 6),
    ht_trendmode        INTEGER,
    source              VARCHAR(20) NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_trend', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_indicators_trend_lookup
    ON indicators_trend (symbol, timeframe, timestamp DESC);

-- Volume Indicators (optimized table)
CREATE TABLE IF NOT EXISTS indicators_volume (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    obv             DECIMAL(30, 8),
    ad_line         DECIMAL(30, 8),
    ad_oscillator   DECIMAL(20, 8),
    chaikin_mf      DECIMAL(10, 6),
    volume_sma_20   DECIMAL(30, 8),
    volume_ratio    DECIMAL(10, 4),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_volume', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_indicators_volume_lookup
    ON indicators_volume (symbol, timeframe, timestamp DESC);

-- Pivot Points & Levels (optimized table)
CREATE TABLE IF NOT EXISTS indicators_levels (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    pivot           DECIMAL(20, 8),
    r1              DECIMAL(20, 8),
    r2              DECIMAL(20, 8),
    r3              DECIMAL(20, 8),
    s1              DECIMAL(20, 8),
    s2              DECIMAL(20, 8),
    s3              DECIMAL(20, 8),
    fib_r1          DECIMAL(20, 8),
    fib_r2          DECIMAL(20, 8),
    fib_r3          DECIMAL(20, 8),
    fib_s1          DECIMAL(20, 8),
    fib_s2          DECIMAL(20, 8),
    fib_s3          DECIMAL(20, 8),
    cam_r1          DECIMAL(20, 8),
    cam_r2          DECIMAL(20, 8),
    cam_r3          DECIMAL(20, 8),
    cam_r4          DECIMAL(20, 8),
    cam_s1          DECIMAL(20, 8),
    cam_s2          DECIMAL(20, 8),
    cam_s3          DECIMAL(20, 8),
    cam_s4          DECIMAL(20, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, timeframe)
);

SELECT create_hypertable('indicators_levels', 'timestamp',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_indicators_levels_lookup
    ON indicators_levels (symbol, timeframe, timestamp DESC);

-- =====================================================
-- MARKET SNAPSHOTS (Real-time data)
-- =====================================================

CREATE TABLE IF NOT EXISTS market_snapshots (
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    bid             DECIMAL(20, 8),
    ask             DECIMAL(20, 8),
    last_price      DECIMAL(20, 8) NOT NULL,
    volume          DECIMAL(30, 8),
    spread          DECIMAL(20, 8),
    day_open        DECIMAL(20, 8),
    day_high        DECIMAL(20, 8),
    day_low         DECIMAL(20, 8),
    prev_close      DECIMAL(20, 8),
    source          VARCHAR(20) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('market_snapshots', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_market_snapshots_symbol
    ON market_snapshots (symbol, timestamp DESC);

-- =====================================================
-- SYMBOLS METADATA
-- =====================================================

CREATE TABLE IF NOT EXISTS symbols (
    symbol              VARCHAR(20) PRIMARY KEY,
    display_name        VARCHAR(100),
    category            VARCHAR(20) NOT NULL,
    subcategory         VARCHAR(20),
    base_currency       VARCHAR(10),
    quote_currency      VARCHAR(10),
    twelvedata_symbol   VARCHAR(20),
    easyinsight_symbol  VARCHAR(20),
    yfinance_symbol     VARCHAR(20),
    pip_value           DECIMAL(10, 6),
    min_lot_size        DECIMAL(10, 4),
    max_lot_size        DECIMAL(10, 2),
    lot_step            DECIMAL(10, 4),
    is_active           BOOLEAN DEFAULT TRUE,
    is_favorite         BOOLEAN DEFAULT FALSE,
    first_data_at       TIMESTAMPTZ,
    last_data_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_symbols_category ON symbols (category, subcategory);
CREATE INDEX IF NOT EXISTS idx_symbols_active ON symbols (is_active) WHERE is_active = TRUE;

-- =====================================================
-- DATA FRESHNESS TRACKING
-- =====================================================

CREATE TABLE IF NOT EXISTS data_freshness (
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,
    data_type       VARCHAR(30) NOT NULL,
    last_updated    TIMESTAMPTZ NOT NULL,
    last_timestamp  TIMESTAMPTZ NOT NULL,
    record_count    BIGINT DEFAULT 0,
    source          VARCHAR(20),
    PRIMARY KEY (symbol, timeframe, data_type)
);

CREATE INDEX IF NOT EXISTS idx_freshness_updated ON data_freshness (last_updated);

-- =====================================================
-- EXTERNAL DATA SOURCES (for RAG)
-- =====================================================

-- Economic Calendar
CREATE TABLE IF NOT EXISTS economic_events (
    id              SERIAL,
    timestamp       TIMESTAMPTZ NOT NULL,
    event_name      VARCHAR(200) NOT NULL,
    country         VARCHAR(10) NOT NULL,
    currency        VARCHAR(10),
    importance      VARCHAR(10),
    actual          VARCHAR(50),
    forecast        VARCHAR(50),
    previous        VARCHAR(50),
    source          VARCHAR(50),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable('economic_events', 'timestamp',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_economic_events_lookup
    ON economic_events (country, timestamp DESC);

-- Sentiment Data
CREATE TABLE IF NOT EXISTS sentiment_data (
    timestamp           TIMESTAMPTZ NOT NULL,
    symbol              VARCHAR(20),
    fear_greed_index    INTEGER,
    social_sentiment    DECIMAL(5, 2),
    news_sentiment      DECIMAL(5, 2),
    vix                 DECIMAL(10, 4),
    put_call_ratio      DECIMAL(5, 3),
    raw_data            JSONB,
    source              VARCHAR(50),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, COALESCE(symbol, 'MARKET'))
);

SELECT create_hypertable('sentiment_data', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- On-Chain Data (Crypto)
CREATE TABLE IF NOT EXISTS onchain_data (
    timestamp           TIMESTAMPTZ NOT NULL,
    symbol              VARCHAR(20) NOT NULL,
    whale_transactions  INTEGER,
    large_tx_volume     DECIMAL(30, 8),
    exchange_inflow     DECIMAL(30, 8),
    exchange_outflow    DECIMAL(30, 8),
    exchange_netflow    DECIMAL(30, 8),
    hash_rate           DECIMAL(30, 2),
    difficulty          DECIMAL(30, 2),
    tvl                 DECIMAL(30, 2),
    raw_data            JSONB,
    source              VARCHAR(50),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable('onchain_data', 'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_onchain_data_symbol
    ON onchain_data (symbol, timestamp DESC);

-- =====================================================
-- RETENTION POLICIES
-- =====================================================

-- M1: 30 days retention
SELECT add_retention_policy('ohlcv_m1', INTERVAL '30 days', if_not_exists => TRUE);

-- M5: 90 days retention
SELECT add_retention_policy('ohlcv_m5', INTERVAL '90 days', if_not_exists => TRUE);

-- Indicators JSONB: 30 days retention
SELECT add_retention_policy('indicators', INTERVAL '30 days', if_not_exists => TRUE);

-- Market snapshots: 7 days retention
SELECT add_retention_policy('market_snapshots', INTERVAL '7 days', if_not_exists => TRUE);

-- =====================================================
-- COMPRESSION POLICIES
-- =====================================================

-- Enable compression on OHLCV tables
ALTER TABLE ohlcv_m1 SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('ohlcv_m1', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE ohlcv_m5 SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('ohlcv_m5', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE ohlcv_m15 SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('ohlcv_m15', INTERVAL '14 days', if_not_exists => TRUE);

ALTER TABLE ohlcv_h1 SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('ohlcv_h1', INTERVAL '30 days', if_not_exists => TRUE);

ALTER TABLE ohlcv_d1 SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('ohlcv_d1', INTERVAL '90 days', if_not_exists => TRUE);

-- Enable compression on indicator tables
ALTER TABLE indicators SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,timeframe,indicator_name'
);
SELECT add_compression_policy('indicators', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE indicators_ma SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,timeframe'
);
SELECT add_compression_policy('indicators_ma', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE indicators_momentum SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,timeframe'
);
SELECT add_compression_policy('indicators_momentum', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE indicators_volatility SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,timeframe'
);
SELECT add_compression_policy('indicators_volatility', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE indicators_trend SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,timeframe'
);
SELECT add_compression_policy('indicators_trend', INTERVAL '7 days', if_not_exists => TRUE);

-- =====================================================
-- DONE
-- =====================================================

-- Grant permissions to trading user
GRANT ALL ON ALL TABLES IN SCHEMA public TO trading;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO trading;
