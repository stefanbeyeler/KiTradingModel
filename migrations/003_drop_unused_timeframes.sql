-- Migration: Drop unused timeframe tables (H2, M45)
-- Date: 2026-01-09
-- Description: Remove ohlcv_h2 and ohlcv_m45 tables as they are not used in practice
--              and contain no data.

-- Drop hypertables for unused timeframes
DROP TABLE IF EXISTS ohlcv_h2 CASCADE;
DROP TABLE IF EXISTS ohlcv_m45 CASCADE;

-- Clean up data_freshness entries for these timeframes
DELETE FROM data_freshness WHERE timeframe IN ('H2', 'M45', 'h2', 'm45');

-- Log the migration
DO $$
BEGIN
    RAISE NOTICE 'Migration 003: Dropped ohlcv_h2 and ohlcv_m45 tables';
END $$;
