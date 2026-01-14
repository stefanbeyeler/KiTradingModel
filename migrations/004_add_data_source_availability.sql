-- TimescaleDB Migration: Add data source availability columns
-- Version: 004
-- Date: 2026-01-14
-- Description: Adds columns for explicit data source availability flags

-- =====================================================
-- EXTEND SYMBOLS TABLE
-- =====================================================

-- Add TwelveData availability flag
-- NULL = auto-detect, TRUE = available, FALSE = use fallback
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS twelvedata_available BOOLEAN;

-- Add EasyInsight availability flag
-- NULL = auto-detect, TRUE = available, FALSE = use fallback
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS easyinsight_available BOOLEAN;

-- Add preferred data source
-- NULL = auto, 'twelvedata', 'easyinsight', 'yfinance'
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS preferred_data_source VARCHAR(20);

-- =====================================================
-- CREATE INDEXES
-- =====================================================

-- Index for symbols where TwelveData is explicitly disabled
CREATE INDEX IF NOT EXISTS idx_symbols_td_unavailable
    ON symbols (twelvedata_available)
    WHERE twelvedata_available = FALSE;

-- Index for symbols where EasyInsight is explicitly disabled
CREATE INDEX IF NOT EXISTS idx_symbols_ei_unavailable
    ON symbols (easyinsight_available)
    WHERE easyinsight_available = FALSE;

-- =====================================================
-- ADD CONSTRAINT FOR preferred_data_source
-- =====================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'symbols_preferred_source_check'
    ) THEN
        ALTER TABLE symbols ADD CONSTRAINT symbols_preferred_source_check
            CHECK (preferred_data_source IS NULL OR preferred_data_source IN ('twelvedata', 'easyinsight', 'yfinance'));
    END IF;
END $$;

-- =====================================================
-- SET DEFAULT VALUES FOR INDEX SYMBOLS
-- =====================================================

-- Automatically mark known index symbols as TwelveData unavailable
UPDATE symbols
SET twelvedata_available = FALSE
WHERE category = 'index'
  AND twelvedata_available IS NULL;

-- =====================================================
-- DONE
-- =====================================================
