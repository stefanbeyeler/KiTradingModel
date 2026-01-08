-- TimescaleDB Migration: Extend symbols table
-- Version: 002
-- Date: 2026-01-08
-- Description: Adds missing columns to symbols table for full ManagedSymbol support

-- =====================================================
-- EXTEND SYMBOLS TABLE
-- =====================================================

-- Add status column
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'active';

-- Add description column
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS description TEXT;

-- Add lot_step if missing (already in schema but verify)
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS lot_step DECIMAL(10, 4);

-- Add data availability columns
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS has_timescaledb_data BOOLEAN DEFAULT FALSE;
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS total_records BIGINT DEFAULT 0;

-- Add model status columns
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS has_nhits_model BOOLEAN DEFAULT FALSE;
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS nhits_model_trained_at TIMESTAMPTZ;

-- Add user preference columns
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS notes TEXT;
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS tags TEXT[] DEFAULT '{}';
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS aliases TEXT[] DEFAULT '{}';

-- Add yfinance_symbol if missing
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS yfinance_symbol VARCHAR(30);

-- =====================================================
-- CREATE INDEXES
-- =====================================================

CREATE INDEX IF NOT EXISTS idx_symbols_status ON symbols (status) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_symbols_favorite ON symbols (is_favorite) WHERE is_favorite = TRUE;
CREATE INDEX IF NOT EXISTS idx_symbols_has_data ON symbols (has_timescaledb_data) WHERE has_timescaledb_data = TRUE;

-- =====================================================
-- UPDATE CONSTRAINTS
-- =====================================================

-- Add check constraint for status
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'symbols_status_check'
    ) THEN
        ALTER TABLE symbols ADD CONSTRAINT symbols_status_check
            CHECK (status IN ('active', 'inactive', 'suspended'));
    END IF;
END $$;

-- =====================================================
-- DONE
-- =====================================================
