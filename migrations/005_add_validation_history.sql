-- Validation History Tables Migration
-- Version: 005
-- Date: 2026-01-19
-- Description: Adds tables for persistent validation run history

-- =====================================================
-- VALIDATION RUNS TABLE
-- Stores metadata for each validation run
-- =====================================================

CREATE TABLE IF NOT EXISTS validation_runs (
    id              SERIAL PRIMARY KEY,
    run_id          UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    status          VARCHAR(20) NOT NULL DEFAULT 'running',  -- running, completed, failed, aborted
    triggered_by    VARCHAR(50) DEFAULT 'manual',  -- manual, scheduled, api

    -- Summary counts
    total_tests     INTEGER DEFAULT 0,
    tests_ok        INTEGER DEFAULT 0,
    tests_warning   INTEGER DEFAULT 0,
    tests_error     INTEGER DEFAULT 0,
    success_rate    DECIMAL(5, 2),

    -- Component results (JSONB for flexibility)
    results         JSONB,

    -- Error details
    error_details   JSONB,

    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_validation_runs_started_at ON validation_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_validation_runs_status ON validation_runs (status);
CREATE INDEX IF NOT EXISTS idx_validation_runs_run_id ON validation_runs (run_id);

-- =====================================================
-- VALIDATION COMPONENT RESULTS TABLE
-- Stores detailed results per component per run
-- =====================================================

CREATE TABLE IF NOT EXISTS validation_component_results (
    id              SERIAL PRIMARY KEY,
    run_id          UUID NOT NULL REFERENCES validation_runs(run_id) ON DELETE CASCADE,
    component       VARCHAR(50) NOT NULL,  -- twelvedata, easyinsight, yfinance, timescaledb, redis, endpoints
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    status          VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending, running, ok, warning, error

    -- Result counts
    tests_ok        INTEGER DEFAULT 0,
    tests_warning   INTEGER DEFAULT 0,
    tests_error     INTEGER DEFAULT 0,

    -- Detailed results
    details         JSONB,

    created_at      TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(run_id, component)
);

CREATE INDEX IF NOT EXISTS idx_validation_component_run ON validation_component_results (run_id);
CREATE INDEX IF NOT EXISTS idx_validation_component_name ON validation_component_results (component);

-- =====================================================
-- VALIDATION ERRORS TABLE
-- Stores individual validation errors for analysis
-- =====================================================

CREATE TABLE IF NOT EXISTS validation_errors (
    id              SERIAL PRIMARY KEY,
    run_id          UUID NOT NULL REFERENCES validation_runs(run_id) ON DELETE CASCADE,
    component       VARCHAR(50) NOT NULL,
    error_type      VARCHAR(50) NOT NULL,  -- connection, data_quality, timeout, api_error, etc.
    severity        VARCHAR(20) NOT NULL DEFAULT 'error',  -- warning, error
    symbol          VARCHAR(20),
    timeframe       VARCHAR(10),
    message         TEXT NOT NULL,
    details         JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_validation_errors_run ON validation_errors (run_id);
CREATE INDEX IF NOT EXISTS idx_validation_errors_component ON validation_errors (component);
CREATE INDEX IF NOT EXISTS idx_validation_errors_symbol ON validation_errors (symbol) WHERE symbol IS NOT NULL;

-- =====================================================
-- HELPER FUNCTION: Clean old validation runs
-- Keeps last 100 runs by default
-- =====================================================

CREATE OR REPLACE FUNCTION cleanup_old_validation_runs(keep_count INTEGER DEFAULT 100)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    WITH old_runs AS (
        SELECT id FROM validation_runs
        ORDER BY started_at DESC
        OFFSET keep_count
    )
    DELETE FROM validation_runs WHERE id IN (SELECT id FROM old_runs);

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- DONE
-- =====================================================
