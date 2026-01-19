-- Service Runs and Prediction History Tables Migration
-- Version: 006
-- Date: 2026-01-19
-- Description: Adds generic service_runs table and prediction_history for all microservices

-- =====================================================
-- SERVICE RUNS TABLE (Generisch für alle Services)
-- Ersetzt/erweitert validation_runs für allgemeine Nutzung
-- =====================================================

CREATE TABLE IF NOT EXISTS service_runs (
    id              SERIAL PRIMARY KEY,
    run_id          UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),

    -- Service identification
    service         VARCHAR(50) NOT NULL,   -- data, nhits, tcn, hmm, candlestick, cnn-lstm, rag, llm, workplace
    run_type        VARCHAR(50) NOT NULL,   -- validation, prediction, training, analysis, scan

    -- Optional context
    symbol          VARCHAR(20),            -- Für symbolspezifische Runs
    timeframe       VARCHAR(10),            -- Für timeframe-spezifische Runs

    -- Timing
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    duration_ms     INTEGER,                -- Dauer in Millisekunden

    -- Status
    status          VARCHAR(20) NOT NULL DEFAULT 'running',  -- running, completed, failed, aborted
    triggered_by    VARCHAR(50) DEFAULT 'manual',            -- manual, scheduled, api, auto-scan

    -- Summary counts (optional, für Validierungen)
    total_items     INTEGER DEFAULT 0,
    items_ok        INTEGER DEFAULT 0,
    items_warning   INTEGER DEFAULT 0,
    items_error     INTEGER DEFAULT 0,
    success_rate    DECIMAL(5, 2),

    -- Flexible Daten (JSONB für maximale Flexibilität)
    input_params    JSONB,                  -- Eingabeparameter
    results         JSONB,                  -- Ergebnisse
    metrics         JSONB,                  -- Metriken (Accuracy, Confidence, etc.)
    error_details   JSONB,                  -- Fehlerdetails

    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indizes für häufige Abfragen
CREATE INDEX IF NOT EXISTS idx_service_runs_service ON service_runs (service);
CREATE INDEX IF NOT EXISTS idx_service_runs_type ON service_runs (run_type);
CREATE INDEX IF NOT EXISTS idx_service_runs_started ON service_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_service_runs_status ON service_runs (status);
CREATE INDEX IF NOT EXISTS idx_service_runs_symbol ON service_runs (symbol) WHERE symbol IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_service_runs_service_type ON service_runs (service, run_type);
CREATE INDEX IF NOT EXISTS idx_service_runs_service_symbol ON service_runs (service, symbol) WHERE symbol IS NOT NULL;

-- =====================================================
-- PREDICTION HISTORY TABLE
-- Speichert individuelle Vorhersagen für spätere Evaluierung
-- =====================================================

CREATE TABLE IF NOT EXISTS prediction_history (
    id              SERIAL PRIMARY KEY,
    prediction_id   UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),

    -- Service & Context
    service         VARCHAR(50) NOT NULL,   -- nhits, tcn, hmm, candlestick, cnn-lstm, workplace
    symbol          VARCHAR(20) NOT NULL,
    timeframe       VARCHAR(10) NOT NULL,

    -- Timing
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    target_time     TIMESTAMPTZ,            -- Wann die Vorhersage eintreffen soll
    horizon         VARCHAR(20),            -- 1h, 4h, 1d, 1w, etc.

    -- Prediction Data
    prediction_type VARCHAR(50) NOT NULL,   -- price, direction, pattern, regime, signal
    prediction      JSONB NOT NULL,         -- Die eigentliche Vorhersage
    confidence      DECIMAL(5, 4),          -- 0.0000 - 1.0000

    -- Model Info
    model_version   VARCHAR(50),            -- Modell-Version
    model_params    JSONB,                  -- Verwendete Parameter

    -- Input Features (für Reproduzierbarkeit)
    input_features  JSONB,                  -- Wichtigste Eingabe-Features

    -- Evaluation (später ausgefüllt)
    actual_outcome  JSONB,                  -- Tatsächliches Ergebnis
    evaluated_at    TIMESTAMPTZ,            -- Wann evaluiert
    is_correct      BOOLEAN,                -- War Vorhersage korrekt?
    accuracy_score  DECIMAL(5, 4),          -- Genauigkeitswert
    error_amount    DECIMAL(20, 8),         -- Abweichung (für numerische Vorhersagen)

    -- Metadata
    triggered_by    VARCHAR(50) DEFAULT 'api',  -- api, scheduled, manual, scan
    tags            TEXT[],                     -- Flexible Tags für Filterung
    notes           TEXT,                       -- Optionale Notizen

    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indizes für häufige Abfragen
CREATE INDEX IF NOT EXISTS idx_prediction_service ON prediction_history (service);
CREATE INDEX IF NOT EXISTS idx_prediction_symbol ON prediction_history (symbol);
CREATE INDEX IF NOT EXISTS idx_prediction_timeframe ON prediction_history (timeframe);
CREATE INDEX IF NOT EXISTS idx_prediction_type ON prediction_history (prediction_type);
CREATE INDEX IF NOT EXISTS idx_prediction_time ON prediction_history (predicted_at DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_target ON prediction_history (target_time) WHERE target_time IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_prediction_evaluated ON prediction_history (is_correct) WHERE evaluated_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_prediction_service_symbol ON prediction_history (service, symbol);
CREATE INDEX IF NOT EXISTS idx_prediction_symbol_timeframe ON prediction_history (symbol, timeframe);

-- Hypertable für TimescaleDB (optional, für sehr grosse Datenmengen)
-- SELECT create_hypertable('prediction_history', 'predicted_at',
--     chunk_time_interval => INTERVAL '7 days',
--     if_not_exists => TRUE
-- );

-- =====================================================
-- PREDICTION AGGREGATES VIEW
-- Aggregierte Statistiken pro Service/Symbol
-- =====================================================

CREATE OR REPLACE VIEW prediction_stats AS
SELECT
    service,
    symbol,
    timeframe,
    prediction_type,
    COUNT(*) as total_predictions,
    COUNT(*) FILTER (WHERE evaluated_at IS NOT NULL) as evaluated_count,
    COUNT(*) FILTER (WHERE is_correct = true) as correct_count,
    ROUND(
        (COUNT(*) FILTER (WHERE is_correct = true)::DECIMAL /
         NULLIF(COUNT(*) FILTER (WHERE evaluated_at IS NOT NULL), 0) * 100), 2
    ) as accuracy_percent,
    ROUND(AVG(confidence)::DECIMAL, 4) as avg_confidence,
    ROUND(AVG(accuracy_score) FILTER (WHERE accuracy_score IS NOT NULL)::DECIMAL, 4) as avg_accuracy_score,
    MIN(predicted_at) as first_prediction,
    MAX(predicted_at) as last_prediction
FROM prediction_history
GROUP BY service, symbol, timeframe, prediction_type;

-- =====================================================
-- SERVICE RUNS AGGREGATES VIEW
-- Aggregierte Statistiken pro Service
-- =====================================================

CREATE OR REPLACE VIEW service_runs_stats AS
SELECT
    service,
    run_type,
    COUNT(*) as total_runs,
    COUNT(*) FILTER (WHERE status = 'completed') as completed_runs,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_runs,
    ROUND(AVG(success_rate) FILTER (WHERE success_rate IS NOT NULL)::DECIMAL, 2) as avg_success_rate,
    ROUND(AVG(duration_ms) FILTER (WHERE duration_ms IS NOT NULL)::DECIMAL, 0) as avg_duration_ms,
    MIN(started_at) as first_run,
    MAX(started_at) as last_run
FROM service_runs
GROUP BY service, run_type;

-- =====================================================
-- HELPER FUNCTIONS
-- =====================================================

-- Cleanup alte Service-Runs (behält letzte N pro Service/Type)
CREATE OR REPLACE FUNCTION cleanup_old_service_runs(
    p_service VARCHAR DEFAULT NULL,
    p_run_type VARCHAR DEFAULT NULL,
    p_keep_count INTEGER DEFAULT 100
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
    rec RECORD;
BEGIN
    FOR rec IN
        SELECT DISTINCT service, run_type
        FROM service_runs
        WHERE (p_service IS NULL OR service = p_service)
          AND (p_run_type IS NULL OR run_type = p_run_type)
    LOOP
        WITH old_runs AS (
            SELECT id FROM service_runs
            WHERE service = rec.service AND run_type = rec.run_type
            ORDER BY started_at DESC
            OFFSET p_keep_count
        )
        DELETE FROM service_runs WHERE id IN (SELECT id FROM old_runs);

        deleted_count := deleted_count + (SELECT COUNT(*) FROM old_runs);
    END LOOP;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Cleanup alte Predictions (behält letzte N Tage)
CREATE OR REPLACE FUNCTION cleanup_old_predictions(
    p_days_to_keep INTEGER DEFAULT 90
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM prediction_history
    WHERE predicted_at < NOW() - (p_days_to_keep || ' days')::INTERVAL;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Markiere Predictions zur Evaluierung fällig
CREATE OR REPLACE FUNCTION get_predictions_due_for_evaluation(
    p_service VARCHAR DEFAULT NULL,
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
    prediction_id UUID,
    service VARCHAR,
    symbol VARCHAR,
    timeframe VARCHAR,
    prediction_type VARCHAR,
    prediction JSONB,
    target_time TIMESTAMPTZ,
    predicted_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ph.prediction_id,
        ph.service,
        ph.symbol,
        ph.timeframe,
        ph.prediction_type,
        ph.prediction,
        ph.target_time,
        ph.predicted_at
    FROM prediction_history ph
    WHERE ph.evaluated_at IS NULL
      AND ph.target_time IS NOT NULL
      AND ph.target_time <= NOW()
      AND (p_service IS NULL OR ph.service = p_service)
    ORDER BY ph.target_time ASC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- DONE
-- =====================================================
