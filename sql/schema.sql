-- Run automatically by ingestion/load.py. Idempotent.
CREATE SCHEMA IF NOT EXISTS raw;        -- validated source data
CREATE SCHEMA IF NOT EXISTS audit;      -- pipeline governance tables
CREATE SCHEMA IF NOT EXISTS staging;    -- dbt
CREATE SCHEMA IF NOT EXISTS analytics;  -- dbt marts, ML outputs, Power BI source

CREATE TABLE IF NOT EXISTS audit.pipeline_runs (
    run_id            TEXT PRIMARY KEY,
    started_at        TIMESTAMPTZ NOT NULL,
    finished_at       TIMESTAMPTZ,
    status            TEXT NOT NULL,           -- running | success | failed
    records_read      BIGINT,
    records_inserted  BIGINT,
    records_rejected  BIGINT,
    validation_failures INT,                   -- number of rules with failures
    error_details     TEXT
);

CREATE TABLE IF NOT EXISTS audit.validation_results (
    run_id          TEXT REFERENCES audit.pipeline_runs(run_id),
    table_name      TEXT,
    rule            TEXT,
    severity        TEXT,
    records_checked BIGINT,
    records_failed  BIGINT,
    action          TEXT
);

CREATE TABLE IF NOT EXISTS audit.rejected_records (
    run_id      TEXT REFERENCES audit.pipeline_runs(run_id),
    table_name  TEXT,
    rule        TEXT,
    record_key  TEXT,
    record_json TEXT
);
