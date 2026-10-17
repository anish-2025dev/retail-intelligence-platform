"""Load: extract -> validate -> load clean data to raw.* in PostgreSQL, with a full audit trail.

Usage:
    python -m ingestion.load                 # needs PostgreSQL (docker compose up -d postgres)
    python -m ingestion.load --dry-run       # no database: writes reports to outputs/
"""
from __future__ import annotations

import argparse
import logging
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_quality.validation import validate
from ingestion.config import OUTPUT_DIR, RAW_DIR, ROOT, database_url
from ingestion.extract import extract

log = logging.getLogger("load")

TABLE_NAMES = {"payments": "payments", "reviews": "reviews"}  # logical -> raw table name (same here)


def _run_sql_file(engine, path: Path) -> None:
    from sqlalchemy import text
    with engine.begin() as conn:
        for stmt in [s.strip() for s in path.read_text().split(";") if s.strip()]:
            conn.execute(text(stmt))


def run(data_dir: Path = RAW_DIR, dry_run: bool = False, run_id: str | None = None) -> dict:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    started = datetime.now(timezone.utc)
    engine = None
    summary = {"run_id": run_id, "status": "running"}
    try:
        if not dry_run:
            from sqlalchemy import create_engine, text
            engine = create_engine(database_url())
            _run_sql_file(engine, ROOT / "sql" / "schema.sql")
            with engine.begin() as conn:
                conn.execute(text("INSERT INTO audit.pipeline_runs (run_id, started_at, status) "
                                  "VALUES (:r, :s, 'running')"), {"r": run_id, "s": started})

        frames = extract(data_dir)
        read = sum(len(v) for k, v in frames.items() if k != "category_translation")
        optional = {k: frames.pop(k) for k in ["category_translation"] if k in frames}

        clean, report, rejected = validate(frames)
        inserted = sum(len(v) for v in clean.values())
        n_rejected = len(rejected)
        failing_rules = int((report.records_failed > 0).sum())

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        report.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)
        rejected.to_csv(OUTPUT_DIR / "rejected_records.csv", index=False)

        if not dry_run:
            with engine.begin() as conn:  # full refresh of raw schema in one transaction
                for t in list(clean) + list(optional):
                    conn.exec_driver_sql(f"DROP TABLE IF EXISTS raw.{t} CASCADE")
                for t, df in {**clean, **optional}.items():
                    df.to_sql(t, conn, schema="raw", index=False, method="multi", chunksize=5000)
            _run_sql_file(engine, ROOT / "sql" / "constraints_and_indexes.sql")
            report.assign(run_id=run_id).rename(columns={"table": "table_name", "records_checked": "records_checked"}) \
                  [["run_id", "table_name", "rule", "severity", "records_checked", "records_failed", "action"]] \
                  .to_sql("validation_results", engine, schema="audit", if_exists="append", index=False)
            if n_rejected:
                rejected.assign(run_id=run_id).to_sql("rejected_records", engine, schema="audit",
                                                      if_exists="append", index=False, chunksize=5000)
            from sqlalchemy import text
            with engine.begin() as conn:
                conn.execute(text(
                    "UPDATE audit.pipeline_runs SET finished_at=:f, status='success', records_read=:r, "
                    "records_inserted=:i, records_rejected=:x, validation_failures=:v WHERE run_id=:id"),
                    {"f": datetime.now(timezone.utc), "r": read, "i": inserted, "x": n_rejected, "v": failing_rules, "id": run_id})

        summary.update(status="success", records_read=read, records_inserted=inserted,
                       records_rejected=n_rejected, rules_with_failures=failing_rules)
        log.info("run %s finished: %s", run_id, summary)
        return summary

    except Exception as exc:  # record the failure, then re-raise so Airflow marks the task failed
        log.exception("pipeline failed")
        if engine is not None:
            try:  # best-effort: if the database itself is the problem, don't let this mask the original error
                from sqlalchemy import text
                with engine.begin() as conn:
                    conn.execute(text("UPDATE audit.pipeline_runs SET finished_at=:f, status='failed', error_details=:e "
                                      "WHERE run_id=:id"),
                                 {"f": datetime.now(timezone.utc), "e": traceback.format_exc()[-4000:], "id": run_id})
            except Exception:  # noqa: BLE001
                log.warning("could not record failure in audit.pipeline_runs", exc_info=True)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(RAW_DIR))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    print(run(Path(a.data_dir), dry_run=a.dry_run))
