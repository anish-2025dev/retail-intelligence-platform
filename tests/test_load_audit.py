"""Tests for ingestion.load / ingestion.report against a *fake* SQLAlchemy engine (no PostgreSQL available).

They verify control flow, SQL/DDL consistency, transaction boundaries and audit behaviour - not real SQL execution.
"""
import contextlib
import pathlib
import re
import sys
import types
from unittest import mock

import pandas as pd

from data_quality.validation import validate
from ingestion import load, report
from ingestion.extract import extract
from ingestion.generate_sample_data import generate

ROOT = pathlib.Path(__file__).resolve().parents[1]


class FakeConn:
    def __init__(self, log, tx): self.log, self.tx = log, tx
    def execute(self, stmt, params=None):
        self.log.append(("execute", str(stmt), params, self.tx))
        if self.log.fail_on and self.log.fail_on in str(stmt):
            raise RuntimeError("db down")
    def exec_driver_sql(self, sql): self.log.append(("driver_sql", sql, None, self.tx))


class Log(list):
    fail_on = None


class FakeEngine:
    def __init__(self, log): self.log, self.n = log, 0
    @contextlib.contextmanager
    def begin(self):
        self.n += 1
        yield FakeConn(self.log, self.n)


def _run(tmp_path, fail_on=None, data_dir=None, **kw):
    """Run load.run() against the fake engine; returns (result_or_exc, sql_log, to_sql_calls)."""
    raw = tmp_path / "raw"
    if data_dir is None:
        generate(raw, n_customers=150, n_orders=250); data_dir = raw
    log = Log(); log.fail_on = fail_on
    engine = FakeEngine(log)
    fake_sa = types.SimpleNamespace(create_engine=lambda url: engine, text=lambda s: s)
    calls = []

    def fake_to_sql(self, name, con, **kwargs):
        calls.append({"name": name, "con": con, "df": self.copy(), **kwargs})

    with mock.patch.dict(sys.modules, {"sqlalchemy": fake_sa}), \
         mock.patch.object(pd.DataFrame, "to_sql", fake_to_sql), \
         mock.patch.object(load, "OUTPUT_DIR", tmp_path / "out"):
        try:
            result = load.run(pathlib.Path(data_dir), **kw)
        except Exception as e:  # noqa: BLE001
            result = e
    return result, log, calls


def test_successful_load_writes_everything_in_the_right_order(tmp_path):
    result, log, calls = _run(tmp_path)
    assert result["status"] == "success"
    stmts = [s for _, s, _, _ in log]
    kinds = [c["name"] for c in calls]

    # schema bootstrap happens before the run is registered
    i_schema = next(i for i, s in enumerate(stmts) if "CREATE SCHEMA IF NOT EXISTS raw" in s)
    i_start = next(i for i, s in enumerate(stmts) if "INSERT INTO audit.pipeline_runs" in s)
    i_constraints = next(i for i, s in enumerate(stmts) if "ADD PRIMARY KEY" in s)
    i_done = next(i for i, s in enumerate(stmts) if "status='success'" in s)
    assert i_schema < i_start < i_constraints < i_done                      # success is recorded last

    raw_tables = {c["name"] for c in calls if c.get("schema") == "raw"}
    assert raw_tables == {"customers", "orders", "order_items", "payments", "reviews", "products", "sellers", "category_translation"}
    assert "validation_results" in kinds and "rejected_records" in kinds
    dropped = {re.search(r"raw\.(\w+)", s).group(1) for k, s, _, _ in log if k == "driver_sql"}
    assert dropped == raw_tables


def test_raw_refresh_is_one_transaction(tmp_path):
    _, log, calls = _run(tmp_path)
    drop_tx = {tx for k, s, _, tx in log if k == "driver_sql"}
    load_tx = {c["con"].tx for c in calls if c.get("schema") == "raw"}
    assert len(drop_tx) == 1 and drop_tx == load_tx, "DROP + reload must share a transaction so a failure can't leave raw empty"


def test_audit_numbers_match_the_validator(tmp_path):
    result, log, calls = _run(tmp_path)
    frames = extract(tmp_path / "raw"); frames.pop("category_translation")
    clean, rep, rejected = validate(frames)
    assert result["records_read"] == sum(len(v) for v in frames.values())
    assert result["records_inserted"] == sum(len(v) for v in clean.values())
    assert result["records_rejected"] == len(rejected)
    assert result["records_read"] - result["records_inserted"] == result["records_rejected"]
    update = next(p for k, s, p, _ in log if "status='success'" in s)
    assert update["r"] == result["records_read"] and update["x"] == result["records_rejected"] and update["id"] == result["run_id"]
    assert update["v"] == int((rep.records_failed > 0).sum())


def test_audit_frames_match_the_ddl(tmp_path):
    _, log, calls = _run(tmp_path)
    ddl = (ROOT / "sql" / "schema.sql").read_text()
    def table_cols(name):
        body = re.search(rf"CREATE TABLE IF NOT EXISTS audit\.{name} \((.*?)\n\);", ddl, re.S).group(1)
        return {line.split()[0] for line in body.strip().splitlines() if line.strip() and not line.strip().startswith("--")}
    for name in ("validation_results", "rejected_records"):
        frame = next(c["df"] for c in calls if c["name"] == name)
        assert set(frame.columns) <= table_cols(name), f"{name}: {set(frame.columns) - table_cols(name)}"
        assert (frame.run_id == frame.run_id.iloc[0]).all()
    vr = next(c["df"] for c in calls if c["name"] == "validation_results")
    assert list(vr.columns) == ["run_id", "table_name", "rule", "severity", "records_checked", "records_failed", "action"]
    # every column named in the UPDATE/INSERT statements exists in audit.pipeline_runs
    runs_cols = table_cols("pipeline_runs")
    for _, s, _, _ in log:
        if "UPDATE audit.pipeline_runs" in s:
            assigned = set(re.findall(r"(\w+)\s*=\s*:", s.split("WHERE")[0]))
            assert assigned <= runs_cols, assigned - runs_cols


def test_sql_files_are_safe_for_sqlalchemy_text():
    # text() treats ':name' as a bind parameter, and load splits files on ';'
    for f in ("schema.sql", "constraints_and_indexes.sql"):
        sql = (ROOT / "sql" / f).read_text()
        assert not re.search(r"(?<!:):[A-Za-z_]", sql), f"{f}: contains ':name' that SQLAlchemy would treat as a parameter"
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            body = "\n".join(l for l in stmt.splitlines() if not l.strip().startswith("--")).strip()
            assert body.split()[0].upper() in {"CREATE", "ALTER"}, f"{f}: unexpected statement start {body[:40]!r}"


def test_failure_marks_run_failed_and_reraises(tmp_path):
    result, log, _ = _run(tmp_path, data_dir=tmp_path / "does-not-exist")
    assert isinstance(result, FileNotFoundError)
    failed = [(s, p) for _, s, p, _ in log if "status='failed'" in s]
    assert len(failed) == 1 and "FileNotFoundError" in failed[0][1]["e"]
    assert not any("status='success'" in s for _, s, _, _ in log)


def test_database_outage_surfaces_the_original_error(tmp_path):
    result, log, _ = _run(tmp_path, fail_on="CREATE SCHEMA")           # engine unusable from the first statement
    assert isinstance(result, RuntimeError) and "db down" in str(result), f"original error was masked by: {result!r}"


def test_audit_update_failure_does_not_mask_pipeline_error(tmp_path):
    result, log, _ = _run(tmp_path, data_dir=tmp_path / "missing", fail_on="status='failed'")
    assert isinstance(result, FileNotFoundError), f"got {result!r}"


def test_dry_run_never_touches_the_database(tmp_path):
    generate(tmp_path / "raw", n_customers=100, n_orders=150)
    with mock.patch.dict(sys.modules, {"sqlalchemy": None}), mock.patch.object(load, "OUTPUT_DIR", tmp_path / "out"):
        result = load.run(tmp_path / "raw", dry_run=True)
    assert result["status"] == "success" and result["records_rejected"] > 0
    rep = pd.read_csv(tmp_path / "out" / "data_quality_report.csv")
    assert {"table", "rule", "severity", "records_failed"} <= set(rep.columns)
    assert (tmp_path / "out" / "rejected_records.csv").exists()


def test_run_ids_are_unique(tmp_path):
    generate(tmp_path / "raw", n_customers=60, n_orders=80)
    with mock.patch.object(load, "OUTPUT_DIR", tmp_path / "out"):
        ids = {load.run(tmp_path / "raw", dry_run=True)["run_id"] for _ in range(5)}
    assert len(ids) == 5


# ---------------------------------------------------------------- report.py
def _report(run_row, rules):
    fake_sa = types.SimpleNamespace(create_engine=lambda url: "E")
    frames = iter([pd.DataFrame([run_row]), rules])
    return mock.patch.dict(sys.modules, {"sqlalchemy": fake_sa}), mock.patch("pandas.read_sql", side_effect=lambda *a, **k: next(frames))


def test_pipeline_report_markdown(tmp_path):
    run_row = {"run_id": "r1", "status": "success", "started_at": "2024-01-01", "finished_at": "2024-01-01",
               "records_read": 12345, "records_inserted": 12000, "records_rejected": 345}
    rules = pd.DataFrame({"table_name": ["orders"], "rule": ["order_id must be unique"], "severity": ["reject"],
                          "records_failed": [15], "action": ["rows removed"]})
    p1, p2 = _report(run_row, rules)
    with p1, p2, mock.patch.object(report, "OUTPUT_DIR", tmp_path):
        report.main()
    md = (tmp_path / "pipeline_report.md").read_text()
    assert "run r1" in md and "12,345" in md and "345" in md and "order_id must be unique" in md


def test_pipeline_report_handles_failed_run_and_no_rule_failures(tmp_path):
    run_row = {"run_id": "r2", "status": "failed", "started_at": "2024-01-01", "finished_at": None,
               "records_read": float("nan"), "records_inserted": float("nan"), "records_rejected": float("nan")}
    p1, p2 = _report(run_row, pd.DataFrame(columns=["table_name", "rule", "severity", "records_failed", "action"]))
    with p1, p2, mock.patch.object(report, "OUTPUT_DIR", tmp_path):
        report.main()
    md = (tmp_path / "pipeline_report.md").read_text()
    assert "failed" in md and "None" in md
