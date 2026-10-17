"""End-to-end tests: run the real command-line entry points in subprocesses (no database)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _cli(*args, raw, out, check=True):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "RAW_DATA_DIR": str(raw), "OUTPUT_DIR": str(out)}
    r = subprocess.run([sys.executable, "-m", *args], cwd=out.parent, env=env, capture_output=True, text=True, timeout=300)
    if check:
        assert r.returncode == 0, f"{' '.join(args)} failed:\n{r.stderr[-1500:]}"
    return r


def _pipeline(base: Path, n_orders=1500):
    raw, out = base / "raw", base / "out"
    out.mkdir(parents=True, exist_ok=True)
    _cli("ingestion.generate_sample_data", "--out-dir", str(raw), raw=raw, out=out)
    load = _cli("ingestion.load", "--data-dir", str(raw), "--dry-run", raw=raw, out=out)
    _cli("ml.customer_segmentation", "--csv-dir", str(raw), raw=raw, out=out)
    _cli("ml.sales_forecasting", "--csv-dir", str(raw), "--horizon", "6", raw=raw, out=out)
    return raw, out, load


def test_full_pipeline_cli_end_to_end(tmp_path):
    raw, out, load = _pipeline(tmp_path)
    for f in ("data_quality_report.csv", "rejected_records.csv", "customer_segments.csv", "customer_segment_profiles.csv",
              "sales_forecast.csv", "forecast_model_metrics.csv"):
        assert (out / f).exists() and (out / f).stat().st_size > 0, f

    # ---- data quality: the injected defects are found and accounted for
    rep = pd.read_csv(out / "data_quality_report.csv").set_index("rule")
    assert rep.loc["order_id must be unique", "records_failed"] == 15
    assert rep.loc["order_status must be a known value", "records_failed"] == 1
    assert rep.loc["price must be a number > 0", "records_failed"] == 10
    assert rep.loc["review_score must be an integer 1-5", "records_failed"] == 9
    assert rep.loc["order timestamps must be chronologically consistent", "records_failed"] >= 12
    rejected = pd.read_csv(out / "rejected_records.csv")
    summary = load.stdout.strip().splitlines()[-1]
    assert f"'records_rejected': {len(rejected)}" in summary and "'status': 'success'" in summary
    assert rep[rep.severity == "reject"].records_failed.sum() == len(rejected)
    json.loads(rejected.record_json.iloc[0])                                    # payload is valid JSON

    # ---- segmentation
    seg = pd.read_csv(out / "customer_segments.csv")
    prof = pd.read_csv(out / "customer_segment_profiles.csv", index_col=0)
    assert seg.customer_unique_id.is_unique and prof.customers.sum() == len(seg)
    assert prof.suggested_segment.is_unique and (seg.recency_days >= 1).all() and (seg.monetary > 0).all()

    # ---- forecast
    fc = pd.read_csv(out / "sales_forecast.csv", parse_dates=["week_end"])
    assert set(fc.kind) == {"actual", "backtest_prediction", "forecast"}
    assert (fc.kind == "forecast").sum() == 6 and fc.sales.notna().all()
    assert fc[fc.kind == "forecast"].week_end.min() > fc[fc.kind == "actual"].week_end.max()
    m = pd.read_csv(out / "forecast_model_metrics.csv", index_col=0)
    assert fc.model.iloc[0] == m.mae.idxmin()


def test_pipeline_is_reproducible_across_runs(tmp_path):
    _, out1, _ = _pipeline(tmp_path / "run1")
    _, out2, _ = _pipeline(tmp_path / "run2")
    for f in ("data_quality_report.csv", "rejected_records.csv", "customer_segments.csv", "sales_forecast.csv", "forecast_model_metrics.csv"):
        assert (out1 / f).read_bytes() == (out2 / f).read_bytes(), f"{f} differs between identical runs"


def test_cli_fails_cleanly_when_data_is_missing(tmp_path):
    out = tmp_path / "out"; out.mkdir()
    r = _cli("ingestion.load", "--data-dir", str(tmp_path / "nope"), "--dry-run", raw=tmp_path / "nope", out=out, check=False)
    assert r.returncode != 0 and "Kaggle" in r.stderr and "olist_customers_dataset.csv" in r.stderr
    r = _cli("ml.customer_segmentation", "--csv-dir", str(tmp_path / "nope"), raw=tmp_path / "nope", out=out, check=False)
    assert r.returncode != 0 and "FileNotFoundError" in r.stderr


def test_pipeline_survives_real_world_messiness(tmp_path):
    """Olist-like quirks: missing approval dates, blank cells, dup review ids across orders, zero payments, extra whitespace-free ids."""
    raw, out = tmp_path / "raw", tmp_path / "out"; out.mkdir()
    _cli("ingestion.generate_sample_data", "--out-dir", str(raw), raw=raw, out=out)
    orders = pd.read_csv(raw / "olist_orders_dataset.csv", dtype=str)
    rng = np.random.default_rng(0)
    orders.loc[rng.choice(len(orders), 120, replace=False), "order_approved_at"] = np.nan
    orders.loc[rng.choice(len(orders), 60, replace=False), "order_delivered_carrier_date"] = np.nan
    orders.to_csv(raw / "olist_orders_dataset.csv", index=False)
    rev = pd.read_csv(raw / "olist_order_reviews_dataset.csv", dtype=str)
    rev["review_id"] = rev.review_id.where(~rev.index.isin(rev.index[:40]), "r_shared")      # same review_id on many orders
    rev.to_csv(raw / "olist_order_reviews_dataset.csv", index=False)
    pay = pd.read_csv(raw / "olist_order_payments_dataset.csv", dtype=str)
    pay.loc[pay.index[:25], "payment_value"] = "0.00"; pay.loc[pay.index[25:30], "payment_type"] = "not_defined"
    pay.to_csv(raw / "olist_order_payments_dataset.csv", index=False)

    r = _cli("ingestion.load", "--data-dir", str(raw), "--dry-run", raw=raw, out=out)
    rep = pd.read_csv(out / "data_quality_report.csv").set_index("rule")
    assert rep.loc["(review_id, order_id) must be unique", "records_failed"] == 0       # shared review_id across orders is legitimate
    assert rep.loc["payment_value must be a number >= 0", "records_failed"] == 6        # zero is valid; only the injected negatives fail
    assert rep.loc["payment_type should be defined", "records_failed"] == 5
    _cli("ml.customer_segmentation", "--csv-dir", str(raw), raw=raw, out=out)
    _cli("ml.sales_forecasting", "--csv-dir", str(raw), raw=raw, out=out)


def test_validation_scales_to_hundreds_of_thousands_of_rows():
    from data_quality.validation import validate
    n = 120_000
    rng = np.random.default_rng(1)
    ids = np.arange(n).astype(str)
    ts = (pd.Timestamp("2017-01-01") + pd.to_timedelta(rng.integers(0, 600 * 86400, n), unit="s"))
    fmt = lambda s: s.strftime("%Y-%m-%d %H:%M:%S")
    frames = {
        "customers": pd.DataFrame({"customer_id": "c" + pd.Series(ids), "customer_unique_id": "u" + pd.Series(ids),
                                   "customer_zip_code_prefix": "1", "customer_city": "x", "customer_state": "SP"}),
        "products": pd.DataFrame({"product_id": "p" + pd.Series(ids[:30000]), "product_category_name": "toys"}),
        "sellers": pd.DataFrame({"seller_id": "s" + pd.Series(ids[:3000])}),
        "orders": pd.DataFrame({"order_id": "o" + pd.Series(ids), "customer_id": "c" + pd.Series(ids), "order_status": "delivered",
                                "order_purchase_timestamp": fmt(ts), "order_approved_at": fmt(ts + pd.Timedelta(hours=1)),
                                "order_delivered_carrier_date": fmt(ts + pd.Timedelta(days=1)),
                                "order_delivered_customer_date": fmt(ts + pd.Timedelta(days=5)),
                                "order_estimated_delivery_date": fmt(ts + pd.Timedelta(days=9))}),
        "order_items": pd.DataFrame({"order_id": "o" + pd.Series(ids), "order_item_id": "1", "product_id": "p" + pd.Series(ids[:30000]).sample(n, replace=True, random_state=0).values,
                                     "seller_id": "s0", "shipping_limit_date": fmt(ts), "price": "10.0", "freight_value": "1.0"}),
        "payments": pd.DataFrame({"order_id": "o" + pd.Series(ids), "payment_sequential": "1", "payment_type": "credit_card",
                                  "payment_installments": "1", "payment_value": "11.0"}),
        "reviews": pd.DataFrame({"review_id": "r" + pd.Series(ids), "order_id": "o" + pd.Series(ids), "review_score": "5",
                                 "review_creation_date": fmt(ts), "review_answer_timestamp": fmt(ts)}),
    }
    bad = rng.choice(n, 1200, replace=False)                                            # 1% defective prices
    frames["order_items"].loc[bad, "price"] = "-1"
    t0 = time.time()
    clean, report, rejected = validate(frames)
    elapsed = time.time() - t0
    assert elapsed < 60, f"validation too slow: {elapsed:.1f}s"
    assert len(clean["order_items"]) == n - 1200 and len(rejected) >= 1200
    print(f"(validated {sum(len(v) for v in frames.values()):,} rows in {elapsed:.1f}s)")
