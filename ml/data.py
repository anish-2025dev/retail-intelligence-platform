"""Load an order-level frame either from the analytics warehouse or directly from CSVs."""
from __future__ import annotations

import pandas as pd

from data_quality.validation import validate
from ingestion.extract import extract

EXCLUDED_STATUSES = {"canceled", "unavailable"}


def from_csv(data_dir) -> pd.DataFrame:
    frames = extract(data_dir)
    frames.pop("category_translation", None)
    clean, _, _ = validate(frames)
    items = clean["order_items"]
    line_value = pd.to_numeric(items.price) + pd.to_numeric(items.freight_value)
    value = line_value.groupby(items.order_id).sum()   # avoids groupby.apply (behaviour differs across pandas versions)
    o = clean["orders"].merge(clean["customers"][["customer_id", "customer_unique_id"]], on="customer_id")
    o = o.assign(order_value=o.order_id.map(value)).dropna(subset=["order_value"])
    return o.rename(columns={"order_purchase_timestamp": "purchase_ts"})[
        ["order_id", "customer_unique_id", "purchase_ts", "order_value", "order_status"]]


def from_db() -> pd.DataFrame:
    from sqlalchemy import create_engine
    from ingestion.config import database_url
    q = """SELECT f.order_id, c.customer_unique_id, f.purchase_ts, f.order_value, f.order_status
           FROM analytics.fact_orders f JOIN analytics.dim_customers c USING (customer_key)
           WHERE f.order_value IS NOT NULL"""
    return pd.read_sql(q, create_engine(database_url()), parse_dates=["purchase_ts"])


def load_orders(csv_dir: str | None = None) -> pd.DataFrame:
    df = from_csv(csv_dir) if csv_dir else from_db()
    return df[~df.order_status.isin(EXCLUDED_STATUSES)].copy()
