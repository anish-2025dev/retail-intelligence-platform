"""Unit tests: config, extract, sample-data generator."""
import os
import pathlib
from unittest import mock

import pandas as pd

from ingestion import config
from ingestion.extract import extract
from ingestion.generate_sample_data import generate


def test_database_url_defaults_and_overrides():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert config.database_url() == "postgresql+psycopg2://retail:retail@localhost:5432/retail"
    env = {"POSTGRES_USER": "u", "POSTGRES_PASSWORD": "p", "POSTGRES_HOST": "db", "POSTGRES_PORT": "6543", "POSTGRES_DB": "x"}
    with mock.patch.dict(os.environ, env, clear=True):
        assert config.database_url() == "postgresql+psycopg2://u:p@db:6543/x"
    with mock.patch.dict(os.environ, {**env, "DATABASE_URL": "postgresql://explicit"}, clear=True):
        assert config.database_url() == "postgresql://explicit"          # explicit URL wins


def test_generate_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate(a, n_customers=100, n_orders=150); generate(b, n_customers=100, n_orders=150)
    for f in a.glob("*.csv"):
        assert f.read_bytes() == (b / f.name).read_bytes(), f.name


def test_generated_files_match_the_olist_schema(tmp_path):
    generate(tmp_path, n_customers=100, n_orders=150)
    expected = {
        "olist_orders_dataset.csv": ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at",
                                     "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"],
        "olist_customers_dataset.csv": ["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"],
        "olist_order_items_dataset.csv": ["order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_date", "price", "freight_value"],
        "olist_order_payments_dataset.csv": ["order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"],
        "olist_sellers_dataset.csv": ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"],
    }
    for fname, cols in expected.items():
        assert pd.read_csv(tmp_path / fname, nrows=1).columns.tolist() == cols, fname
    for logical, fname in {**config.FILES, **config.OPTIONAL_FILES}.items():
        assert (tmp_path / fname).exists(), f"generator does not produce {fname}"


def test_generated_data_contains_the_documented_defects(tmp_path):
    generate(tmp_path, n_customers=300, n_orders=500)
    orders = pd.read_csv(tmp_path / "olist_orders_dataset.csv")
    assert orders.order_id.duplicated().sum() == 15
    assert (orders.order_status == "teleported").sum() == 1
    assert (pd.to_numeric(pd.read_csv(tmp_path / "olist_order_items_dataset.csv").price) < 0).sum() == 10
    assert (pd.read_csv(tmp_path / "olist_order_reviews_dataset.csv").review_score == 7).sum() == 9


def test_extract_reads_everything_as_text_and_keeps_leading_zeros(tmp_path):
    generate(tmp_path, n_customers=50, n_orders=80)
    pd.DataFrame({"customer_id": ["c1"], "customer_unique_id": ["u1"], "customer_zip_code_prefix": ["01234"],
                  "customer_city": ["x"], "customer_state": ["SP"]}).to_csv(tmp_path / config.FILES["customers"], index=False)
    frames = extract(tmp_path)
    assert frames["customers"].customer_zip_code_prefix.iloc[0] == "01234"
    assert all(str(df[c].dtype) in ("object", "str", "string") for df in frames.values() for c in df.columns)
    assert "category_translation" in frames


def test_extract_without_optional_translation_file(tmp_path):
    generate(tmp_path, n_customers=50, n_orders=80)
    (tmp_path / config.OPTIONAL_FILES["category_translation"]).unlink()
    assert "category_translation" not in extract(tmp_path)


def test_extract_missing_file_gives_actionable_error(tmp_path):
    generate(tmp_path, n_customers=50, n_orders=80)
    (tmp_path / config.FILES["sellers"]).unlink()
    try:
        extract(tmp_path)
    except FileNotFoundError as e:
        assert "olist_sellers_dataset.csv" in str(e) and "Kaggle" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")
