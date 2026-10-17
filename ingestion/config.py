"""Central configuration, driven by environment variables (see .env.example)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.getenv("RAW_DATA_DIR", ROOT / "data" / "raw"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "outputs"))


def database_url() -> str:
    user = os.getenv("POSTGRES_USER", "retail")
    pwd = os.getenv("POSTGRES_PASSWORD", "retail")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "retail")
    return os.getenv("DATABASE_URL", f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}")


# logical table name -> Olist csv file name
FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
}
OPTIONAL_FILES = {"category_translation": "product_category_name_translation.csv"}

# columns that must be parsed as timestamps (kept as text until validation)
TIMESTAMP_COLUMNS = {
    "orders": [
        "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}
