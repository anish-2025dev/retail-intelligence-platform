"""Generate a small synthetic dataset with the same schema as the Olist dataset.

Lets you run and test the whole pipeline without downloading anything. It deliberately
injects defects (duplicates, orphan keys, impossible timestamps, bad scores) so the
data-quality layer has something to catch. Use the real Olist data for the portfolio version.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion.config import RAW_DIR

CATEGORIES = ["beleza_saude", "informatica_acessorios", "cama_mesa_banho", "esporte_lazer", "moveis_decoracao"]
TRANSLATION = ["health_beauty", "computers_accessories", "bed_bath_table", "sports_leisure", "furniture_decor"]
STATES = ["SP", "RJ", "MG", "RS", "BA", "PR"]


def generate(out_dir: Path, n_customers: int = 3000, n_orders: int = 4500, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%Y-%m-%d %H:%M:%S"

    cust_ids = [f"c{i:06d}" for i in range(n_orders)]  # one customer_id per order, like Olist
    unique_ids = rng.integers(0, n_customers, n_orders)
    customers = pd.DataFrame({
        "customer_id": cust_ids,
        "customer_unique_id": [f"u{u:05d}" for u in unique_ids],
        "customer_zip_code_prefix": rng.integers(1000, 99999, n_orders).astype(str),
        "customer_city": rng.choice(["sao paulo", "rio de janeiro", "belo horizonte", "curitiba"], n_orders),
        "customer_state": rng.choice(STATES, n_orders, p=[.4, .2, .15, .1, .1, .05]),
    })

    sellers = pd.DataFrame({
        "seller_id": [f"s{i:04d}" for i in range(120)],
        "seller_zip_code_prefix": rng.integers(1000, 99999, 120).astype(str),
        "seller_city": rng.choice(["sao paulo", "campinas", "curitiba"], 120),
        "seller_state": rng.choice(STATES, 120),
    })
    late_prone = set(rng.choice(sellers.seller_id, 15, replace=False))  # known "bad" sellers

    products = pd.DataFrame({
        "product_id": [f"p{i:05d}" for i in range(600)],
        "product_category_name": rng.choice(CATEGORIES, 600),
        "product_weight_g": rng.integers(100, 5000, 600).astype(str),
    })
    products.loc[rng.choice(600, 12, replace=False), "product_category_name"] = np.nan

    start = pd.Timestamp("2017-01-01")
    purchase = start + pd.to_timedelta(rng.integers(0, 600 * 24 * 3600, n_orders), unit="s")
    # mild upward trend + seasonality: resample dates with weight
    purchase = pd.Series(np.sort(purchase.values))
    statuses = rng.choice(["delivered", "shipped", "canceled", "invoiced"], n_orders, p=[.93, .03, .03, .01])
    approved = purchase + pd.to_timedelta(rng.integers(600, 86400, n_orders), unit="s")
    carrier = approved + pd.to_timedelta(rng.integers(3600, 5 * 86400, n_orders), unit="s")

    items_rows, order_seller = [], {}
    for i in range(n_orders):
        for k in range(1, rng.choice([1, 1, 1, 2, 3]) + 1):
            sid = rng.choice(sellers.seller_id) if k == 1 else order_seller[i]
            order_seller.setdefault(i, sid)
            items_rows.append({
                "order_id": f"o{i:06d}", "order_item_id": str(k),
                "product_id": rng.choice(products.product_id), "seller_id": sid,
                "shipping_limit_date": (approved[i] + pd.Timedelta(days=3)).strftime(fmt),
                "price": f"{rng.gamma(2.0, 45) + 5:.2f}", "freight_value": f"{rng.uniform(5, 40):.2f}",
            })
    items = pd.DataFrame(items_rows)

    seller_delay = np.array([rng.integers(6, 14) if order_seller[i] in late_prone else rng.integers(-3, 3)
                             for i in range(n_orders)])
    transit = rng.integers(4, 15, n_orders)
    estimated = carrier + pd.to_timedelta(transit + 3, unit="D")
    delivered = carrier + pd.to_timedelta(transit + seller_delay, unit="D")
    delivered = delivered.where(statuses == "delivered", pd.NaT)

    orders = pd.DataFrame({
        "order_id": [f"o{i:06d}" for i in range(n_orders)], "customer_id": cust_ids, "order_status": statuses,
        "order_purchase_timestamp": purchase.dt.strftime(fmt),
        "order_approved_at": approved.dt.strftime(fmt),
        "order_delivered_carrier_date": carrier.dt.strftime(fmt),
        "order_delivered_customer_date": delivered.dt.strftime(fmt),
        "order_estimated_delivery_date": estimated.dt.strftime(fmt),
    })

    totals = items.assign(t=items.price.astype(float) + items.freight_value.astype(float)).groupby("order_id").t.sum()
    payments = pd.DataFrame({
        "order_id": totals.index, "payment_sequential": "1",
        "payment_type": rng.choice(["credit_card", "boleto", "voucher", "debit_card"], len(totals), p=[.74, .19, .05, .02]),
        "payment_installments": rng.integers(1, 10, len(totals)).astype(str),
        "payment_value": totals.round(2).astype(str).values,
    })

    late = (delivered > estimated).fillna(False).values
    score = np.where(late, rng.choice([1, 2, 3, 4, 5], n_orders, p=[.35, .2, .2, .15, .1]),
                     rng.choice([1, 2, 3, 4, 5], n_orders, p=[.03, .03, .1, .29, .55]))
    reviews = pd.DataFrame({
        "review_id": [f"r{i:06d}" for i in range(n_orders)], "order_id": orders.order_id,
        "review_score": score.astype(str), "review_comment_title": np.nan, "review_comment_message": np.nan,
        "review_creation_date": (delivered.fillna(carrier) + pd.Timedelta(days=1)).dt.strftime(fmt),
        "review_answer_timestamp": (delivered.fillna(carrier) + pd.Timedelta(days=2)).dt.strftime(fmt),
    })

    # ---- inject defects for the validation layer to find ----
    orders = pd.concat([orders, orders.sample(15, random_state=1)])                      # duplicate orders
    bad = orders.sample(12, random_state=2).index
    orders.loc[bad[:12], "order_delivered_customer_date"] = "2016-01-01 00:00:00"        # delivered before purchase
    orders.loc[orders.index[5], "order_status"] = "teleported"                          # invalid status
    items.loc[items.sample(10, random_state=3).index, "price"] = "-5.00"                 # negative price
    items.loc[items.sample(8, random_state=4).index, "product_id"] = "p_missing"         # orphan FK
    payments.loc[payments.sample(6, random_state=5).index, "payment_value"] = "-1.00"
    reviews.loc[reviews.sample(9, random_state=6).index, "review_score"] = "7"           # out of range

    translation = pd.DataFrame({"product_category_name": CATEGORIES, "product_category_name_english": TRANSLATION})

    for name, df in {
        "olist_customers_dataset": customers, "olist_orders_dataset": orders, "olist_order_items_dataset": items,
        "olist_order_payments_dataset": payments, "olist_order_reviews_dataset": reviews,
        "olist_products_dataset": products, "olist_sellers_dataset": sellers,
        "product_category_name_translation": translation,
    }.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    print(f"Sample data written to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(RAW_DIR))
    generate(Path(ap.parse_args().out_dir))
