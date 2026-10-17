"""Shared test helpers: a tiny hand-made dataset whose correct answers are easy to compute by hand."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd


def mini_frames() -> dict[str, pd.DataFrame]:
    """Fully valid dataset (everything is text, like ingestion.extract returns)."""
    def df(rows, cols):
        return pd.DataFrame(rows, columns=cols, dtype=str)
    return {
        "customers": df([["c1", "u1", "01000", "sao paulo", "SP"], ["c2", "u2", "02000", "rio", "RJ"]],
                        ["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"]),
        "products": df([["p1", "toys", "100"], ["p2", "books", "200"]],
                       ["product_id", "product_category_name", "product_weight_g"]),
        "sellers": df([["s1", "01000", "sao paulo", "SP"]], ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]),
        "orders": df([
            ["o1", "c1", "delivered", "2024-01-01 10:00:00", "2024-01-01 11:00:00", "2024-01-02 09:00:00", "2024-01-05 09:00:00", "2024-01-10 00:00:00"],
            ["o2", "c2", "shipped", "2024-01-03 10:00:00", "2024-01-03 11:00:00", "2024-01-04 09:00:00", None, "2024-01-12 00:00:00"],
        ], ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at",
            "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"]),
        "order_items": df([
            ["o1", "1", "p1", "s1", "2024-01-04 00:00:00", "100.00", "10.00"],
            ["o1", "2", "p2", "s1", "2024-01-04 00:00:00", "50.00", "5.00"],
            ["o2", "1", "p1", "s1", "2024-01-06 00:00:00", "30.00", "5.00"],
        ], ["order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_date", "price", "freight_value"]),
        "payments": df([["o1", "1", "credit_card", "1", "165.00"], ["o2", "1", "boleto", "1", "35.00"]],
                       ["order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"]),
        "reviews": df([["r1", "o1", "5", None, None, "2024-01-06 00:00:00", "2024-01-07 00:00:00"]],
                      ["review_id", "order_id", "review_score", "review_comment_title", "review_comment_message",
                       "review_creation_date", "review_answer_timestamp"]),
    }


def mutate(frames, table, row, col, value):
    f = copy.deepcopy(frames)
    f[table].loc[row, col] = value
    return f


def append_row(frames, table, row_idx_to_copy, **overrides):
    f = copy.deepcopy(frames)
    new = f[table].iloc[[row_idx_to_copy]].copy()
    for k, v in overrides.items():
        new[k] = v
    f[table] = pd.concat([f[table], new], ignore_index=True)
    return f


def failed_count(report: pd.DataFrame, table: str, rule: str) -> int:
    r = report[(report.table == table) & (report.rule == rule)]
    assert len(r) == 1, f"rule not found / ambiguous: {table!r} {rule!r}"
    return int(r.records_failed.iloc[0])
