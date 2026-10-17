"""Data-quality layer.

Every rule is recorded (table, rule, severity, how many records failed, what happened).
- severity "reject": failing rows are removed from the clean data and stored in the rejected log
- severity "warn":   failing rows are kept but counted in the report

Nothing is cleaned silently: `validate()` returns the clean frames, a rule report and the rejected rows.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import combinations

import pandas as pd

from ingestion.config import TIMESTAMP_COLUMNS

VALID_STATUSES = {
    "created", "approved", "invoiced", "processing", "shipped", "delivered", "unavailable", "canceled",
}
KEYS = {
    "customers": ["customer_id"], "orders": ["order_id"], "order_items": ["order_id", "order_item_id"],
    "payments": ["order_id", "payment_sequential"], "reviews": ["review_id", "order_id"],
    "products": ["product_id"], "sellers": ["seller_id"],
}


@dataclass
class RuleResult:
    table: str
    rule: str
    severity: str  # reject | warn
    records_checked: int
    records_failed: int
    action: str


class _Validator:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = {k: v.copy() for k, v in frames.items()}
        self.results: list[RuleResult] = []
        self.rejected: list[pd.DataFrame] = []

    def check(self, table: str, rule: str, failed_mask: pd.Series, severity: str = "reject") -> None:
        df = self.frames[table]
        failed_mask = failed_mask.reindex(df.index, fill_value=False).astype(bool)
        n_failed = int(failed_mask.sum())
        if severity == "reject" and n_failed:
            bad = df[failed_mask]
            key = bad[KEYS[table]].fillna("<null>").astype(str).agg("|".join, axis=1)
            self.rejected.append(pd.DataFrame({
                "table_name": table, "rule": rule, "record_key": key.values,
                "record_json": bad.astype(object).where(bad.notna(), None).apply(
                    lambda r: json.dumps(r.to_dict(), default=str), axis=1).values,
            }))
            self.frames[table] = df[~failed_mask]
        action = "rows removed" if severity == "reject" else "rows kept, flagged"
        self.results.append(RuleResult(table, rule, severity, len(df), n_failed, action if n_failed else "none"))

    def parse_timestamps(self) -> None:
        for table, cols in TIMESTAMP_COLUMNS.items():
            if table not in self.frames:
                continue
            for col in cols:
                raw = self.frames[table][col]
                parsed = pd.to_datetime(raw, errors="coerce")
                unparseable = raw.notna() & parsed.isna()
                self.frames[table][col] = parsed
                self.check(table, f"{col} must be a valid timestamp", unparseable)


def validate(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    v = _Validator(frames)
    v.parse_timestamps()

    # ---- dimensions: key uniqueness ----
    for table in ("customers", "products", "sellers"):
        key = KEYS[table][0]
        v.check(table, f"{key} must be unique", v.frames[table][key].duplicated(keep="first"))
        v.check(table, f"{key} must not be null", v.frames[table][key].isna())

    # ---- orders ----
    o = v.frames["orders"]
    v.check("orders", "order_id must not be null", o.order_id.isna())
    o = v.frames["orders"]
    v.check("orders", "order_id must be unique", o.order_id.duplicated(keep="first"))
    o = v.frames["orders"]
    v.check("orders", "order_purchase_timestamp must not be null", o.order_purchase_timestamp.isna())
    o = v.frames["orders"]
    v.check("orders", "order_status must be a known value", ~o.order_status.isin(VALID_STATUSES))
    o = v.frames["orders"]
    v.check("orders", "customer_id must exist in customers (FK)", ~o.customer_id.isin(v.frames["customers"].customer_id))
    o = v.frames["orders"]
    chain = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date"]
    out_of_order = pd.Series(False, index=o.index)
    for a, b in combinations(chain, 2):  # every pair of *present* timestamps, so a missing step can't hide a problem
        out_of_order |= o[a].notna() & o[b].notna() & (o[a] > o[b])
    v.check("orders", "order timestamps must be chronologically consistent", out_of_order)
    o = v.frames["orders"]
    v.check("orders", "delivered orders should have a delivery date", (o.order_status == "delivered") & o.order_delivered_customer_date.isna(), "warn")
    v.check("orders", "order_estimated_delivery_date should not be null", o.order_estimated_delivery_date.isna(), "warn")
    v.check("orders", "non-delivered orders should not have a delivery date", (o.order_status != "delivered") & o.order_delivered_customer_date.notna(), "warn")

    # ---- order items ----
    i = v.frames["order_items"]
    v.check("order_items", "order_id and order_item_id must not be null", i.order_id.isna() | i.order_item_id.isna())
    i = v.frames["order_items"]
    price = pd.to_numeric(i.price, errors="coerce")
    freight = pd.to_numeric(i.freight_value, errors="coerce")
    v.check("order_items", "price must be a number > 0", price.isna() | (price <= 0))
    v.check("order_items", "freight_value must be a number >= 0", freight.isna() | (freight < 0))
    i = v.frames["order_items"]
    v.check("order_items", "order_id must exist in orders (FK)", ~i.order_id.isin(v.frames["orders"].order_id))
    i = v.frames["order_items"]
    v.check("order_items", "product_id must exist in products (FK)", ~i.product_id.isin(v.frames["products"].product_id))
    i = v.frames["order_items"]
    v.check("order_items", "seller_id must exist in sellers (FK)", ~i.seller_id.isin(v.frames["sellers"].seller_id))

    # ---- payments ----
    p = v.frames["payments"]
    value = pd.to_numeric(p.payment_value, errors="coerce")
    v.check("payments", "payment_value must be a number >= 0", value.isna() | (value < 0))
    p = v.frames["payments"]
    v.check("payments", "order_id must exist in orders (FK)", ~p.order_id.isin(v.frames["orders"].order_id))
    p = v.frames["payments"]
    v.check("payments", "payment_type should be defined", p.payment_type.eq("not_defined"), "warn")

    # ---- reviews ----
    r = v.frames["reviews"]
    score = pd.to_numeric(r.review_score, errors="coerce")
    v.check("reviews", "review_score must be an integer 1-5", ~score.isin([1, 2, 3, 4, 5]))
    r = v.frames["reviews"]
    v.check("reviews", "(review_id, order_id) must be unique", r.duplicated(subset=["review_id", "order_id"], keep="first"))
    r = v.frames["reviews"]
    v.check("reviews", "order_id must exist in orders (FK)", ~r.order_id.isin(v.frames["orders"].order_id))

    # ---- soft checks ----
    pr = v.frames["products"]
    v.check("products", "product_category_name should not be null", pr.product_category_name.isna(), "warn")
    orders_with_items = v.frames["order_items"].order_id.unique()
    o = v.frames["orders"]
    v.check("orders", "order should have at least one item", ~o.order_id.isin(orders_with_items), "warn")

    report = pd.DataFrame([asdict(r) for r in v.results])
    rejected = (pd.concat(v.rejected, ignore_index=True) if v.rejected
                else pd.DataFrame(columns=["table_name", "rule", "record_key", "record_json"]))
    return v.frames, report, rejected
