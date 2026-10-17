"""Unit tests for data_quality.validation - one test per behaviour, using the hand-made mini dataset."""
import json

import pandas as pd

from data_quality.validation import validate
from tests.helpers import append_row, failed_count, mini_frames, mutate


def test_valid_data_passes_untouched():
    frames = mini_frames()
    clean, report, rejected = validate(frames)
    assert report.records_failed.sum() == 0, report[report.records_failed > 0]
    assert len(rejected) == 0
    for t in frames:
        assert len(clean[t]) == len(frames[t]), t


def test_input_frames_are_not_mutated():
    frames = mini_frames()
    snapshot = {k: v.copy() for k, v in frames.items()}
    validate(frames)
    for k in frames:
        pd.testing.assert_frame_equal(frames[k], snapshot[k])


# (table, rule, mutated frames, expected rows removed from that table)
def _reject_cases():
    f = mini_frames()
    return [
        ("customers", "customer_id must be unique", append_row(f, "customers", 0), 1),
        ("customers", "customer_id must not be null", mutate(f, "customers", 1, "customer_id", None), 1),
        ("products", "product_id must be unique", append_row(f, "products", 0), 1),
        ("sellers", "seller_id must be unique", append_row(f, "sellers", 0), 1),
        ("orders", "order_id must be unique", append_row(f, "orders", 0), 1),
        ("orders", "order_id must not be null", mutate(f, "orders", 1, "order_id", None), 1),
        ("orders", "order_purchase_timestamp must not be null", mutate(f, "orders", 1, "order_purchase_timestamp", None), 1),
        ("orders", "order_status must be a known value", mutate(f, "orders", 1, "order_status", "teleported"), 1),
        ("orders", "order_status must be a known value", mutate(f, "orders", 1, "order_status", "Delivered"), 1),  # case-sensitive
        ("orders", "order_status must be a known value", mutate(f, "orders", 1, "order_status", None), 1),
        ("orders", "customer_id must exist in customers (FK)", mutate(f, "orders", 1, "customer_id", "c999"), 1),
        ("orders", "customer_id must exist in customers (FK)", mutate(f, "orders", 1, "customer_id", None), 1),
        ("orders", "order timestamps must be chronologically consistent",
         mutate(f, "orders", 0, "order_delivered_customer_date", "2023-12-01 00:00:00"), 1),
        ("orders", "order_purchase_timestamp must be a valid timestamp", mutate(f, "orders", 1, "order_purchase_timestamp", "not-a-date"), 1),
        ("orders", "order_delivered_customer_date must be a valid timestamp", mutate(f, "orders", 0, "order_delivered_customer_date", "31/31/2024"), 1),
        ("order_items", "price must be a number > 0", mutate(f, "order_items", 0, "price", "abc"), 1),
        ("order_items", "price must be a number > 0", mutate(f, "order_items", 0, "price", "0"), 1),
        ("order_items", "price must be a number > 0", mutate(f, "order_items", 0, "price", "-5"), 1),
        ("order_items", "price must be a number > 0", mutate(f, "order_items", 0, "price", None), 1),
        ("order_items", "freight_value must be a number >= 0", mutate(f, "order_items", 0, "freight_value", "-1"), 1),
        ("order_items", "order_id and order_item_id must not be null", mutate(f, "order_items", 0, "order_item_id", None), 1),
        ("order_items", "order_id must exist in orders (FK)", mutate(f, "order_items", 0, "order_id", "o999"), 1),
        ("order_items", "product_id must exist in products (FK)", mutate(f, "order_items", 0, "product_id", "p999"), 1),
        ("order_items", "seller_id must exist in sellers (FK)", mutate(f, "order_items", 0, "seller_id", "s999"), 1),
        ("payments", "payment_value must be a number >= 0", mutate(f, "payments", 0, "payment_value", "-1"), 1),
        ("payments", "payment_value must be a number >= 0", mutate(f, "payments", 0, "payment_value", "n/a"), 1),
        ("payments", "order_id must exist in orders (FK)", mutate(f, "payments", 0, "order_id", "o999"), 1),
        ("reviews", "review_score must be an integer 1-5", mutate(f, "reviews", 0, "review_score", "0"), 1),
        ("reviews", "review_score must be an integer 1-5", mutate(f, "reviews", 0, "review_score", "6"), 1),
        ("reviews", "review_score must be an integer 1-5", mutate(f, "reviews", 0, "review_score", "3.5"), 1),
        ("reviews", "review_score must be an integer 1-5", mutate(f, "reviews", 0, "review_score", "great"), 1),
        ("reviews", "(review_id, order_id) must be unique", append_row(f, "reviews", 0), 1),
        ("reviews", "order_id must exist in orders (FK)", mutate(f, "reviews", 0, "order_id", "o999"), 1),
    ]


def test_every_reject_rule_fires_and_removes_exactly_the_bad_rows():
    problems = []
    for table, rule, frames, expected in _reject_cases():
        clean, report, rejected = validate(frames)
        try:
            n = failed_count(report, table, rule)
            removed = len(frames[table]) - len(clean[table])
            assert n >= expected, f"rule counted {n}, expected >= {expected}"
            assert removed >= expected, f"{removed} rows removed from {table}"
            assert (rejected.rule == rule).sum() == n, "rejected log disagrees with report"
        except AssertionError as e:
            problems.append(f"{table} | {rule} | {e}")
    assert not problems, "\n" + "\n".join(problems)


def test_warn_rules_flag_but_keep_rows():
    f = mini_frames()
    cases = [
        ("orders", "delivered orders should have a delivery date", mutate(f, "orders", 0, "order_delivered_customer_date", None)),
        ("orders", "non-delivered orders should not have a delivery date", mutate(f, "orders", 1, "order_delivered_customer_date", "2024-01-08 00:00:00")),
        ("payments", "payment_type should be defined", mutate(f, "payments", 0, "payment_type", "not_defined")),
        ("products", "product_category_name should not be null", mutate(f, "products", 0, "product_category_name", None)),
        ("orders", "order_estimated_delivery_date should not be null", mutate(f, "orders", 1, "order_estimated_delivery_date", None)),
    ]
    for table, rule, frames in cases:
        clean, report, rejected = validate(frames)
        assert failed_count(report, table, rule) == 1, rule
        assert len(clean[table]) == len(frames[table]), f"warn rule removed rows: {rule}"
        assert report[(report.table == table) & (report.rule == rule)].severity.iloc[0] == "warn"
        assert len(rejected) == 0, rule


def test_order_without_items_is_warned_not_rejected():
    f = mini_frames()
    f["order_items"] = f["order_items"][f["order_items"].order_id != "o2"]
    clean, report, _ = validate(f)
    assert failed_count(report, "orders", "order should have at least one item") == 1
    assert "o2" in set(clean["orders"].order_id)


def test_missing_delivery_dates_for_undelivered_orders_are_legitimate():
    clean, report, _ = validate(mini_frames())
    assert clean["orders"].order_delivered_customer_date.isna().sum() == 1
    assert failed_count(report, "orders", "delivered orders should have a delivery date") == 0


def test_chronology_is_checked_even_when_intermediate_timestamps_are_missing():
    # approved + carrier missing, but delivery is before purchase -> still impossible
    f = mini_frames()
    f = mutate(f, "orders", 0, "order_approved_at", None)
    f = mutate(f, "orders", 0, "order_delivered_carrier_date", None)
    f = mutate(f, "orders", 0, "order_delivered_customer_date", "2023-06-01 00:00:00")
    _, report, _ = validate(f)
    assert failed_count(report, "orders", "order timestamps must be chronologically consistent") == 1


def test_missing_timestamps_alone_never_trigger_chronology_rule():
    f = mini_frames()
    for c in ["order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date"]:
        f = mutate(f, "orders", 0, c, None)
    _, report, _ = validate(f)
    assert failed_count(report, "orders", "order timestamps must be chronologically consistent") == 0


def test_rejecting_an_order_cascades_to_children_and_logs_them():
    f = mutate(mini_frames(), "orders", 0, "order_status", "teleported")  # o1 invalid
    clean, report, rejected = validate(f)
    assert "o1" not in set(clean["orders"].order_id)
    assert not (clean["order_items"].order_id == "o1").any()
    assert not (clean["payments"].order_id == "o1").any()
    assert not (clean["reviews"].order_id == "o1").any()
    assert {"orders", "order_items", "payments", "reviews"} <= set(rejected.table_name)


def test_rejected_log_is_complete_and_machine_readable():
    f = mutate(mini_frames(), "order_items", 0, "price", "-5")
    _, _, rejected = validate(f)
    assert list(rejected.columns) == ["table_name", "rule", "record_key", "record_json"]
    row = rejected.iloc[0]
    assert row.table_name == "order_items" and row.record_key == "o1|1"
    payload = json.loads(row.record_json)
    assert payload["price"] == "-5" and payload["product_id"] == "p1"


def test_report_accounting_adds_up():
    from ingestion.generate_sample_data import generate
    import tempfile, pathlib
    from ingestion.extract import extract
    d = pathlib.Path(tempfile.mkdtemp()); generate(d, n_customers=200, n_orders=300)
    frames = extract(d); frames.pop("category_translation")
    clean, report, rejected = validate(frames)
    assert set(report.severity) <= {"reject", "warn"}
    assert (report.records_checked >= 0).all() and (report.records_failed <= report.records_checked).all()
    # each rejected row is removed once, so rows removed == sum of failures of reject rules
    assert report[report.severity == "reject"].records_failed.sum() == len(rejected)
    removed = sum(len(frames[t]) - len(clean[t]) for t in frames)
    assert removed == len(rejected)


def test_empty_tables_do_not_crash():
    f = {k: v.iloc[0:0] for k, v in mini_frames().items()}
    clean, report, rejected = validate(f)
    assert report.records_failed.sum() == 0 and len(rejected) == 0
    assert all(len(v) == 0 for v in clean.values())


def test_equal_timestamps_are_valid_boundary():
    # approved in the same second as purchase, delivered the moment it reached the carrier: unusual but not impossible
    f = mini_frames()
    f = mutate(f, "orders", 0, "order_approved_at", f["orders"].loc[0, "order_purchase_timestamp"])
    f = mutate(f, "orders", 0, "order_delivered_customer_date", f["orders"].loc[0, "order_delivered_carrier_date"])
    clean, report, _ = validate(f)
    assert failed_count(report, "orders", "order timestamps must be chronologically consistent") == 0
    assert "o1" in set(clean["orders"].order_id)
