"""dbt model tests on SQLite (see tests/dbt_sqlite_harness.py for scope and limits)."""
import functools
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tests.dbt_sqlite_harness import DBT_DIR, ROOT, DbtSqlite, clean_frames_from_generated, transpile


@functools.lru_cache(maxsize=2)
def _built(n_customers=400, n_orders=700):
    frames, _ = clean_frames_from_generated(Path(tempfile.mkdtemp()), n_customers, n_orders)
    db = DbtSqlite(frames)
    db.run()
    return db, frames


# ------------------------------------------------------------------ static project checks
def test_all_refs_resolve_and_no_cycles():
    db = DbtSqlite({})
    order = db.order()                                   # raises on unknown refs or cycles
    assert order.index("stg_orders") < order.index("fact_orders") < order.index("mart_seller_performance")
    assert len(order) == len(db.models) == 16


def test_sources_used_by_models_are_declared():
    import yaml
    declared = {t["name"] for s in yaml.safe_load((DBT_DIR / "models" / "sources.yml").read_text())["sources"] for t in s["tables"]}
    used = set()
    for p in (DBT_DIR / "models").rglob("*.sql"):
        used |= set(re.findall(r"source\('raw', '(\w+)'\)", p.read_text()))
    assert used <= declared, used - declared
    raw_tables_written = {"customers", "orders", "order_items", "payments", "reviews", "products", "sellers", "category_translation"}
    assert declared == raw_tables_written                 # dbt sources == tables ingestion/load.py creates in schema raw


def test_dbt_project_and_profile_are_consistent():
    import yaml
    proj = yaml.safe_load((DBT_DIR / "dbt_project.yml").read_text())
    prof = yaml.safe_load((DBT_DIR / "profiles.yml").read_text())
    assert proj["profile"] in prof and prof[proj["profile"]]["target"] in prof[proj["profile"]]["outputs"]
    cfg = proj["models"]["retail_platform"]
    assert cfg["staging"]["+schema"] == "staging" and cfg["marts"]["+schema"] == "analytics"
    assert "generate_schema_name" in (DBT_DIR / "macros" / "generate_schema_name.sql").read_text()
    # staging models live in models/staging, marts in models/marts (so the folder config above applies)
    assert all(p.stem.startswith("stg_") for p in (DBT_DIR / "models" / "staging").glob("*.sql"))
    assert not any(p.stem.startswith("stg_") for p in (DBT_DIR / "models" / "marts").glob("*.sql"))


def test_shim_transpiler_unit():
    assert transpile("select x::date") == "select date(x)"
    assert transpile("select sum(a)::numeric(12,2)") == "select cast(sum(a) as real)"
    assert transpile("select (a > b)::int") == "select cast((a > b) as integer)"
    assert "julianday" in transpile("extract(epoch from (o.b - o.a)) / 86400")


# ------------------------------------------------------------------ dbt `schema.yml` tests
def test_schema_yml_tests_all_pass():
    db, _ = _built()
    results = db.run_schema_tests()
    assert len(results) >= 15
    failing = [r for r in results if r[3] > 0]
    assert not failing, failing


def test_schema_yml_tests_actually_detect_problems():
    # sanity check of the harness itself: corrupt a model and make sure the test notices
    db, frames = _built()
    import copy
    db2 = DbtSqlite(copy.deepcopy(frames)); db2.run()
    db2.con.execute("insert into fact_orders (order_id, customer_key, order_status) values ('o000000', 'ghost', 'bogus')")
    failing = {(m, c, t) for m, c, t, n in db2.run_schema_tests() if n > 0}
    assert ("fact_orders", "order_id", "unique") in failing
    assert ("fact_orders", "customer_key", "relationships") in failing
    assert ("fact_orders", "order_status", "accepted_values") in failing


# ------------------------------------------------------------------ business logic vs independent pandas calc
def _truth(frames):
    o = frames["orders"].copy()
    items = frames["order_items"].assign(price=lambda d: d.price.astype(float), freight=lambda d: d.freight_value.astype(float))
    return o, items


def test_staging_row_counts_and_types():
    db, frames = _built()
    for stg, raw in [("stg_orders", "orders"), ("stg_customers", "customers"), ("stg_order_items", "order_items"),
                     ("stg_payments", "payments"), ("stg_reviews", "reviews"), ("stg_products", "products"), ("stg_sellers", "sellers")]:
        assert len(db.df(stg)) == len(frames[raw]), stg
    assert set(db.df("stg_order_items").price.map(type)) <= {float, int}
    assert db.df("stg_customers").customer_state.str.isupper().all()


def test_stg_products_category_translation_and_unknown():
    db, frames = _built()
    stg = db.df("stg_products").set_index("product_id").category
    assert "unknown" in set(stg) and not stg.isna().any()
    p = frames["products"].set_index("product_id")
    trans = frames["category_translation"].set_index("product_category_name").product_category_name_english
    for pid in p.index[:50]:
        raw = p.loc[pid, "product_category_name"]
        exp = "unknown" if pd.isna(raw) else trans.get(raw, raw)
        assert stg[pid] == exp


def test_fact_orders_grain_and_values_match_pandas():
    db, frames = _built()
    f = db.df("fact_orders").set_index("order_id")
    o, items = _truth(frames)
    assert len(f) == len(o) and f.index.is_unique                      # no join fan-out, no lost orders

    value = (items.price + items.freight).groupby(items.order_id).sum()
    got = f.order_value.dropna()
    np.testing.assert_allclose(got.values, value[got.index].values, rtol=1e-9)
    assert set(f.index[f.order_value.isna()]) == set(o.order_id) - set(value.index)   # orders w/o items -> NULL, not 0

    assert (f.item_count.dropna() == items.groupby("order_id").size()[f.item_count.dropna().index]).all()
    pay = frames["payments"].assign(v=lambda d: d.payment_value.astype(float)).groupby("order_id").v.sum()
    g = f.payment_value.dropna()
    np.testing.assert_allclose(g.values, pay[g.index].values, rtol=1e-9)

    rev = frames["reviews"].assign(s=lambda d: d.review_score.astype(float)).groupby("order_id").s.mean()
    g = f.review_score.dropna()
    np.testing.assert_allclose(g.values, rev[g.index].values, atol=1e-9)


def test_fact_orders_delivery_metrics():
    db, frames = _built()
    f = db.df("fact_orders").set_index("order_id")
    o = frames["orders"].set_index("order_id")
    delivered = o.order_delivered_customer_date.notna()
    # undelivered -> metrics are NULL (not 0 / not "on time")
    assert f.loc[delivered.index[~delivered], "is_late"].isna().all()
    assert f.loc[delivered.index[~delivered], "delivery_days"].isna().all()
    assert f.loc[delivered.index[delivered], "is_late"].notna().all()
    exp_late = (o.order_delivered_customer_date > o.order_estimated_delivery_date)[delivered]
    assert (f.loc[exp_late.index, "is_late"].astype(bool) == exp_late).all()
    exp_days = (o.order_delivered_customer_date - o.order_purchase_timestamp).dt.total_seconds()[delivered] / 86400
    np.testing.assert_allclose(f.loc[exp_days.index, "delivery_days"].values, exp_days.values, atol=1e-4)
    assert (f.delivery_days.dropna() > 0).all()
    assert (f.date_key == o.order_purchase_timestamp.dt.strftime("%Y-%m-%d").reindex(f.index)).all()


def test_fact_order_items_no_fanout_and_keys():
    db, frames = _built()
    fi = db.df("fact_order_items")
    assert len(fi) == len(frames["order_items"])
    assert not fi.duplicated(["order_id", "order_item_id"]).any()
    fo = db.df("fact_orders").set_index("order_id")
    assert (fi.date_key.values == fo.loc[fi.order_id, "date_key"].values).all()


def test_dimensions():
    db, frames = _built()
    assert len(db.df("dim_customers")) == len(frames["customers"])
    assert len(db.df("dim_products")) == len(frames["products"]) and len(db.df("dim_sellers")) == len(frames["sellers"])
    d = db.df("dim_date")
    dates = pd.to_datetime(d.date_key)
    o = frames["orders"]
    assert d.date_key.is_unique and (dates.diff().dropna() == pd.Timedelta(days=1)).all()      # contiguous date spine
    assert dates.min() == o.order_purchase_timestamp.min().normalize()
    assert dates.max() == o.order_estimated_delivery_date.max().normalize()
    assert (d.iso_weekday.values == dates.dt.dayofweek.values + 1).all()
    assert (d.year_month.values == dates.dt.strftime("%Y-%m").values).all()
    assert (d.month_start.values == dates.dt.to_period("M").dt.start_time.dt.strftime("%Y-%m-%d").values).all()
    assert set(d.date_key) >= set(db.df("fact_orders").date_key)       # every fact date joins to the dimension


def test_mart_category_monthly_sales():
    db, frames = _built()
    got = db.df("mart_category_monthly_sales")
    o, items = _truth(frames)
    cat = db.df("stg_products").set_index("product_id").category
    x = items.merge(o[["order_id", "order_status", "order_purchase_timestamp"]], on="order_id")
    x = x[~x.order_status.isin(["canceled", "unavailable"])]
    x["category"] = x.product_id.map(cat)
    x["month_start"] = x.order_purchase_timestamp.dt.to_period("M").dt.start_time.dt.strftime("%Y-%m-%d")
    exp = x.groupby(["month_start", "category"]).agg(orders=("order_id", "nunique"), revenue=("price", "sum")).reset_index()
    m = got.merge(exp, on=["month_start", "category"], suffixes=("", "_exp"), how="outer", indicator=True)
    assert (m._merge == "both").all(), m[m._merge != "both"].head()
    np.testing.assert_allclose(m.revenue, m.revenue_exp, rtol=1e-9)
    assert (m.orders == m.orders_exp).all()
    # month-over-month change is computed within each category, in time order
    exp = exp.sort_values(["category", "month_start"])
    exp["chg"] = exp.groupby("category").revenue.diff()
    chk = got.merge(exp[["month_start", "category", "chg"]], on=["month_start", "category"])
    both = chk.dropna(subset=["revenue_change_vs_prev_month", "chg"])
    np.testing.assert_allclose(both.revenue_change_vs_prev_month, both.chg, atol=1e-6)
    assert chk[chk.chg.isna()].revenue_change_vs_prev_month.isna().all()     # first month per category has no change
    assert not got.duplicated(["month_start", "category"]).any()


def test_mart_seller_performance():
    db, frames = _built()
    got = db.df("mart_seller_performance").set_index("seller_key")
    o, items = _truth(frames)
    fo = db.df("fact_orders").set_index("order_id")
    so = items[["seller_id", "order_id"]].drop_duplicates()                 # an order with 2 items from 1 seller counts once
    so = so.join(fo[["delivered_ts", "is_late", "review_score"]], on="order_id")
    g = so.groupby("seller_id")
    assert len(got) == so.seller_id.nunique()
    assert (got.orders == g.size()[got.index]).all()
    assert (got.delivered_orders == g.delivered_ts.count()[got.index]).all()
    late = so[so.delivered_ts.notna()].groupby("seller_id").is_late.mean()
    both = got.late_delivery_rate.dropna().index
    np.testing.assert_allclose(got.late_delivery_rate[both], late[both], atol=1e-9)
    assert got.late_delivery_rate.dropna().between(0, 1).all()
    np.testing.assert_allclose(got.avg_review_score.dropna(), g.review_score.mean()[got.avg_review_score.dropna().index], atol=1e-9)


def test_mart_delivery_vs_reviews():
    db, frames = _built()
    got = db.df("mart_delivery_vs_reviews").set_index("customer_state")
    fo = db.df("fact_orders").merge(db.df("dim_customers"), on="customer_key")
    fo = fo[fo.delivered_ts.notna()]
    g = fo.groupby("customer_state")
    assert (got.delivered_orders == g.size()[got.index]).all()
    np.testing.assert_allclose(got.late_rate, g.is_late.mean()[got.index], atol=1e-9)
    np.testing.assert_allclose(got.avg_delivery_days, g.delivery_days.mean()[got.index], atol=1e-6)
    assert (got.avg_review_when_late.dropna() < got.avg_review_when_on_time[got.avg_review_when_late.dropna().index]).all()


# ------------------------------------------------------------------ the analytics SQL files and consumers of the marts
def _run_analytics_sql(db, fname):
    sql = (ROOT / "sql" / "analytics" / fname).read_text().replace("analytics.", "").rstrip().rstrip(";")
    return db.df(transpile(sql))


def test_analytics_queries_run_and_make_sense():
    db, frames = _built(3000, 4500)       # seller query needs >= 30 delivered orders per seller -> full-size dataset
    q1 = _run_analytics_sql(db, "01_category_monthly_sales.sql")
    assert len(q1) > 0 and q1.groupby("month_start").size().max() <= 5 and q1.rank_in_month.min() == 1
    q2 = _run_analytics_sql(db, "02_late_delivery_hotspots.sql")
    assert 0 < len(q2) <= 20 and q2.late_rate.is_monotonic_decreasing and (q2.delivered_orders >= 30).all()
    overall = db.df("fact_orders").query("delivered_ts == delivered_ts").is_late.mean()
    assert q2.late_rate.iloc[0] > overall + 0.15                          # the 15 planted bad sellers surface at the top
    q3 = _run_analytics_sql(db, "03_delivery_vs_review.sql").set_index("delivery")
    assert q3.loc["late", "avg_review_score"] < q3.loc["on time", "avg_review_score"]
    q4 = _run_analytics_sql(db, "04_repeat_purchase.sql").iloc[0]
    o = frames["orders"].merge(frames["customers"], on="customer_id")
    o = o[~o.order_status.isin(["canceled", "unavailable"])]
    per = o.groupby("customer_unique_id").size()
    assert q4.customers == len(per) and abs(q4.repeat_rate - (per > 1).mean()) < 1e-9


def test_ml_db_query_columns_exist_in_marts():
    db, _ = _built()
    fo_cols, dc_cols = set(db.df("fact_orders").columns), set(db.df("dim_customers").columns)
    assert {"order_id", "purchase_ts", "order_value", "order_status", "customer_key"} <= fo_cols
    assert {"customer_unique_id", "customer_key"} <= dc_cols
    q = (ROOT / "ml" / "data.py").read_text()
    sql = re.search(r'q = """(.*?)"""', q, re.S).group(1).replace("analytics.", "")
    out = db.df(transpile(sql.replace("USING (customer_key)", "USING (customer_key)")))
    assert list(out.columns) == ["order_id", "customer_unique_id", "purchase_ts", "order_value", "order_status"]
    assert out.order_value.notna().all() and len(out) > 0


def test_powerbi_guide_references_real_tables_and_columns():
    guide = (ROOT / "dashboards" / "powerbi_guide.md").read_text()
    tables = set(re.findall(r"`([^`]*(?:fact_|dim_|mart_|ml_)[^`]*)`", guide)[0].replace("\n", " ").replace(" ", "").split(","))
    db, _ = _built()
    ml_tables = set(re.findall(r'"(ml_\w+)"', (ROOT / "ml" / "customer_segmentation.py").read_text() + (ROOT / "ml" / "sales_forecasting.py").read_text()))
    assert tables <= set(db.models) | ml_tables, tables - set(db.models) - ml_tables
    cols = set(db.df("fact_orders").columns)
    for ref in re.findall(r"fact_orders\[(\w+)\]", guide):
        assert ref in cols, f"guide references fact_orders[{ref}] which does not exist"
    fc_src = (ROOT / "ml" / "sales_forecasting.py").read_text()
    for token in ("actual", "backtest_prediction", "forecast", "week_end"):       # names the guide tells the user to colour/filter by
        assert f'"{token}"' in fc_src or f"'{token}'" in fc_src, token
    assert "analytics" in fc_src and "ml_sales_forecast" in fc_src
