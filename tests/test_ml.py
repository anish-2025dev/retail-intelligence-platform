"""Unit tests for ml/*: metrics, RFM, K-Means helpers, forecasting features, chronological split, data loading."""
import sys
import types
from unittest import mock

import numpy as np
import pandas as pd

from ml import customer_segmentation as seg
from ml import sales_forecasting as fc
from ml.data import EXCLUDED_STATUSES, from_csv, load_orders
from ml.evaluation import mae, mape, rmse, tracked_run


# ------------------------------------------------------------------ metrics
def test_metrics_known_values():
    y, p = [100, 200, 300], [110, 190, 330]
    assert mae(y, p) == (10 + 10 + 30) / 3
    assert abs(rmse(y, p) - np.sqrt((100 + 100 + 900) / 3)) < 1e-9
    assert abs(mape(y, p) - np.mean([10 / 100, 10 / 200, 30 / 300]) * 100) < 1e-9


def test_mape_ignores_zero_actuals_and_handles_all_zero():
    assert mape([0, 100], [5, 110]) == 10.0          # the zero-actual point is excluded, not divided by
    assert np.isnan(mape([0, 0], [1, 2]))


def test_perfect_prediction_has_zero_error():
    y = [1.0, 2.0, 3.0]
    assert mae(y, y) == 0 and rmse(y, y) == 0 and mape(y, y) == 0


def test_tracked_run_is_noop_without_mlflow():
    with mock.patch.dict(sys.modules, {"mlflow": None}):   # None => ImportError on import
        with tracked_run("exp", "run", {"a": 1}) as log:
            log({"mae": 1.0})                              # must not raise


def test_tracked_run_logs_to_mlflow_and_drops_nan():
    fake = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"mlflow": fake}):
        with tracked_run("exp", "run1", {"k": 3}) as log:
            log({"mae": 2.0, "mape": float("nan")})
    fake.set_experiment.assert_called_once_with("exp")
    fake.start_run.assert_called_once_with(run_name="run1")
    fake.log_params.assert_called_once_with({"k": 3})
    fake.log_metrics.assert_called_once_with({"mae": 2.0})   # NaN removed (MLflow would reject it)


# ------------------------------------------------------------------ RFM / clustering
def _orders():
    t = pd.Timestamp
    return pd.DataFrame({
        "order_id": ["a", "b", "c", "d"],
        "customer_unique_id": ["u1", "u1", "u2", "u3"],
        "purchase_ts": [t("2024-01-01"), t("2024-01-11"), t("2024-01-05"), t("2023-12-01")],
        "order_value": [100.0, 50.0, 20.0, 500.0],
        "order_status": "delivered",
    })


def test_build_rfm_hand_calculated():
    rfm = seg.build_rfm(_orders()).set_index("customer_unique_id")
    # snapshot = last purchase (2024-01-11) + 1 day = 2024-01-12
    assert rfm.loc["u1"].tolist() == [1, 2, 150.0]       # last bought 2024-01-11 -> 1 day; 2 orders; 150 spent
    assert rfm.loc["u2"].tolist() == [7, 1, 20.0]
    assert rfm.loc["u3"].tolist() == [42, 1, 500.0]


def test_build_rfm_counts_distinct_orders_per_customer():
    o = pd.concat([_orders(), _orders().iloc[[0]]])      # duplicated order row must not inflate frequency
    assert seg.build_rfm(o).set_index("customer_unique_id").loc["u1", "frequency"] == 2


def test_choose_k_recovers_three_well_separated_clusters():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.2, (100, 3)) for c in ([0, 0, 0], [6, 6, 6], [12, 0, 12])])
    k, scores = seg.choose_k(X)
    assert k == 3 and set(scores) == set(range(2, 7))
    assert all(-1 <= v <= 1 for v in scores.values())


def test_choose_k_is_deterministic():
    X = np.random.default_rng(1).normal(size=(300, 3))
    assert seg.choose_k(X, seed=7) == seg.choose_k(X, seed=7)


def test_suggested_segment_names_are_unique_per_cluster():
    # 5 clusters, several share the same relationship to the averages -> labels must still be distinguishable
    profile = pd.DataFrame({
        "customers": [100, 100, 100, 100, 100],
        "recency_days": [300, 310, 320, 20, 30],
        "frequency": [1.0, 1.0, 1.1, 1.0, 2.0],
        "monetary": [50, 60, 70, 55, 900],
    })
    names = seg.suggest_names(profile)
    assert set(names) == set(profile.index)
    assert len(set(names.values())) == len(names), names


def _patched_segmentation_run(tmp_path, orders, **kw):
    with mock.patch.object(seg, "load_orders", return_value=orders), mock.patch.object(seg, "OUTPUT_DIR", tmp_path):
        return seg.run(csv_dir="x", **kw)


def _synthetic_orders(n_customers=400, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_customers):
        n = rng.choice([1, 1, 1, 2, 4])
        for j in range(n):
            rows.append((f"o{i}_{j}", f"u{i}", pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 600))),
                         float(rng.gamma(2, 60) + 10), "delivered"))
    return pd.DataFrame(rows, columns=["order_id", "customer_unique_id", "purchase_ts", "order_value", "order_status"])


def test_segmentation_run_outputs_are_consistent(tmp_path):
    orders = _synthetic_orders()
    rfm = _patched_segmentation_run(tmp_path, orders)
    assert len(rfm) == orders.customer_unique_id.nunique()            # every customer assigned exactly once
    assert rfm.customer_unique_id.is_unique and rfm.cluster.notna().all() and rfm.segment.notna().all()
    prof = pd.read_csv(tmp_path / "customer_segment_profiles.csv", index_col=0)
    assert prof.customers.sum() == len(rfm)
    assert set(prof.index) == set(rfm.cluster.unique())
    assert (tmp_path / "customer_segments.csv").exists()


def test_segmentation_is_reproducible(tmp_path):
    orders = _synthetic_orders()
    a = _patched_segmentation_run(tmp_path, orders, seed=11)
    b = _patched_segmentation_run(tmp_path, orders, seed=11)
    pd.testing.assert_frame_equal(a, b)


def test_segmentation_with_a_single_order_per_customer_does_not_crash(tmp_path):
    o = _synthetic_orders().drop_duplicates("customer_unique_id")      # frequency is constant (=1) -> zero variance feature
    rfm = _patched_segmentation_run(tmp_path, o)
    assert len(rfm) == len(o)


# ------------------------------------------------------------------ forecasting
def _daily_orders(weeks=70, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2023-01-02", periods=weeks * 7, freq="D")        # starts on a Monday
    base = 50 + 0.3 * np.arange(len(days)) + 10 * np.sin(np.arange(len(days)) / 7 * 2 * np.pi / 8)
    vals = np.clip(base + rng.normal(0, 3, len(days)), 1, None)
    return pd.DataFrame({"order_id": range(len(days)), "customer_unique_id": "u", "purchase_ts": days,
                         "order_value": vals, "order_status": "delivered"})


def test_weekly_series_drops_partial_first_and_last_weeks():
    o = _daily_orders(weeks=10)
    full = o.set_index("purchase_ts").order_value.resample("W-SUN").sum()
    s = fc.weekly_series(o)
    assert len(s) == len(full) - 2 and s.index[0] == full.index[1] and s.index[-1] == full.index[-2]


def test_weekly_series_fills_empty_weeks_with_zero():
    o = _daily_orders(weeks=10)
    o = o[~o.purchase_ts.between("2023-02-06", "2023-02-19")]               # two weeks with no orders
    s = fc.weekly_series(o)
    assert (s == 0).sum() == 2


def test_make_xy_features_use_only_the_past():
    s = pd.Series(np.arange(10, 30, dtype=float), index=pd.date_range("2024-01-07", periods=20, freq="W-SUN"))
    X, y, idx = fc.make_xy(s)
    assert len(y) == len(s) - 4 and idx[0] == s.index[4]
    for row, target, t in zip(X, y, range(4, len(s))):
        assert row[:4].tolist() == [s.iloc[t - 1], s.iloc[t - 2], s.iloc[t - 3], s.iloc[t - 4]]
        assert target == s.iloc[t] and target not in row[:5]                 # target never leaks into its own features
        assert row[4] == np.mean(row[:4])


def test_forecast_run_chronological_split_and_future_weeks(tmp_path):
    o = _daily_orders(weeks=70)
    with mock.patch.object(fc, "load_orders", return_value=o), mock.patch.object(fc, "OUTPUT_DIR", tmp_path):
        out = fc.run(csv_dir="x", horizon=8)
    actual = out[out.kind == "actual"].reset_index(drop=True)
    back = out[out.kind == "backtest_prediction"]
    future = out[out.kind == "forecast"].reset_index(drop=True)

    n_xy = len(actual) - 4
    split = int(n_xy * 0.8)
    assert back.week_end.min() == actual.week_end.iloc[4 + split]             # test window is the *last* 20% of weeks
    assert len(back) == n_xy - split
    assert len(future) == 8
    assert future.week_end.min() > actual.week_end.max()                      # forecasts are strictly in the future
    assert (future.week_end.diff().dropna() == pd.Timedelta(weeks=1)).all()   # consecutive weeks
    assert (future.sales >= 0).all() and future.sales.notna().all()
    assert 0.5 * actual.sales.tail(4).mean() < future.sales.mean() < 1.8 * actual.sales.tail(4).mean()


def test_forecast_picks_model_with_lowest_test_mae(tmp_path):
    o = _daily_orders(weeks=70)
    with mock.patch.object(fc, "load_orders", return_value=o), mock.patch.object(fc, "OUTPUT_DIR", tmp_path):
        out = fc.run(csv_dir="x", horizon=4)
    metrics = pd.read_csv(tmp_path / "forecast_model_metrics.csv", index_col=0)
    assert set(metrics.index) == {"naive_last_week", "moving_avg_4w", "gradient_boosting"}
    assert out.model.unique().tolist() == [metrics.mae.idxmin()]
    assert (metrics[["mae", "rmse"]] >= 0).all().all() and (metrics.rmse >= metrics.mae - 1e-9).all()


def test_forecast_is_reproducible(tmp_path):
    o = _daily_orders(weeks=60)
    with mock.patch.object(fc, "load_orders", return_value=o), mock.patch.object(fc, "OUTPUT_DIR", tmp_path):
        a, b = fc.run(csv_dir="x", horizon=4), fc.run(csv_dir="x", horizon=4)
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------------------ data loading
def test_from_csv_order_value_matches_independent_calculation(tmp_path):
    from ingestion.generate_sample_data import generate
    from ingestion.extract import extract
    generate(tmp_path, n_customers=200, n_orders=300)
    got = from_csv(tmp_path).set_index("order_id").order_value

    items = pd.read_csv(tmp_path / "olist_order_items_dataset.csv")
    items = items[(items.price > 0) & (items.freight_value >= 0) & (items.product_id != "p_missing")]
    expected = (items.price + items.freight_value).groupby(items.order_id).sum()
    common = got.index.intersection(expected.index)
    assert len(common) > 250
    np.testing.assert_allclose(got[common].values, expected[common].values, rtol=1e-9)
    assert got.index.is_unique


def test_load_orders_excludes_canceled_and_unavailable():
    df = pd.DataFrame({"order_id": list("abcd"), "customer_unique_id": "u", "purchase_ts": pd.Timestamp("2024-01-01"),
                       "order_value": 1.0, "order_status": ["delivered", "canceled", "unavailable", "shipped"]})
    with mock.patch("ml.data.from_csv", return_value=df):
        out = load_orders("x")
    assert out.order_id.tolist() == ["a", "d"] and EXCLUDED_STATUSES == {"canceled", "unavailable"}


def test_load_orders_db_mode_queries_analytics_schema():
    df = pd.DataFrame({"order_id": ["a", "b"], "customer_unique_id": "u", "purchase_ts": pd.Timestamp("2024-01-01"),
                       "order_value": 1.0, "order_status": ["delivered", "canceled"]})
    fake_sa = types.SimpleNamespace(create_engine=mock.MagicMock(return_value="ENGINE"))
    with mock.patch.dict(sys.modules, {"sqlalchemy": fake_sa}), mock.patch("pandas.read_sql", return_value=df) as rs:
        out = load_orders()
    sql = rs.call_args[0][0]
    assert "analytics.fact_orders" in sql and "analytics.dim_customers" in sql and "order_value IS NOT NULL" in sql
    assert out.order_id.tolist() == ["a"]


def test_model_is_trained_only_on_the_earliest_weeks_in_order(tmp_path):
    """Spy on fit(): the evaluation fit must see exactly the first 80% of weeks (no shuffling, no future data)."""
    fits = []

    class Spy(fc.GradientBoostingRegressor):
        def fit(self, X, y, *a, **k):
            fits.append((np.array(X), np.array(y)))
            return super().fit(X, y, *a, **k)

    o = _daily_orders(weeks=70)
    with mock.patch.object(fc, "load_orders", return_value=o), mock.patch.object(fc, "OUTPUT_DIR", tmp_path), \
         mock.patch.object(fc, "GradientBoostingRegressor", Spy):
        fc.run(csv_dir="x", horizon=2)
    s = fc.weekly_series(o)
    X_all, y_all, _ = fc.make_xy(s)
    split = int(len(y_all) * 0.8)
    (Xtr, ytr), (Xfull, yfull) = fits[0], fits[1]
    np.testing.assert_array_equal(ytr, y_all[:split])                 # first 80%, original order
    np.testing.assert_array_equal(Xtr, X_all[:split])
    np.testing.assert_array_equal(yfull, y_all)                       # only the final refit may use everything
    assert len(ytr) < len(yfull)
