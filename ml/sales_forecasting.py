"""Weekly sales forecasting with a chronological train/test split.

Compares a naive baseline (last week) and a 4-week moving average against gradient boosting,
reports MAE / RMSE / MAPE, tracks runs in MLflow, then forecasts the next N weeks.

    python -m ml.sales_forecasting [--csv-dir data/raw] [--horizon 8]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from ingestion.config import OUTPUT_DIR
from ml.data import load_orders
from ml.evaluation import mae, mape, rmse, tracked_run

LAGS = [1, 2, 3, 4]


def weekly_series(orders: pd.DataFrame) -> pd.Series:
    s = orders.set_index("purchase_ts").order_value.resample("W-SUN").sum()
    # first and last weeks are usually partial -> drop so they don't look like demand collapses
    return s.iloc[1:-1]


def features(history: pd.Series, week_end: pd.Timestamp) -> list[float]:
    lags = [history.iloc[-l] for l in LAGS]
    return lags + [float(np.mean(lags)), week_end.isocalendar().week]


def make_xy(s: pd.Series):
    rows, y, idx = [], [], []
    for t in range(max(LAGS), len(s)):
        rows.append(features(s.iloc[:t], s.index[t])); y.append(s.iloc[t]); idx.append(s.index[t])
    return np.array(rows), np.array(y), pd.DatetimeIndex(idx)


def run(csv_dir: str | None = None, horizon: int = 8, test_frac: float = 0.2, seed: int = 42) -> pd.DataFrame:
    s = weekly_series(load_orders(csv_dir))
    X, y, idx = make_xy(s)
    split = int(len(y) * (1 - test_frac))              # chronological split: no shuffling
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

    preds = {"naive_last_week": Xte[:, 0], "moving_avg_4w": Xte[:, 4]}
    params = dict(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=seed)
    gbr = GradientBoostingRegressor(**params).fit(Xtr, ytr)
    preds["gradient_boosting"] = gbr.predict(Xte)

    results = []
    for name, p in preds.items():
        m = {"mae": mae(yte, p), "rmse": rmse(yte, p), "mape": mape(yte, p)}
        with tracked_run("sales_forecasting", name, params if name == "gradient_boosting" else {"model": name}) as log:
            log(m)
        results.append({"model": name, **m})
    res = pd.DataFrame(results).set_index("model").round(2)
    print(res)
    best = res.mae.idxmin()
    print(f"best on test MAE: {best}  (a model that can't beat the naive baseline is not worth deploying)")

    # refit on everything, then forecast recursively
    final = GradientBoostingRegressor(**params).fit(X, y)
    hist = s.copy()
    future = []
    for _ in range(horizon):
        t = hist.index[-1] + pd.Timedelta(weeks=1)
        if best == "naive_last_week":
            val = hist.iloc[-1]
        elif best == "moving_avg_4w":
            val = hist.iloc[-4:].mean()
        else:
            val = max(float(final.predict([features(hist, t)])[0]), 0.0)
        hist.loc[t] = val; future.append((t, val))

    out = pd.concat([
        pd.DataFrame({"week_end": s.index, "sales": s.values, "kind": "actual"}),
        pd.DataFrame({"week_end": idx[split:], "sales": preds[best], "kind": "backtest_prediction"}),
        pd.DataFrame(future, columns=["week_end", "sales"]).assign(kind="forecast"),
    ])
    out["model"] = best
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    out.to_csv(OUTPUT_DIR / "sales_forecast.csv", index=False)
    res.to_csv(OUTPUT_DIR / "forecast_model_metrics.csv")
    if not csv_dir:
        from sqlalchemy import create_engine
        from ingestion.config import database_url
        out.to_sql("ml_sales_forecast", create_engine(database_url()), schema="analytics", if_exists="replace", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir"); ap.add_argument("--horizon", type=int, default=8)
    a = ap.parse_args(); run(a.csv_dir, a.horizon)
