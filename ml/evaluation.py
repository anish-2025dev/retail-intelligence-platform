"""Shared metrics and (optional) MLflow logging."""
from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np


def mae(y, p): return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mape(y, p):
    """MAPE over points where actual > 0 (undefined otherwise)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y > 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100) if m.any() else float("nan")


@contextmanager
def tracked_run(experiment: str, run_name: str, params: dict | None = None):
    """Yield a `log(metrics)` callable. Logs to MLflow if installed, else a no-op."""
    try:
        import mlflow
    except ImportError:
        yield lambda metrics: None
        return
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        yield lambda metrics: mlflow.log_metrics({k: v for k, v in metrics.items() if v == v})
