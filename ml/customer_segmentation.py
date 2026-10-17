"""RFM customer segmentation with K-Means.

    python -m ml.customer_segmentation                  # reads analytics.* from PostgreSQL
    python -m ml.customer_segmentation --csv-dir data/raw

NOTE: cluster names are *suggestions* from a simple centroid heuristic. Inspect
outputs/customer_segment_profiles.csv and rename segments yourself before presenting them.
In Olist most customers buy once, so frequency is low for almost everyone - say so in your findings.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ingestion.config import OUTPUT_DIR
from ml.data import load_orders
from ml.evaluation import tracked_run


def build_rfm(orders: pd.DataFrame) -> pd.DataFrame:
    snapshot = orders.purchase_ts.max() + pd.Timedelta(days=1)
    return orders.groupby("customer_unique_id").agg(
        recency_days=("purchase_ts", lambda s: (snapshot - s.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("order_value", "sum"),
    ).reset_index()


def choose_k(X: np.ndarray, k_range=range(2, 7), seed: int = 42):
    scores = {}
    sample = X if len(X) <= 10000 else X[np.random.default_rng(seed).choice(len(X), 10000, replace=False)]
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(sample)
        scores[k] = float(silhouette_score(sample, labels))
    return max(scores, key=scores.get), scores


def suggest_names(profile: pd.DataFrame) -> dict[int, str]:
    """Descriptive starting labels based on each centroid vs the customer-weighted average.
    Rename these yourself after inspecting the clusters (e.g. 'Lapsed high spenders')."""
    w = profile.customers / profile.customers.sum()
    avg = {c: float((profile[c] * w).sum()) for c in ("recency_days", "frequency", "monetary")}
    names = {}
    for i, r in profile.iterrows():
        spend = "higher spend" if r.monetary > avg["monetary"] else "lower spend"
        freq = "repeat buyers" if r.frequency > avg["frequency"] else "fewer orders"
        rec = "more recent" if r.recency_days < avg["recency_days"] else "less recent"
        names[i] = f"{spend}, {freq}, {rec}"
    counts = pd.Series(names).value_counts()
    # clusters can share the same coarse description; keep every label unique so charts/legends stay unambiguous
    return {i: (n if counts[n] == 1 else f"{n} [cluster {i}]") for i, n in names.items()}


def run(csv_dir: str | None = None, seed: int = 42) -> pd.DataFrame:
    orders = load_orders(csv_dir)
    rfm = build_rfm(orders)
    X = StandardScaler().fit_transform(np.log1p(rfm[["recency_days", "frequency", "monetary"]]))
    k, sil = choose_k(X, seed=seed)
    with tracked_run("customer_segmentation", f"kmeans_k{k}", {"k": k, "features": "log1p RFM", "seed": seed}) as log:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rfm["cluster"] = model.labels_
        log({"silhouette": sil[k], **{f"silhouette_k{kk}": v for kk, v in sil.items()}})
    profile = rfm.groupby("cluster").agg(customers=("customer_unique_id", "count"), recency_days=("recency_days", "mean"),
                                         frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
    names = suggest_names(profile)
    rfm["segment"] = rfm.cluster.map(names)
    profile["suggested_segment"] = profile.index.map(names)

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    rfm.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)
    profile.to_csv(OUTPUT_DIR / "customer_segment_profiles.csv")
    print(f"k={k}, silhouette={sil[k]:.3f}  (scores by k: { {a: round(b, 3) for a, b in sil.items()} })")
    print(profile)
    if not csv_dir:
        from sqlalchemy import create_engine
        from ingestion.config import database_url
        rfm.to_sql("ml_customer_segments", create_engine(database_url()), schema="analytics", if_exists="replace", index=False)
    return rfm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir")
    run(ap.parse_args().csv_dir)
