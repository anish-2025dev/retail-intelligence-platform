# Retail Intelligence Platform

End-to-end data engineering, analytics and ML on the public [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
Runs fully locally (no cloud): PostgreSQL, dbt Core, Airflow, MLflow, scikit-learn, Power BI.

```
CSV -> extract -> validate (rules, audit log, rejected rows) -> raw.* (PostgreSQL)
    -> dbt staging -> dimensional marts (analytics.*) -> Power BI
                                  \-> RFM segmentation + weekly sales forecasting -> MLflow
Airflow orchestrates: load -> dbt run -> dbt test -> ML -> report
```

## Quick start (no database needed)
```bash
pip install -r requirements.txt
python -m ingestion.generate_sample_data      # synthetic Olist-shaped data with injected defects
python -m ingestion.load --dry-run            # validation report -> outputs/data_quality_report.csv
python -m ml.customer_segmentation --csv-dir data/raw
python -m ml.sales_forecasting --csv-dir data/raw
python -m pytest -q tests
```

## Full run with the real data
1. Download the Olist CSVs from Kaggle into `data/raw/` (replace the synthetic files).
2. `docker compose up -d postgres mlflow` (MLflow UI: http://localhost:5000)
3. `python -m ingestion.load` -> validated tables in `raw.*`, audit trail in `audit.*`
4. `cd transformations && dbt run --profiles-dir . && dbt test --profiles-dir .`
5. `python -m ml.customer_segmentation && python -m ml.sales_forecasting` (writes `analytics.ml_*` tables)
6. Connect Power BI: see `dashboards/powerbi_guide.md`
7. Or run it all on a schedule: `docker compose up -d` then open Airflow at http://localhost:8080 (DAG `retail_intelligence_pipeline`; the admin password is printed in the airflow container logs).

## Testing
```bash
python -m pytest -q tests          # or, without pytest:  python tests/run_all.py   (~40 s)
python tests/run_all.py -m dbt     # one module;  -k <text> filters by test name
```
| Module | Covers |
|---|---|
| `test_validation` | every validation rule (reject vs warn), cascades, rejected-record log, edge cases, boundaries |
| `test_ml` | metrics, RFM (hand-calculated), K-Means helpers, forecast features, chronological split, MLflow logging |
| `test_ingestion` | config, extract, sample-data generator |
| `test_load_audit` | `load.py` and `report.py` against a fake DB engine: audit trail, transactions, failure handling, DDL consistency |
| `test_dbt_models` | all dbt models and `schema.yml` tests run on SQLite, cross-checked against independent pandas calculations |
| `test_orchestration_and_packaging` | Airflow DAG graph, docker-compose, PK/FK constraints vs data, requirements, documented commands |
| `test_pipeline_integration` | real CLI end-to-end, reproducibility, messy data, scale (600k+ rows) |

**What the tests cannot prove:** PostgreSQL itself, `dbt run` on PostgreSQL, a running Airflow/Docker stack and Power BI.
The dbt models are tested on SQLite through a small dialect shim, and Airflow and SQLAlchemy are stubbed.
Run `make up load dbt` once against a real PostgreSQL to close that gap.

## Layout
| Path | Purpose |
|---|---|
| `ingestion/` | extract, load, audit logging, sample-data generator |
| `data_quality/validation.py` | all validation rules (reject vs warn) |
| `sql/` | schema, constraints/indexes, analytical queries (CTEs, window functions) |
| `transformations/` | dbt project: staging, dimensions, facts, marts, tests |
| `ml/` | RFM + K-Means, forecasting with baselines and chronological split, MLflow |
| `orchestration/dags/` | Airflow DAG |
| `governance/` | data dictionary, limitations |
| `docs/` | business requirements, findings template, presentation outline |

## Status and honest caveats
- Tested here: ingestion, validation rules, unit tests, segmentation and forecasting on synthetic data.
- **Not run here:** PostgreSQL load, dbt models, Airflow and Docker (no database or network in the build environment). Expect small fixes on first run.
- Synthetic-data results mean nothing. Write your findings (`docs/findings_template.md`) only from the real Olist run.
- Power BI dashboards must be built in Power BI Desktop; the guide lists pages, tables and DAX.
- No cost data in Olist, so no profit analysis.
