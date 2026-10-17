.PHONY: install sample dryrun test test-lite up load dbt ml report
install:  ; pip install -r requirements.txt
sample:   ; python -m ingestion.generate_sample_data
dryrun:   ; python -m ingestion.load --dry-run
test:     ; python -m pytest -q tests
test-lite: ; python tests/run_all.py   # no pytest needed
up:       ; docker compose up -d postgres mlflow
load:     ; python -m ingestion.load
dbt:      ; cd transformations && dbt run --profiles-dir . && dbt test --profiles-dir .
ml:       ; python -m ml.customer_segmentation && python -m ml.sales_forecasting
report:   ; python -m ingestion.report
