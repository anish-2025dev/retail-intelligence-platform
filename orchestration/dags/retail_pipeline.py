"""Airflow DAG: extract/validate/load -> dbt run -> dbt test -> ML -> pipeline report.

A failed validation or failed dbt test stops the DAG, so bad data never reaches the marts/ML outputs.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/project"
DBT = f"cd {PROJECT}/transformations && dbt {{cmd}} --profiles-dir . --project-dir ."
ENV = {"PYTHONPATH": PROJECT}

default_args = {"owner": "data-team", "retries": 2, "retry_delay": timedelta(minutes=2)}

with DAG(
    dag_id="retail_intelligence_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["retail", "portfolio"],
) as dag:
    load_and_validate = BashOperator(
        task_id="extract_validate_load", cwd=PROJECT, env=ENV, append_env=True,
        bash_command="python -m ingestion.load")
    dbt_run = BashOperator(task_id="dbt_run", bash_command=DBT.format(cmd="run"), env=ENV, append_env=True)
    dbt_test = BashOperator(task_id="dbt_test", bash_command=DBT.format(cmd="test"), env=ENV, append_env=True)
    segmentation = BashOperator(task_id="customer_segmentation", cwd=PROJECT, env=ENV, append_env=True,
                                bash_command="python -m ml.customer_segmentation")
    forecasting = BashOperator(task_id="sales_forecasting", cwd=PROJECT, env=ENV, append_env=True,
                               bash_command="python -m ml.sales_forecasting")
    report = BashOperator(task_id="pipeline_report", cwd=PROJECT, env=ENV, append_env=True,
                          bash_command="python -m ingestion.report")

    load_and_validate >> dbt_run >> dbt_test >> [segmentation, forecasting] >> report
