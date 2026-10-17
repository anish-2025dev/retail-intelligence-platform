"""Structure tests for the Airflow DAG, docker-compose, SQL constraints, dependencies and documented commands."""
import ast
import importlib.util
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ Airflow DAG (airflow itself is stubbed)
class _FakeDAG:
    current = None
    def __init__(self, **kw): self.kw, self.tasks = kw, {}
    def __enter__(self): _FakeDAG.current = self; return self
    def __exit__(self, *a): _FakeDAG.current = None


class _FakeOp:
    def __init__(self, task_id, **kw):
        self.task_id, self.kw, self.up, self.down = task_id, kw, set(), set()
        _FakeDAG.current.tasks[task_id] = self
    def __rshift__(self, other):
        for o in (other if isinstance(other, list) else [other]):
            self.down.add(o.task_id); o.up.add(self.task_id)
        return other
    def __rrshift__(self, other):                       # [a, b] >> op
        for o in other:
            o.down.add(self.task_id); self.up.add(o.task_id)
        return self


def _load_dag():
    mods = {"airflow": types.ModuleType("airflow"), "airflow.operators": types.ModuleType("airflow.operators"),
            "airflow.operators.bash": types.ModuleType("airflow.operators.bash")}
    mods["airflow"].DAG = _FakeDAG
    mods["airflow.operators.bash"].BashOperator = _FakeOp
    with mock.patch.dict(sys.modules, mods):
        spec = importlib.util.spec_from_file_location("retail_dag", ROOT / "orchestration" / "dags" / "retail_pipeline.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_dag_structure_and_dependencies():
    dag = _load_dag().dag
    t = dag.tasks
    assert set(t) == {"extract_validate_load", "dbt_run", "dbt_test", "customer_segmentation", "sales_forecasting", "pipeline_report"}
    assert t["extract_validate_load"].up == set()
    assert t["dbt_run"].up == {"extract_validate_load"} and t["dbt_test"].up == {"dbt_run"}
    assert t["customer_segmentation"].up == {"dbt_test"} and t["sales_forecasting"].up == {"dbt_test"}   # ML only after tests pass
    assert t["pipeline_report"].up == {"customer_segmentation", "sales_forecasting"} and t["pipeline_report"].down == set()
    # acyclic: repeatedly remove nodes with no remaining upstream
    left = {k: set(v.up) for k, v in t.items()}
    while left:
        ready = [k for k, u in left.items() if not u]
        assert ready, "cycle in DAG"
        for k in ready: left.pop(k)
        for u in left.values(): u.difference_update(ready)


def test_dag_settings_and_commands():
    mod = _load_dag(); dag = mod.dag
    assert dag.kw["dag_id"] == "retail_intelligence_pipeline" and dag.kw["schedule"] == "@daily" and dag.kw["catchup"] is False
    assert dag.kw["default_args"]["retries"] == 2
    t = dag.tasks
    assert t["extract_validate_load"].kw["bash_command"] == "python -m ingestion.load"
    assert "dbt run --profiles-dir . --project-dir ." in t["dbt_run"].kw["bash_command"]
    assert "dbt test --profiles-dir . --project-dir ." in t["dbt_test"].kw["bash_command"]
    assert t["dbt_run"].kw["bash_command"].startswith("cd /opt/airflow/project/transformations &&")
    for task in t.values():                                       # every task can import the project and keeps the container env
        assert task.kw["env"]["PYTHONPATH"] == "/opt/airflow/project" and task.kw["append_env"] is True
    modules = re.findall(r"python -m ([\w.]+)", " ".join(x.kw["bash_command"] for x in t.values()))
    for m in modules:
        path = ROOT / (m.replace(".", "/") + ".py")
        assert path.exists() and "__main__" in path.read_text(), m


def test_dag_paths_match_docker_compose():
    mod = _load_dag()
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["airflow"]
    assert any(v.endswith(":" + mod.PROJECT) for v in compose["volumes"])
    assert compose["environment"]["AIRFLOW__CORE__DAGS_FOLDER"] == f"{mod.PROJECT}/orchestration/dags"


# ------------------------------------------------------------------ docker-compose / Dockerfile / env
def test_docker_compose_is_well_formed():
    doc = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    svc = doc["services"]
    assert set(svc) == {"postgres", "mlflow", "airflow"}
    assert svc["airflow"]["depends_on"]["postgres"]["condition"] == "service_healthy" and "healthcheck" in svc["postgres"]
    assert svc["airflow"]["environment"]["POSTGRES_HOST"] == "postgres"          # service name, not localhost, inside the network
    assert svc["airflow"]["environment"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    host_ports = [p.split(":")[0] for s in svc.values() for p in s.get("ports", [])]
    assert len(host_ports) == len(set(host_ports)) == 3 and set(host_ports) == {"5432", "5000", "8080"}
    declared = set(doc["volumes"])
    used = {v.split(":")[0] for s in svc.values() for v in s.get("volumes", []) if not v.startswith(".")}
    assert used <= declared, used - declared
    assert "${POSTGRES_USER:-retail}" in svc["postgres"]["environment"]["POSTGRES_USER"]
    assert (ROOT / svc["airflow"]["build"]["dockerfile"]).exists()
    assert "COPY requirements.txt" in (ROOT / "Dockerfile.airflow").read_text() and (ROOT / "requirements.txt").exists()


def test_env_example_matches_config_and_compose():
    env = dict(l.split("=") for l in (ROOT / ".env.example").read_text().split() if "=" in l)
    cfg = (ROOT / "ingestion" / "config.py").read_text()
    for key in env:
        assert key in cfg, f"{key} in .env.example is unused by config.py"
    for key in re.findall(r'getenv\("(POSTGRES_\w+)"', cfg):
        assert key in env, f"config.py reads {key} but .env.example does not define it"
    dbt_profile = (ROOT / "transformations" / "profiles.yml").read_text()
    for key in env:
        assert key in dbt_profile, f"dbt profile ignores {key}"


def test_gitignore_protects_data_and_artifacts():
    g = (ROOT / ".gitignore").read_text()
    for pat in ("data/raw/*.csv", ".env", "mlruns/", "outputs/", "transformations/target/"):
        assert pat in g, pat


# ------------------------------------------------------------------ SQL constraints vs the real (validated) data
def test_pk_fk_constraints_would_succeed_on_validated_data():
    from tests.dbt_sqlite_harness import clean_frames_from_generated
    frames, _ = clean_frames_from_generated(Path(tempfile.mkdtemp()), 800, 1200)
    sql = (ROOT / "sql" / "constraints_and_indexes.sql").read_text()
    pks = re.findall(r"ALTER TABLE raw\.(\w+)\s+ADD PRIMARY KEY \(([^)]*)\)", sql)
    fks = re.findall(r"ALTER TABLE raw\.(\w+)\s+ADD FOREIGN KEY \((\w+)\)\s+REFERENCES raw\.(\w+)\((\w+)\)", sql)
    idx = re.findall(r"ON raw\.(\w+) \((\w+)\)", sql)
    assert len(pks) == 5 and len(fks) == 6 and len(idx) == 4
    for table, cols in pks:
        cols = [c.strip() for c in cols.split(",")]
        df = frames[table]
        assert set(cols) <= set(df.columns), (table, cols)
        assert df[cols].notna().all().all(), f"PK on raw.{table} would fail: NULLs present"
        assert not df.duplicated(cols).any(), f"PK on raw.{table} would fail: duplicates present"
    pk_tables = {t for t, _ in pks}
    for child, ccol, parent, pcol in fks:
        assert child in frames and parent in frames and ccol in frames[child].columns and pcol in frames[parent].columns
        vals = frames[child][ccol]
        assert vals.notna().all(), f"FK {child}.{ccol}: NULLs"
        assert vals.isin(frames[parent][pcol]).all(), f"FK {child}.{ccol} -> {parent}.{pcol} has orphans after validation"
    for table, col in idx:
        assert col in frames[table].columns, (table, col)
    # referenced parent columns must be primary-key columns (PostgreSQL requires a unique constraint)
    pk_cols = {(t, c.strip()) for t, cols in pks for c in cols.split(",")}
    for _, _, parent, pcol in fks:
        assert (parent, pcol) in pk_cols, f"FK target raw.{parent}({pcol}) is not a primary key"


# ------------------------------------------------------------------ dependencies & documented commands
_PKG = {"sklearn": "scikit-learn", "yaml": "pyyaml", "psycopg2": "psycopg2-binary"}
_STDLIB = set(sys.stdlib_module_names)


def test_requirements_cover_all_third_party_imports():
    reqs = {re.split(r"[<>=\[]", l)[0].strip().lower() for l in (ROOT / "requirements.txt").read_text().splitlines() if l.strip()}
    local = {p.name for p in ROOT.iterdir() if p.is_dir()} | {"tests"}
    needed = set()
    for d in ("ingestion", "data_quality", "ml", "tests"):
        for f in (ROOT / d).glob("*.py"):
            for node in ast.walk(ast.parse(f.read_text())):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                        [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else []
                for n in names:
                    top = n.split(".")[0]
                    if top not in _STDLIB and top not in local:
                        needed.add(_PKG.get(top, top).lower())
    needed.discard("airflow"); needed.discard("mlflow") if "mlflow" in reqs else None
    assert needed <= reqs, f"missing from requirements.txt: {needed - reqs}"
    assert "tabulate" in reqs and "to_markdown" in (ROOT / "ingestion" / "report.py").read_text()   # pandas.to_markdown needs it
    assert {"dbt-postgres", "mlflow", "psycopg2-binary", "sqlalchemy"} <= reqs                       # used by DB/dbt/MLflow paths


def test_documented_commands_exist():
    text = (ROOT / "README.md").read_text() + (ROOT / "Makefile").read_text()
    modules = set(re.findall(r"python -m ([a-z_]+\.[a-z_]+)", text)) - {"pytest"}
    assert {"ingestion.generate_sample_data", "ingestion.load", "ml.customer_segmentation", "ml.sales_forecasting", "ingestion.report"} <= modules
    for m in modules:
        src = (ROOT / (m.replace(".", "/") + ".py"))
        assert src.exists(), m
        body = src.read_text()
        flags = set(re.findall(rf"python -m {re.escape(m)}[^\n`]*?(--[a-z-]+)", text))
        for flag in flags:
            assert f'"{flag}"' in body, f"{m} documented with {flag} but does not define it"
    compose_services = set(yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"])
    for services in re.findall(r"docker compose up -d ([a-z ]+)", text):
        assert set(services.split()) <= compose_services, services
    for target in re.findall(r"^([a-z]+):", (ROOT / "Makefile").read_text(), re.M):
        assert target in (ROOT / "Makefile").read_text().split(".PHONY:")[1].split("\n")[0]


def test_every_cli_responds_to_help():
    for m in ("ingestion.generate_sample_data", "ingestion.load", "ml.customer_segmentation", "ml.sales_forecasting"):
        r = subprocess.run([sys.executable, "-m", m, "--help"], cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0 and "usage" in r.stdout.lower(), (m, r.stderr[-300:])


def test_data_dictionary_covers_every_raw_and_mart_table():
    dd = (ROOT / "governance" / "data_dictionary.md").read_text()
    for t in ("orders", "customers", "order_items", "payments", "reviews", "products", "sellers", "category_translation"):
        assert f"raw.{t}" in dd, t
    for m in ("fact_orders", "fact_order_items", "dim_customers", "dim_date", "mart_category_monthly_sales",
              "mart_seller_performance", "mart_delivery_vs_reviews"):
        assert m in dd, m
    for f in ("audit.validation_results", "audit.rejected_records", "audit.pipeline_runs"):
        assert f in dd
