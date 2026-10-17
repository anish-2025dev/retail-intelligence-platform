"""Run the project's dbt models and schema.yml tests on SQLite (dbt/PostgreSQL are not available in the sandbox).

What this checks: model dependencies, column names, join grain/fan-out, null handling and business logic.
What this does NOT check: PostgreSQL syntax/semantics. A small dialect shim translates the Postgres-isms used in the
models (::casts, date_trunc, extract, to_char, generate_series) into SQLite equivalents.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import jinja2
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DBT_DIR = ROOT / "transformations"


def _operand_start(sql: str, end: int) -> int:
    """Index where the operand that ends at `end` (exclusive) begins: identifier chain or balanced call/paren."""
    i = end - 1
    if sql[i] == ")":
        depth = 0
        while i >= 0:
            depth += sql[i] == ")"
            depth -= sql[i] == "("
            if depth == 0:
                break
            i -= 1
        while i > 0 and (sql[i - 1].isalnum() or sql[i - 1] == "_"):   # include function name
            i -= 1
        return i
    while i > 0 and (sql[i - 1].isalnum() or sql[i - 1] in "_."):
        i -= 1
    return i


def transpile(sql: str) -> str:
    # date spine: generate_series -> recursive CTE
    if "generate_series" in sql:
        sql = sql.replace("with bounds as (", "with recursive bounds as (", 1)
        sql = sql.replace("\n)\nselect d::date", "\n),\nseries(d) as (select d0 from bounds union all "
                          "select date(d, '+1 day') from series, bounds where d < d1)\nselect d::date", 1)
        sql = re.sub(r"from bounds, generate_series\(d0, d1, interval '1 day'\) as d", "from series", sql)
    sql = re.sub(r"extract\(epoch from \(([\w.]+) - ([\w.]+)\)\)", r"((julianday(\1) - julianday(\2)) * 86400.0)", sql)
    sql = re.sub(r"extract\(year from (\w+)\)", r"cast(strftime('%Y', \1) as integer)", sql)
    sql = re.sub(r"extract\(month from (\w+)\)", r"cast(strftime('%m', \1) as integer)", sql)
    sql = re.sub(r"extract\(isodow from (\w+)\)", r"((cast(strftime('%w', \1) as integer) + 6) % 7 + 1)", sql)
    sql = re.sub(r"date_trunc\('month', (\w+)\)", r"date(\1, 'start of month')", sql)
    sql = re.sub(r"to_char\((\w+), 'YYYY-MM'\)", r"strftime('%Y-%m', \1)", sql)
    while "::" in sql:
        pos = sql.index("::")
        start = _operand_start(sql, pos)
        m = re.match(r"(\w+)(\(\d+(?:,\d+)?\))?", sql[pos + 2:])
        typ, end = m.group(1).lower(), pos + 2 + m.end()
        operand = sql[start:pos]
        repl = {"date": f"date({operand})", "int": f"cast({operand} as integer)", "integer": f"cast({operand} as integer)",
                "numeric": f"cast({operand} as real)", "timestamp": operand}[typ]
        sql = sql[:start] + repl + sql[end:]
    return sql


class DbtSqlite:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.con = sqlite3.connect(":memory:")
        for name, df in frames.items():
            df.to_sql(f"raw_{name}", self.con, index=False)
        self.env = jinja2.Environment()
        self.models = {p.stem: p for p in (DBT_DIR / "models").rglob("*.sql")}
        self.deps = {n: set(re.findall(r"ref\('(\w+)'\)", p.read_text())) for n, p in self.models.items()}

    def order(self) -> list[str]:
        done, order, visiting = set(), [], set()
        def visit(n):
            if n in done: return
            if n in visiting: raise RuntimeError(f"dependency cycle through {n}")
            visiting.add(n)
            for d in sorted(self.deps[n]):
                if d not in self.models: raise KeyError(f"{n} refs unknown model {d}")
                visit(d)
            visiting.discard(n); done.add(n); order.append(n)
        for n in sorted(self.models): visit(n)
        return order

    def render(self, name: str) -> str:
        tpl = self.env.from_string(self.models[name].read_text())
        return tpl.render(ref=lambda m: m, source=lambda s, t: f"{s}_{t}")

    def run(self) -> None:
        for name in self.order():
            self.con.execute(f"create table {name} as {transpile(self.render(name))}")

    def df(self, sql_or_table: str) -> pd.DataFrame:
        sql = sql_or_table if " " in sql_or_table else f"select * from {sql_or_table}"
        return pd.read_sql_query(sql, self.con)

    # ---- dbt generic tests from schema.yml -------------------------------------------------
    def run_schema_tests(self) -> list[tuple[str, str, str, int]]:
        """Returns [(model, column, test, failing_rows)]."""
        results = []
        spec = yaml.safe_load((DBT_DIR / "models" / "marts" / "schema.yml").read_text())
        for model in spec["models"]:
            m = model["name"]
            cols_in_table = {r[1] for r in self.con.execute(f"pragma table_info({m})")}
            for col in model.get("columns", []):
                c = col["name"]
                assert c in cols_in_table, f"schema.yml documents {m}.{c} but the model has no such column"
                for t in col.get("tests", []):
                    name, cfg = (t, {}) if isinstance(t, str) else next(iter(t.items()))
                    if name == "unique":
                        q = f"select count(*) from (select {c} from {m} where {c} is not null group by {c} having count(*) > 1)"
                    elif name == "not_null":
                        q = f"select count(*) from {m} where {c} is null"
                    elif name == "accepted_values":
                        vals = ",".join(f"'{v}'" for v in cfg["values"])
                        q = f"select count(*) from {m} where {c} is not null and {c} not in ({vals})"
                    elif name == "relationships":
                        pm = re.search(r"ref\('(\w+)'\)", cfg["to"]).group(1)
                        q = f"select count(*) from {m} where {c} is not null and {c} not in (select {cfg['field']} from {pm})"
                    else:
                        raise NotImplementedError(name)
                    results.append((m, c, name, self.con.execute(q).fetchone()[0]))
        return results


def clean_frames_from_generated(tmp_path: Path, n_customers=400, n_orders=700):
    from data_quality.validation import validate
    from ingestion.extract import extract
    from ingestion.generate_sample_data import generate
    generate(tmp_path, n_customers=n_customers, n_orders=n_orders)
    frames = extract(tmp_path)
    translation = frames.pop("category_translation")
    clean, report, rejected = validate(frames)
    return {**clean, "category_translation": translation}, report
