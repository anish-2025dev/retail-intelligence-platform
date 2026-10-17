"""Write outputs/pipeline_report.md from the audit tables (or the CSV reports in --dry-run mode)."""
import pandas as pd

from ingestion.config import OUTPUT_DIR, database_url


def main() -> None:
    from sqlalchemy import create_engine
    e = create_engine(database_url())
    run = pd.read_sql("SELECT * FROM audit.pipeline_runs ORDER BY started_at DESC LIMIT 1", e).iloc[0]
    rules = pd.read_sql("SELECT table_name, rule, severity, records_failed, action FROM audit.validation_results "
                        "WHERE run_id = %(r)s AND records_failed > 0 ORDER BY records_failed DESC", e, params={"r": run.run_id})
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    lines = [f"# Pipeline report - run {run.run_id}", "",
             f"- Status: **{run.status}**", f"- Started: {run.started_at}  |  Finished: {run.finished_at}",
             f"- Records read: {run.records_read:,}  |  inserted: {run.records_inserted:,}  |  rejected: {run.records_rejected:,}",
             "", "## Rules with failures", "", rules.to_markdown(index=False) if len(rules) else "None"]
    (OUTPUT_DIR / "pipeline_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
