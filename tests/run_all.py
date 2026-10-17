"""Minimal test runner for environments without pytest (supports the `tmp_path` fixture only).

    python tests/run_all.py            # run everything
    python tests/run_all.py -k dbt     # only tests whose name contains 'dbt'
    python tests/run_all.py -m ml,dbt  # only test modules whose file name contains 'ml' or 'dbt'
With pytest installed you can simply run `python -m pytest -q tests`.
"""
import importlib
import inspect
import pathlib
import sys
import tempfile
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    pattern = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else ""
    modules = sys.argv[sys.argv.index("-m") + 1].split(",") if "-m" in sys.argv else [""]
    passed, failed = 0, []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        if not any(m in path.stem for m in modules):
            continue
        mod = importlib.import_module(f"tests.{path.stem}")
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != mod.__name__ or pattern not in name:
                continue
            kwargs = {"tmp_path": pathlib.Path(tempfile.mkdtemp())} if "tmp_path" in inspect.signature(fn).parameters else {}
            t0 = time.time()
            try:
                fn(**kwargs)
                passed += 1
                print(f"PASS  {path.stem}::{name}  ({time.time() - t0:.1f}s)")
            except Exception:
                failed.append(f"{path.stem}::{name}")
                print(f"FAIL  {path.stem}::{name}\n" + "".join("      " + l for l in traceback.format_exc().splitlines(True)))
    print(f"\n{passed} passed, {len(failed)} failed")
    for f in failed:
        print("  FAILED:", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
