"""Extract: read the Olist CSV files into DataFrames (no cleaning happens here)."""
import logging
from pathlib import Path

import pandas as pd

from ingestion.config import FILES, OPTIONAL_FILES, RAW_DIR

log = logging.getLogger(__name__)


def extract(data_dir: Path | str = RAW_DIR) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    frames: dict[str, pd.DataFrame] = {}
    for name, fname in FILES.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download the Olist dataset from Kaggle into {data_dir} "
                "or run `python -m ingestion.generate_sample_data`."
            )
        # read everything as text so that bad values reach the validation layer intact
        frames[name] = pd.read_csv(path, dtype=str, keep_default_na=True)
        log.info("extracted %s: %d rows", name, len(frames[name]))
    for name, fname in OPTIONAL_FILES.items():
        path = data_dir / fname
        if path.exists():
            frames[name] = pd.read_csv(path, dtype=str)
    return frames


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extract()
