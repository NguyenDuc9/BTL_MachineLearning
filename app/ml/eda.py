import json
from pathlib import Path

import pandas as pd

from app.ml.data import FEATURES

ROOT_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT_DIR / "reports"
REPORT_PATH = REPORT_DIR / "eda_report.json"


def run_eda(dataset: pd.DataFrame, save_report: bool = True) -> dict:
    """Create a compact, reproducible EDA report before model training."""
    numeric_columns = FEATURES + ["result"]
    numeric_data = dataset[numeric_columns]
    target_counts = dataset["result"].value_counts().sort_index()
    report = {
        "rows": int(dataset.shape[0]),
        "columns": int(dataset.shape[1]),
        "column_names": list(dataset.columns),
        "missing_values": {
            column: int(value) for column, value in dataset.isnull().sum().items()
        },
        "duplicate_rows": int(dataset.duplicated().sum()),
        "target_distribution": {
            str(label): int(count) for label, count in target_counts.items()
        },
        "statistics": json.loads(numeric_data.describe().round(4).to_json()),
    }
    if save_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report