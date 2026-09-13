from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "students.csv"
FEATURES = [
    "study_hours",
    "attendance",
    "assignment_score",
    "midterm_score",
    "practice_score",
]


def _generate_dataset() -> None:
    rng = np.random.default_rng(42)
    rows = 800
    frame = pd.DataFrame({
        "study_hours": np.round(rng.uniform(0, 10, rows), 1),
        "attendance": np.round(rng.uniform(45, 100, rows), 1),
        "assignment_score": np.round(rng.uniform(2, 10, rows), 1),
        "midterm_score": np.round(rng.uniform(2, 10, rows), 1),
        "practice_score": np.round(rng.uniform(2, 10, rows), 1),
    })
    score = (
        0.16 * frame["study_hours"] + 0.025 * frame["attendance"]
        + 0.22 * frame["assignment_score"] + 0.32 * frame["midterm_score"]
        + 0.25 * frame["practice_score"] + rng.normal(0, 0.65, rows)
    )
    frame["result"] = (score >= 6.5).astype(int)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PATH, index=False)


def load_dataset() -> pd.DataFrame:
    """Load and validate the training dataset."""
    dataset = load_raw_dataset()
    return dataset.drop_duplicates().reset_index(drop=True)


def load_raw_dataset() -> pd.DataFrame:
    """Load the dataset before duplicate removal for data-quality analysis."""
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size < 100:
        _generate_dataset()
    dataset = pd.read_csv(DATA_PATH)
    required_columns = FEATURES + ["result"]
    missing_columns = set(required_columns) - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")
    if dataset[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values")
    return dataset[required_columns]


def split_dataset():
    dataset = load_dataset()
    X = dataset[FEATURES]
    y = dataset["result"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
