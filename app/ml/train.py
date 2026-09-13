import logging
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.ml.data import FEATURES, load_raw_dataset, split_dataset
from app.ml.eda import run_eda

LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models"
MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "voting_ensemble",
]


def build_models() -> dict:
    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    random_forest = RandomForestClassifier(
        n_estimators=150, max_depth=7, random_state=42, n_jobs=-1
    )
    gradient_boosting = GradientBoostingClassifier(random_state=42)
    voting = VotingClassifier(
        estimators=[
            ("logistic_regression", logistic),
            ("decision_tree", decision_tree),
            ("random_forest", random_forest),
        ],
        voting="soft",
    )
    return {
        "logistic_regression": logistic,
        "decision_tree": decision_tree,
        "random_forest": random_forest,
        "gradient_boosting": gradient_boosting,
        "voting_ensemble": voting,
    }


def train_models() -> dict:
    run_eda(load_raw_dataset())
    X_train, _, y_train, _ = split_dataset()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models = build_models()
    for name, model in models.items():
        model.fit(X_train[FEATURES], y_train)
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        LOGGER.info("Saved model: %s", name)
    return models


def model_files_exist() -> bool:
    return all((MODEL_DIR / f"{name}.joblib").exists() for name in MODEL_NAMES)


def load_models() -> dict:
    if not model_files_exist():
        train_models()
    return {
        name: joblib.load(MODEL_DIR / f"{name}.joblib")
        for name in MODEL_NAMES
    }
