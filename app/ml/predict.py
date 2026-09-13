import pandas as pd

from app.ml.data import FEATURES
from app.ml.train import load_models

_MODEL_CACHE = None


def clear_model_cache() -> None:
    global _MODEL_CACHE
    _MODEL_CACHE = None


def get_models() -> dict:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = load_models()
    return _MODEL_CACHE


def predict_student(values: dict) -> dict:
    row = pd.DataFrame([[values[feature] for feature in FEATURES]], columns=FEATURES)
    models = get_models()
    individual = {}
    for name in ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting"]:
        individual[name] = "PASS" if int(models[name].predict(row)[0]) == 1 else "FAIL"

    ensemble = models["voting_ensemble"]
    ensemble_prediction = int(ensemble.predict(row)[0])
    probabilities = ensemble.predict_proba(row)[0]
    fail_probability = float(probabilities[0])
    pass_probability = float(probabilities[1])
    return {
        "prediction": "PASS" if ensemble_prediction == 1 else "FAIL",
        "pass_probability": round(pass_probability, 4),
        "fail_probability": round(fail_probability, 4),
        "models": individual,
        "ensemble": {
            "prediction": "PASS" if ensemble_prediction == 1 else "FAIL",
            "probability": round(pass_probability if ensemble_prediction == 1 else fail_probability, 4),
        },
    }
