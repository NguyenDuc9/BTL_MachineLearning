from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from app.ml.data import FEATURES, split_dataset
from app.ml.train import load_models


def evaluate_models() -> dict:
    _, X_test, _, y_test = split_dataset()
    results = {}
    models = load_models()
    for name, model in models.items():
        predictions = model.predict(X_test[FEATURES])
        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        }
    return results
