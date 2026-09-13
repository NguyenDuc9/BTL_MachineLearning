import logging

from fastapi import APIRouter

from app.api.schemas import StudentFeatures
from app.ml.evaluate import evaluate_models
from app.ml.data import load_raw_dataset
from app.ml.eda import run_eda
from app.ml.predict import clear_model_cache, predict_student
from app.ml.train import train_models

LOGGER = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "service": "student-ml-api"}


@router.get("/api/models")
def models():
    return {"models": ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting", "voting_ensemble"]}


@router.post("/api/predict")
def predict(features: StudentFeatures):
    LOGGER.info("Prediction requested")
    return predict_student(features.model_dump())


@router.post("/api/train")
def train():
    train_models()
    clear_model_cache()
    return {"status": "ok", "message": "Models trained and saved"}


@router.get("/api/eda")
def eda():
    return run_eda(load_raw_dataset())


@router.get("/api/evaluate")
def evaluate():
    return {"results": evaluate_models()}
