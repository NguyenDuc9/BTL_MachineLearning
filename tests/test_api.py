from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_input():
    response = client.post("/api/predict", json={
        "study_hours": 5,
        "attendance": 90,
        "assignment_score": 8,
        "midterm_score": 7,
        "practice_score": 8,
    })
    assert response.status_code == 200
    assert response.json()["prediction"] in {"PASS", "FAIL"}


def test_predict_rejects_invalid_input():
    response = client.post("/api/predict", json={
        "study_hours": -1,
        "attendance": 90,
        "assignment_score": 8,
        "midterm_score": 7,
        "practice_score": 8,
    })
    assert response.status_code == 422
