from app.ml.data import FEATURES, load_dataset, load_raw_dataset
from app.ml.eda import run_eda
from app.ml.predict import predict_student


def test_dataset_is_valid():
    dataset = load_dataset()
    assert len(dataset) >= 500
    assert set(FEATURES + ["result"]).issubset(dataset.columns)
    assert dataset.isnull().sum().sum() == 0


def test_prediction_has_valid_probabilities():
    result = predict_student({
        "study_hours": 5,
        "attendance": 90,
        "assignment_score": 8,
        "midterm_score": 7,
        "practice_score": 8,
    })
    assert result["prediction"] in {"PASS", "FAIL"}
    assert 0 <= result["pass_probability"] <= 1
    assert 0 <= result["fail_probability"] <= 1


def test_eda_report_describes_raw_dataset(tmp_path, monkeypatch):
    import app.ml.eda as eda

    report_path = tmp_path / "eda_report.json"
    monkeypatch.setattr(eda, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(eda, "REPORT_PATH", report_path)

    report = run_eda(load_raw_dataset())

    assert report["rows"] >= 500
    assert report["duplicate_rows"] == 0
    assert set(report["target_distribution"]) <= {"0", "1"}
    assert report_path.exists()
