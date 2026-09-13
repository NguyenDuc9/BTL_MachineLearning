Bài tập lớn Học Máy Cơ Bản

# Student Pass/Fail Prediction API

Backend API demo cho đề tài **Ứng dụng Ensemble Learning trong dự đoán kết quả đậu/rớt của sinh viên**. API nhận 5 đặc trưng học tập, chạy 4 model riêng và Voting Ensemble để dự đoán `PASS` hoặc `FAIL`.

## Luồng hệ thống

```text
Client -> FastAPI -> EDA -> Preprocessing -> Multiple ML Models
	-> Voting Ensemble -> Prediction -> PASS / FAIL
```

## Mô hình và dữ liệu

Dataset demo gồm 800 dòng trong `data/students.csv`: `study_hours`, `attendance`, `assignment_score`, `midterm_score`, `practice_score`, `result` (`0=FAIL`, `1=PASS`). Dữ liệu có nhiễu ngẫu nhiên và được tạo bởi `scripts/generate_dataset.py`, không dùng công thức `if/else` trong API.

Trước khi train, bước EDA kiểm tra số dòng/cột, missing values, duplicate, phân bố `result`, thống kê mô tả và ma trận tương quan. Báo cáo được lưu tại `reports/eda_report.json` và có thể xem qua `GET /api/eda`. Sau đó project sử dụng Logistic Regression, Decision Tree, Random Forest, Gradient Boosting và Voting Classifier (`voting="soft"`). Logistic Regression dùng `StandardScaler` trong Pipeline; tập dữ liệu được chia train/test theo tỷ lệ 80/20, `random_state=42`, `stratify=y`. Accuracy, Precision, Recall, F1 và confusion matrix đều tính từ test set.

Random Forest là Bagging: huấn luyện nhiều cây độc lập trên các mẫu khác nhau rồi bỏ phiếu. Gradient Boosting xây dựng cây tuần tự, cây sau tập trung sửa lỗi cây trước. Voting Ensemble kết hợp xác suất của Logistic Regression, Decision Tree và Random Forest để chọn lớp cuối.

## Cấu hình môi trường

Tạo file `.env` từ `.env.example` và điều chỉnh khi cần. File `.env` không được commit.

```env
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
CORS_ORIGINS=*
```

## Chạy local trên Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/generate_dataset.py
uvicorn app.main:app --reload --host $env:APP_HOST --port $env:APP_PORT
```

Tài liệu API ở `http://localhost:8000/docs`, health check ở `http://localhost:8000/health`.

## Docker và deploy

```powershell
docker compose up --build -d
```

Hoặc chạy trực tiếp image:

```powershell
docker build -t student-ml-api .
docker run --env-file .env -p 8000:8000 student-ml-api
```

API sẵn sàng tại `http://localhost:8000`. Có thể dùng `ngrok http 8000` để mở API ra Internet; không cần hard-code URL ngrok.

## API

- `GET /health`
- `GET /api/models`
- `POST /api/predict`
- `POST /api/train`
- `GET /api/eda`
- `GET /api/evaluate`
- `GET /docs`

Ví dụ request:

```json
{
  "study_hours": 5,
  "attendance": 90,
  "assignment_score": 8,
  "midterm_score": 7,
  "practice_score": 8
}
```

Response có dạng:

```json
{
  "prediction": "PASS",
  "pass_probability": 0.92,
  "fail_probability": 0.08,
  "models": {
    "logistic_regression": "PASS",
    "decision_tree": "PASS",
    "random_forest": "PASS",
    "gradient_boosting": "PASS"
  },
  "ensemble": { "prediction": "PASS", "probability": 0.92 }
}
```

Các xác suất trong response là kết quả thực tế của model; ví dụ trên chỉ minh họa cấu trúc.

## Test API

Chạy test tự động:

```powershell
python -m pip install -r requirements-dev.txt
pytest -q
```

Test thủ công bằng PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/api/models
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/predict -ContentType 'application/json' -Body '{"study_hours":5,"attendance":90,"assignment_score":8,"midterm_score":7,"practice_score":8}'
Invoke-RestMethod http://localhost:8000/api/eda
Invoke-RestMethod http://localhost:8000/api/evaluate
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/train
```

Hoặc mở tài liệu tương tác tại `http://localhost:8000/docs`.

## Cấu trúc

Các module chính: `app/ml/data.py` đọc/kiểm tra dữ liệu, `app/ml/eda.py` phân tích và lưu báo cáo EDA, `train.py` train và lưu joblib, `predict.py` dự đoán, `evaluate.py` tính metric, `app/api/routes.py` khai báo API. Model được lưu trong `models/` và tự train khi chưa tồn tại; API không train lại ở mỗi request dự đoán.

Khi thuyết trình, có thể nhấn mạnh: nhiều model giúp so sánh bias/variance và tận dụng điểm mạnh khác nhau; Decision Tree là một cây, Random Forest là nhiều cây độc lập, Bagging huấn luyện song song còn Boosting huấn luyện tuần tự; Voting Ensemble tổng hợp các dự đoán để có quyết định ổn định hơn.
