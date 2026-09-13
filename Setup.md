Tôi đang làm project môn Học máy cơ bản với đề tài:

"Ứng dụng Ensemble Learning trong dự đoán kết quả ĐẬU/RỚT của sinh viên"

Hãy xây dựng cho tôi một DEMO MACHINE LEARNING hoàn chỉnh, đơn giản, dễ hiểu và phù hợp để tôi thuyết trình trước giảng viên.

QUAN TRỌNG:

- Trước tiên hãy kiểm tra cấu trúc project hiện tại.
- Không được xóa hoặc phá vỡ các file/code hiện có nếu không cần thiết.
- Project hiện tại sử dụng Python.
- Tôi muốn project có thể chạy local, Docker và expose ra Internet bằng ngrok.
- Ưu tiên code dễ hiểu cho sinh viên mới học Machine Learning.
- Không làm hệ thống quá phức tạp.
- Code phải chạy được thực tế, không chỉ tạo file mẫu.

==================================================

1. # MỤC TIÊU PROJECT

Xây dựng một web demo:

"Student Pass/Fail Prediction using Ensemble Learning"

Người dùng nhập thông tin của một sinh viên:

- study_hours: số giờ học mỗi ngày
- attendance: tỷ lệ chuyên cần (%)
- assignment_score: điểm bài tập (0-10)
- midterm_score: điểm giữa kỳ (0-10)
- practice_score: điểm thực hành (0-10)

Sau đó hệ thống sử dụng Machine Learning để dự đoán:

- PASS / ĐẬU
  hoặc
- FAIL / RỚT

Ngoài kết quả cuối cùng, hệ thống phải hiển thị:

- xác suất ĐẬU
- xác suất RỚT
- kết quả của từng model
- kết quả Ensemble cuối cùng

Ví dụ:

Input:
study_hours = 5
attendance = 90
assignment_score = 8
midterm_score = 7
practice_score = 8

Output:

Prediction: PASS

Pass probability: 92%
Fail probability: 8%

Individual models:
Logistic Regression: PASS
Decision Tree: PASS
Random Forest: PASS
Gradient Boosting: PASS

Ensemble Voting: PASS

================================================== 2. Ý TƯỞNG MACHINE LEARNING
==================================================

Đây là project về Ensemble Learning.

Hãy sử dụng các model:

BASELINE:

1. Logistic Regression

SINGLE MODEL: 2. Decision Tree

ENSEMBLE: 3. Random Forest 4. Gradient Boosting 5. Voting Classifier

Mục đích là để tôi có thể giải thích:

- Logistic Regression là model đơn.
- Decision Tree là model đơn.
- Random Forest kết hợp nhiều Decision Tree bằng Bagging.
- Gradient Boosting xây dựng các model tuần tự để sửa lỗi model trước.
- Voting Classifier kết hợp dự đoán của nhiều model.

Không cần sử dụng Deep Learning.

================================================== 3. DATASET
==================================================

Nếu project chưa có dataset thật, hãy tạo một dataset demo bằng Python.

Tạo file:

data/students.csv

Các cột:

study_hours
attendance
assignment_score
midterm_score
practice_score
result

Trong đó:

result:
0 = FAIL
1 = PASS

Tạo khoảng 500-1000 dòng dữ liệu hợp lý.

Dữ liệu phải có một chút nhiễu để Machine Learning thực sự phải học chứ không phải chỉ dùng if/else đơn giản.

KHÔNG được xây dựng model dựa trực tiếp vào một công thức cố định kiểu:

if score > 5:
PASS

Thay vào đó:

- dataset được dùng để train model
- model học pattern từ dữ liệu
- sau đó model dự đoán dữ liệu mới.

================================================== 4. DATA PREPROCESSING
==================================================

Implement preprocessing rõ ràng.

Các bước:

1. Load dataset bằng pandas.
2. Kiểm tra missing values.
3. Kiểm tra duplicate.
4. Tách:

X = features
y = result

5. Chia:

train_test_split(
test_size=0.2,
random_state=42,
stratify=y
)

6. Sử dụng StandardScaler cho các model cần scale.

Ưu tiên sử dụng sklearn Pipeline để tránh data leakage.

Ví dụ:

Pipeline([
("scaler", StandardScaler()),
("model", LogisticRegression())
])

================================================== 5. TRAINING
==================================================

Tạo module riêng:

app/ml/

Ví dụ:

app/ml/data.py
app/ml/train.py
app/ml/predict.py
app/ml/evaluate.py

Train các model:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Voting Classifier

Voting Classifier có thể sử dụng:

Logistic Regression
Decision Tree
Random Forest

và voting="soft"

Nếu cần Pipeline thì đảm bảo các model được kết hợp đúng cách.

================================================== 6. MODEL EVALUATION
==================================================

Sau khi train, đánh giá trên test set.

Tính:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Tạo bảng kết quả:

Model | Accuracy | Precision | Recall | F1

Ví dụ:

Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
Voting Ensemble

KHÔNG được tự tạo số liệu kết quả.

Các metric phải được tính trực tiếp từ test dataset.

================================================== 7. API BACKEND
==================================================

Sử dụng FastAPI.

Tạo API:

GET /health

GET /api/models

POST /api/predict

POST /api/train

GET /api/evaluate

GET /docs

API POST /api/predict nhận JSON:

{
"study_hours": 5,
"attendance": 90,
"assignment_score": 8,
"midterm_score": 7,
"practice_score": 8
}

Response:

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
"ensemble": {
"prediction": "PASS",
"probability": 0.92
}
}

Probability phải được lấy từ model thực tế, không hard-code.

================================================== 8. FRONTEND
==================================================

Tạo một giao diện web đơn giản.

Không cần React nếu không cần thiết.

Có thể sử dụng:

- HTML
- CSS
- JavaScript

Giao diện gồm:

TITLE:
"Student Pass/Fail Prediction"

Form:

Số giờ học:
[ 5 ]

Tỷ lệ chuyên cần:
[ 90 ]

Điểm bài tập:
[ 8 ]

Điểm giữa kỳ:
[ 7 ]

Điểm thực hành:
[ 8 ]

[ DỰ ĐOÁN ]

Sau khi click:

Hiển thị:

================================
KẾT QUẢ DỰ ĐOÁN
================================

ĐẬU

Xác suất Đậu: 92%
Xác suất Rớt: 8%

---

## KẾT QUẢ CÁC MODEL

Logistic Regression PASS
Decision Tree PASS
Random Forest PASS
Gradient Boosting PASS

---

## ENSEMBLE

Voting Ensemble PASS

================================

================================================== 9. TRANG MODEL EVALUATION
==================================================

Tạo một trang:

/evaluation

Hiển thị bảng:

Model | Accuracy | Precision | Recall | F1

và biểu đồ so sánh Accuracy.

Có thể sử dụng Chart.js bằng CDN nếu cần.

Ví dụ:

Logistic Regression 82%
Decision Tree 78%
Random Forest 86%
Gradient Boosting 87%
Voting Ensemble 88%

LƯU Ý:
Không được hard-code các số trên.
Đây chỉ là ví dụ giao diện.

Các giá trị phải được lấy từ API /api/evaluate.

================================================== 10. KIẾN TRÚC PROJECT
==================================================

Tổ chức project rõ ràng.

Gợi ý:

HocMayCoBan/
│
├── app/
│ ├── main.py
│ │
│ ├── api/
│ │ ├── routes.py
│ │ └── schemas.py
│ │
│ ├── ml/
│ │ ├── data.py
│ │ ├── train.py
│ │ ├── predict.py
│ │ └── evaluate.py
│ │
│ ├── models/
│ │ └── trained models
│ │
│ └── frontend/
│ ├── index.html
│ ├── style.css
│ └── script.js
│
├── data/
│ └── students.csv
│
├── tests/
│ ├── test_api.py
│ └── test_model.py
│
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── README.md
└── .gitignore

================================================== 11. DOCKER
==================================================

Tạo Dockerfile để chạy FastAPI.

Yêu cầu:

- Python 3.11 hoặc Python 3.12
- cài requirements.txt
- chạy uvicorn
- bind host 0.0.0.0
- port 8000

Ví dụ command:

uvicorn app.main:app --host 0.0.0.0 --port 8000

Tạo docker-compose.yml.

Service:

app

Port:

8000:8000

Có volume nếu cần để giữ dataset/model.

Docker phải build và chạy được bằng:

docker compose up --build

================================================== 12. NGROK
==================================================

Hướng dẫn đầy đủ để expose Docker container ra Internet bằng ngrok.

Sau khi:

docker compose up --build

có thể truy cập:

http://localhost:8000

Sau đó chạy:

ngrok http 8000

Giải thích rõ:

Internet
↓
ngrok
↓
localhost:8000
↓
Docker container
↓
FastAPI
↓
Machine Learning Model

Không hard-code ngrok URL vào source code.

================================================== 13. HEALTH CHECK
==================================================

GET /health trả về:

{
"status": "ok",
"service": "student-ml-api"
}

Có thể kiểm tra:

http://localhost:8000/health

================================================== 14. TEST
==================================================

Tạo test bằng pytest.

Test:

1. /health
2. /api/predict
3. input hợp lệ
4. input không hợp lệ
5. prediction trả về PASS hoặc FAIL
6. probability nằm trong khoảng 0-1

Ví dụ:

study_hours phải >= 0

attendance phải từ 0 đến 100

scores phải từ 0 đến 10.

================================================== 15. VALIDATION
==================================================

Dùng Pydantic.

Schema:

study_hours:
float >= 0

attendance:
float >= 0 và <= 100

assignment_score:
float >= 0 và <= 10

midterm_score:
float >= 0 và <= 10

practice_score:
float >= 0 và <= 10

================================================== 16. REQUIREMENTS
==================================================

requirements.txt cần có các thư viện cần thiết, ví dụ:

fastapi
uvicorn
pandas
numpy
scikit-learn
joblib
pydantic

Không thêm thư viện không cần thiết.

================================================== 17. MODEL PERSISTENCE
==================================================

Sau khi train model, sử dụng joblib để lưu model.

Ví dụ:

models/logistic_regression.joblib
models/decision_tree.joblib
models/random_forest.joblib
models/gradient_boosting.joblib
models/voting_ensemble.joblib

Khi API khởi động:

- load model nếu model đã tồn tại
- nếu chưa tồn tại thì train model

Không train lại model ở mỗi request /api/predict.

================================================== 18. README
==================================================

Viết README.md đầy đủ và dễ hiểu.

README phải giải thích:

1. Project là gì?
2. Ensemble Learning là gì?
3. Input là gì?
4. Output là gì?
5. Dataset gồm những gì?
6. Các model được sử dụng.
7. Random Forest hoạt động như thế nào?
8. Gradient Boosting hoạt động như thế nào?
9. Voting Ensemble hoạt động như thế nào?
10. Cách chạy local.
11. Cách chạy Docker.
12. Cách chạy ngrok.
13. API endpoints.
14. Ví dụ request/response.
15. Cách chạy test.

Đặc biệt thêm sơ đồ:

User
↓
Frontend
↓
FastAPI
↓
Preprocessing
↓
Multiple ML Models
↓
Voting Ensemble
↓
Prediction
↓
PASS / FAIL

================================================== 19. YÊU CẦU CODE
==================================================

Code phải:

- sạch
- dễ đọc
- có comment ở những phần Machine Learning quan trọng
- không viết code quá phức tạp
- không hard-code prediction
- không hard-code accuracy
- không hard-code probability
- xử lý lỗi API
- validate input
- có logging cơ bản

Ưu tiên code phù hợp với sinh viên học Machine Learning cơ bản.

================================================== 20. SAU KHI HOÀN THÀNH
==================================================

Sau khi tạo code, hãy:

1. Kiểm tra toàn bộ project.
2. Kiểm tra import.
3. Kiểm tra requirements.
4. Chạy training.
5. Chạy pytest.
6. Kiểm tra API.
7. Kiểm tra Dockerfile.
8. Kiểm tra docker-compose.yml.
9. Nếu có lỗi thì tự sửa.
10. Không chỉ đưa code mà phải hướng dẫn tôi chạy từng bước trên Windows PowerShell.

Cuối cùng hãy trả cho tôi:

A. Cấu trúc project hoàn chỉnh.

B. Danh sách file đã tạo/sửa.

C. Các lệnh chạy local.

D. Các lệnh Docker.

E. Các lệnh ngrok.

F. URL API sau khi chạy.

G. Một ví dụ JSON input.

H. Một ví dụ JSON output.

I. Giải thích ngắn gọn luồng:
Input → Models → Ensemble → Output.

J. Giải thích để tôi có thể thuyết trình với giảng viên:

- Ensemble Learning là gì?
- Input là gì?
- Output là gì?
- Tại sao dùng nhiều model?
- Random Forest khác Decision Tree thế nào?
- Boosting khác Bagging thế nào?
- Voting Ensemble hoạt động thế nào?
