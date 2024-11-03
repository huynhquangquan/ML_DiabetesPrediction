Mô Hình Dự Đoán Bệnh Tiểu Đường
Dự án này nhằm mục đích dự đoán nguy cơ mắc bệnh tiểu đường bằng cách sử dụng học máy, với trọng tâm là xác định chính xác những người có khả năng mắc bệnh. Mô hình sử dụng tổ hợp RandomForest, Logistic Regression, và CatBoost, được triển khai với FastAPI để đưa ra dự đoán theo thời gian thực.

Mục Lục
Tổng Quan Dự Án
Dữ Liệu
Cấu Trúc Thư Mục
Quá Trình Xây Dựng Mô Hình
Tiền Xử Lý
Lựa Chọn và Đánh Giá Mô Hình
Cài Đặt và Sử Dụng
Cải Thiện Trong Tương Lai
Đóng Góp
Giấy Phép
Tổng Quan Dự Án
Dự án này được thiết kế để phát hiện nguy cơ mắc bệnh tiểu đường thông qua việc phân tích dữ liệu y tế. Sử dụng sự kết hợp các mô hình học máy và kỹ thuật tổ hợp, dự án đạt được độ tin cậy cao trong dự đoán. FastAPI được dùng để triển khai mô hình dưới dạng API REST.

Dữ Liệu
Dữ liệu được lấy từ Kaggle và bao gồm các đặc điểm như Pregnancies, Glucose, Blood Pressure, v.v. Bạn có thể tải về dữ liệu tại đây.

Cấu Trúc Thư Mục
Dưới đây là mô tả ngắn gọn về cấu trúc thư mục:

data/: Chứa các tệp dữ liệu thô và đã qua xử lý.
models/: Lưu các mô hình đã huấn luyện và siêu dữ liệu liên quan.
notebooks/: Các notebook Jupyter để khám phá dữ liệu và thử nghiệm.
reports/: Bao gồm tài liệu dự án, báo cáo hiệu suất, và các hình ảnh trực quan.
results/: Lưu trữ các chỉ số đánh giá mô hình, đầu ra và kết quả của các thí nghiệm.
src/: Chứa mã nguồn, bao gồm huấn luyện mô hình, đánh giá, và các tiện ích.
config.yaml: Tệp cấu hình để thiết lập đường dẫn, siêu tham số, và các hằng số khác.
requirements.txt: Liệt kê các thư viện cần thiết để chạy dự án.
Makefile: Các lệnh để tự động hóa các tác vụ như cài đặt, huấn luyện, và kiểm tra.
LICENSE: Thông tin giấy phép.
README.md: Tài liệu dự án (tệp này).
Quá Trình Xây Dựng Mô Hình
Dự án này so sánh nhiều mô hình học máy, bao gồm RandomForest, Logistic Regression, và CatBoost. Ngoài ra, tổ hợp VotingClassifier được sử dụng để cải thiện độ tin cậy trong dự đoán.

Tiền Xử Lý
Các bước tiền xử lý quan trọng bao gồm:

Xử lý giá trị thiếu và không hợp lệ: Thay thế các giá trị không hợp lệ bằng 0 hoặc giá trị trung vị.
Cân bằng dữ liệu bằng SMOTE để xử lý sự mất cân bằng của các lớp.
Kỹ thuật đặc trưng: Phân loại các đặc trưng như AgeGroup, GlucoseLevel, BMICategory, và InsulinLevel.
Mã hóa đặc trưng phân loại: Sử dụng ColumnTransformer và OneHotEncoder.
Lựa Chọn và Đánh Giá Mô Hình
Các mô hình được đánh giá dựa trên:

Độ chính xác trung bình của cross-validation và độ chính xác trên tập kiểm tra.
Trọng tâm đặc biệt vào độ nhớ của lớp dương tính (bệnh tiểu đường).
Cài Đặt và Sử Dụng
Yêu Cầu
Python 3.8 hoặc cao hơn
Khuyến nghị: sử dụng môi trường ảo (ví dụ: venv hoặc conda)
Cài Đặt
Clone repository:
bash
Copy code
git clone <repo-url>
cd <repo-name>
Cài đặt các thư viện cần thiết:
bash
Copy code
pip install -r requirements.txt
Chạy Mô Hình
Để huấn luyện và kiểm tra mô hình, chạy:

bash
Copy code
make train
Triển Khai API
Để bắt đầu API với FastAPI, chạy:

bash
Copy code
uvicorn src.main:app --reload
Truy cập tài liệu API tại http://127.0.0.1:8000/docs.

Cải Thiện Trong Tương Lai
Nâng cao mô hình: Thử nghiệm thêm các mô hình hoặc chiến lược tinh chỉnh.
Bổ sung dữ liệu: Khám phá thêm các kỹ thuật cân bằng dữ liệu.
Kỹ thuật đặc trưng: Kiểm tra các biến đổi hoặc tương tác đặc trưng mới.
Đóng Góp
Nếu muốn đóng góp, bạn có thể mở một issue hoặc tạo pull request. Đối với những thay đổi lớn, vui lòng thảo luận trước bằng cách tạo một issue.

Giấy Phép
Dự án này được cấp phép theo giấy phép MIT.
