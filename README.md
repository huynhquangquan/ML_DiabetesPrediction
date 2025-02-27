# Mô Hình Dự Đoán Bệnh Tiểu Đường
Dự án này nhằm mục đích dự đoán nguy cơ mắc bệnh tiểu đường bằng cách sử dụng học máy, với trọng tâm là xác định chính xác những người có khả năng mắc bệnh. Mô hình sử dụng là RandomForest, Logistic Regression, CatBoost và mô hình tổ hợp, được triển khai với FastAPI để đưa ra dự đoán theo dữ liệu thực.

# Mục Lục
* Tổng Quan Dự Án
* Dữ Liệu
* Cấu Trúc của Dự Án
* Quá Trình Xây Dựng Mô Hình
* Tiền Xử Lý
* Lựa Chọn và Đánh Giá Mô Hình
* Cài Đặt và Sử Dụng
* Cải Thiện Trong Tương Lai

# Tổng Quan Dự Án
Dự án này được thiết kế để phát hiện nguy cơ mắc bệnh tiểu đường thông qua việc phân tích dữ liệu y tế. Sử dụng các phương pháp xử lý dữ liệu, xây dựng mô hình máy học, so sánh kết quả đánh giá, kết quả dự đoán của mô hình lên các dữ liệu thực và chọn ra mô hình có kết quả tốt nhất, FastAPI được dùng để triển khai mô hình dưới dạng REST API.

# Dữ Liệu
Dữ liệu được lấy từ Kaggle và bao gồm các đặc điểm như Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome. Trong đó, Outcome là đặc điểm phân loại, quyết định liệu bệnh nhân có mắc bệnh tiểu đường.

### Bạn có thể tải về dữ liệu tại đây:
https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset

# Cấu trúc của Dự Án:

```
ML_DiabetesPrediction
    │   lib_list.txt
    │   Makefile
    │   README.md
    │   requirements.txt
    │
    ├───app
    │       API.py
    │
    ├───config
    │       dataset_config.yaml
    │       model_select.yaml
    │       preprocess_config.yaml
    │
    ├───data
    │   ├───processed
    │   │       diabetes.csv
    │   │       test.csv
    │   │       train.csv
    │   │
    │   └───raw
    │           diabetes.csv
    │
    ├───models
    │       CatBoost
    │       Logistic
    │       RandomForest
    │
    ├───notebooks
    │       Diabetes_Exploring.ipynb
    │       Diabetes_Inference.ipynb
    │       Diabetes_Modeling.ipynb
    │
    ├───results
    │   ├───figures
    │   │       Báo cáo phân loại - CatBoost.png
    │   │       Báo cáo phân loại - Logistic.png
    │   │       Báo cáo phân loại - RandomForest.png
    │   │       Ma trận nhầm lẫn - CatBoost.png
    │   │       Ma trận nhầm lẫn - Logistic.png
    │   │       Ma trận nhầm lẫn - RandomForest.png
    │   │
    │   └───reports
    │           Kết quả mô hình CatBoost.csv
    │           Kết quả mô hình Logistic.csv
    │           Kết quả mô hình RandomForest.csv
    │           PredvsData_CatBoost.csv
    │           PredvsData_Logistic.csv
    │           PredvsData_RandomForest.csv
    │
    └───src
        │   features_engineering.py
        │   utilities.py
        │
        ├───dataset
        │       download_data.py
        │
        ├───evaluate
        │       evaluate.py
        │
        ├───models
        │       CatBoost.py
        │       Logistic.py
        │       RandomForest.py
        │
        ├───prediction
        │       prediction.py
        │
        ├───preprocessing
        │   │   data_preprocessing.py
        │   │
        │   ├───balance
        │   │       RUS.py
        │   │       SMOTE.py
        │   │
        │   ├───imputation
        │   │       KNN.py
        │   │       mean.py
        │   │       median.py
        │   │
        │   ├───outliers
        │   │       attribute_wise.py
        │   │       row_wise.py
        │   │
        │   └───scaler
        │           minmax.py
        │           robust.py
        │           standard.py
        │
        └───visualization
                visualization.py
```
CHÚ Ý: Những thư mục, tập tin không nằm trong cấu trúc là được tạo tự động từ việc chạy thư viện, trong đó có thư viện CatBoost nếu chạy sẽ tạo ra thư mục catboost-info, tương tự với các thư mục khác.
### Dưới đây là mô tả về cấu trúc của dự án:

**_Makefile_**: Chứa các lệnh tự động hóa cho việc cài đặt, chạy, hoặc kiểm tra dự án.

**_README.md_**: Mô tả tổng quan về dự án, cách cài đặt và sử dụng.

**_requirements.txt_**: Danh sách các thư viện bao gồm các mô đun đi kèm với thư viện để chạy dự án.

**_lib_list.txt_**: Tương tự với requirements.txt nhưng là danh sách ngắn hơn không bao gồm mô đun đi kèm mà chỉ chứa các tên thư viện quan trọng.

**_app_**: Thư mục để chứa ứng dụng\
**_-API.py_**: Tạo REST API với FastAPI để thực hiện dự đoán.\

**_config_**: Thư mục để chứa các thiết lập\
**_-dataset_config.yaml_**: Chọn tên tập dữ liệu để xử lý, đường dẫn tải tập dữ liệu từ Kaggle.\
**_-model_select.yaml_**: Chọn mô hình để chạy hoặc tổ hợp mô hình, ngưỡng dự đoán của mô hình.\
**_-preprocess_config.yaml_**: Thiết lập quy trình xử lý dữ liệu.

**_data_**: Thư mục để chứa các tập dữ liệu\
**_-external_**: Chứa dữ liệu từ nguồn bên ngoài, tên tập dữ liệu phải là external.csv.\
**_-processed_**: Dữ liệu đã qua xử lý sẵn sàng để huấn luyện, chứa train.csv, test.csv, và diabetes.csv.\
**_-raw_**: Dữ liệu gốc chưa qua xử lý.

**_models_**: Thư mục để chứa các mô hình đã lưu

**_notebooks_**: Thư mục để chứa notebook

**_results_**: Thư mục chứa kết quả mô hình\
**_-figures_**: Lưu trữ các biểu đồ và ma trận nhầm lẫn của từng mô hình.\
**_-reports_**: Báo cáo chi tiết về kết quả dự đoán, các báo cáo phân loại và so sánh giữa dự đoán với thực tế.

**_src_**: Thư mục để chứa các mã nguồn xử lý, huấn luyện và đánh giá mô hình\
**_-features_engineering.py_**: Xử lý kỹ thuật đặc trưng.\
**_-utilities.py_**: Chứa các hàm tiện ích dùng chung trong dự án.\
**_-dataset_**: download_data.py - Tải dữ liệu thô.\
**_-evaluate_**: evaluate.py - Đánh giá hiệu suất mô hình.\
**_-models_**: Tệp chứa các mã nguồn huấn luyện và lưu mô hình.\
**_-prediction_**: prediction.py - Dự đoán đầu ra của mô hình.

**_-preprocessing_**: Tiền xử lý dữ liệu\
**_--data_preprocessing.py_**: Xử lý dữ liệu đầu vào.\
**_--balance_**: Các phương pháp cân bằng dữ liệu như RUS (Random Under Sampling) và SMOTE.\
**_--imputation_**: Các phương pháp xử lý giá trị thiếu, bao gồm KNN, mean, và median.\
**_--outliers_**: Các phương pháp phát hiện và xử lý ngoại lai, attribute_wise.py và row_wise.py.\
**_--scaler_**: Các phương pháp chuẩn hóa dữ liệu, bao gồm MinMax, Robust, và Standard scaler.\
**_--visualization_**: visualization.py - Mã nguồn thực hiện vẽ hình, biểu đồ, ma trận kết quả của mô hình.

# Tiền Xử Lý
### Các bước tiền xử lý quan trọng bao gồm:

**Xử lý giá trị thiếu**: Điền giá trị thiếu, trong đó bao gồm giá trị NaN, null, giá trị 0 và giá trị âm bằng các phương pháp imputation như trung vị (median), trung bình (mean) hoặc K-NearestNeighbor (KNN).\
**Xử lý giá trị ngoại lai**: Giữ lại hoặc loại bỏ các giá trị bất ổn ở các đặc trưng Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI qua phương pháp loại bỏ theo dòng (row_wise) hoặc mối quan hệ giữa các đặc trưng (attribute_wise) để giảm độ phúc tạp và độ nhiễu dữ liệu.\
**Cân bằng dữ liệu**: Không cân bằng hoặc cân bằng qua phương pháp SMOTE hoặc RUS để xử lý sự mất cân bằng của các lớp.\
**Kỹ thuật đặc trưng**: Loại bỏ đặc trưng của một cặp quan hệ có tương quan cao hơn mốc 0.8 để tránh khiến dữ liệu quá phức tạp, gây nhiễu dữ liệu cho quá trình huấn luyện.

# Lựa Chọn và Đánh Giá Mô Hình
### Các mô hình được đánh giá dựa trên:

Hiệu suất trung bình, độ chính xác trung bình của cross-validation, độ chính xác trên tập kiểm tra.

Trọng tâm đặc biệt vào độ nhớ của lớp dương tính (bệnh tiểu đường).

# Cài Đặt và Sử Dụng
### Yêu Cầu

Python 3.8 hoặc cao hơn\
Phần mềm IDE

## Cài Đặt
**Tải ngôn ngữ Python**: https://www.python.org/downloads/

**Sử dụng một môi trường ảo**:\
**Tải Pycharm**: https://www.jetbrains.com/pycharm/download \
**Tải Anaconda**: https://www.anaconda.com/download

**Tải Chocolatey** để sử dụng lệnh make: https://chocolatey.org/install \
Sau đó sử dụng lệnh trên Command Prompt hoặc Terminal trên máy tính với tư cách Administrator:
```
choco install make
```

**Cài đặt các thư viện cần thiết trong terminal ảo**:

Có 2 file yêu cầu, requirements.txt là toàn bộ thư viện bao gồm module nhỏ được tương thích phiên bản với nhau, lib_list là danh sách từng tên của thư viện.\
Sử dụng 1 trong 2 cách tải thư viện ở dưới đây:

-Xem danh sách trong lib_list.txt và cài đặt từng thư viện với lệnh:
```
pip install (tên thư viện)
```
-Sử dụng lệnh này với requirements.txt để tải hết toàn bộ cùng một lúc:
```
pip install -r requirements.txt
```
## Sử dụng
Sử dụng lệnh make thông qua terminal ảo hoặc chạy trực tiếp file python để chạy chương trình.

Chọn cách xử lý dữ lý trong preprocess_config.yaml.\
Trước khi chạy, hãy chọn mô hình sử dụng và ngưỡng dự đoán của mô hình trong model_select.yaml để đánh giá, vẽ biểu đồ trực quan, dự đoán kết quả.\
Đảm bảo mọi thiết lập đều có đầy đủ, đúng tên định nghĩa.

### Chạy mô hình máy học
#### Tải tập dữ liệu thô
```
make download-raw-data
```

#### Tiền xử lý dữ liệu
```
make preprocess
```

#### Huấn luyện
Lệnh này sẽ huấn luyện mô hình trên đường dần src.models.(tên mô hình)
```
make train.(tên mô hình)
```

#### Đánh giá mô hình
```
make evaluate
```
#### Biểu đồ trực quan
```
make visualize
```
#### Dự đoán kết quả
```
make predict
```
### Triển Khai API
Để bắt đầu API với FastAPI, chạy:
```
make run-api
```
hoặc
```
make run-uvicorn
```
Sau khi chạy, truy cập tài liệu API tại http://127.0.0.1:8000/docs.

### Dọn dẹp
Sau lệnh sẽ có OS=(hệ điều hành), ở đây sẽ có 2 hệ điều hành, WindowOS(win), Unix-likeOS (unix)
#### Dọn dữ liệu ngoài
```
make clear-external-data OS=(hệ điều hành)
```
#### Dọn dữ liệu đã qua xử lý
```
make clear-processed-data OS=(hệ điều hành)
```
#### Dọn dữ liệu thô
```
make clear-raw-data OS=(hệ điều hành)
```
#### Dọn sạch mô hình đã huấn luyện, mọi kết quả của mô hình
```
make clear-models-and-results OS=(hệ điều hành)
```

### Lệnh chạy kết hợp
Để tải dữ liệu thô, xử lý dữ liệu
```
make create-data
```
Để huấn luyện, đánh giá độ hiệu quả, hiệu suất và trực quan hóa mô hình, dự đoán kết quả:
```
make model=(tên mô hình) total-evaluation
```
Để xử lý lại dữ liệu, sau đó huấn luyện dựa trên dữ liệu đã được xử lý lại, đánh giá, trực quan hóa và dự đoán kết quả:
```
make model=(tên mô hình) total-evaluation-plus
```
Dọn dẹp tổng hợp
```
make clear-all OS=(hệ điều hành)
```


# Cải Thiện Trong Tương Lai
**Nâng cao mô hình**: Thử nghiệm thêm các mô hình hoặc chiến lược tinh chỉnh tham số.\
**Bổ sung dữ liệu**: Khám phá và thu thập thêm nhiều thông tin, đối tượng cho tập dữ liệu.\
**Kỹ thuật đặc trưng**: Kiểm tra biến đổi đặc trưng, thêm đặc trung hoặc kết hợp cặp đặc trưng để thay thế các đặc trưng số, đặc trưng phân loại.

### <p align="center"> **Chân thành cảm ơn vì đã dành thời gian xem markdown này.** </p>
