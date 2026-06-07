# 🏥 Dự đoán nguy cơ tái nhập viện ở bệnh nhân đái tháo đường

Đồ án môn **Khai phá dữ liệu** – Xây dựng mô hình phân lớp dự đoán khả năng bệnh nhân đái tháo đường tái nhập viện trong vòng 30 ngày sau xuất viện, từ đó hỗ trợ các cơ sở y tế chủ động lập kế hoạch chăm sóc.

## 📌 Mục lục
- [Tổng quan](#-tổng quan)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cài đặt và chạy](#-cài-đặt-và-chạy)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Kết quả](#-kết-quả)
- [Giấy phép](#-giấy-phép)

---

## 📖 Tổng quan

**Bài toán:**
Dự đoán nhị phân – bệnh nhân có tái nhập viện trong vòng 30 ngày hay không (nhãn `readmitted` = `<30` là 1, ngược lại là 0).

**Dữ liệu:**
Bộ dữ liệu công khai [Diabetes 130-US Hospitals (1999-2008)](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) từ UCI Machine Learning Repository, bao gồm hơn 100.000 lượt nhập viện với 50 đặc trưng (nhân khẩu, lâm sàng, thuốc, lịch sử khám chữa bệnh,…).

**Quy trình huấn luyện nâng cao:**
Nhằm giải quyết vấn đề mất cân bằng lớp nghiêm trọng (chỉ ~9% ca tái nhập viện), quy trình đã được tối ưu hóa theo hai hướng:
1. **Mô hình tuyến tính (Logistic Regression)**: Sử dụng phương pháp **SMOTE** để cân bằng lớp trên tập huấn luyện nhằm tìm ranh giới phân lớp tốt hơn.
2. **Mô hình cây quyết định (Random Forest & XGBoost)**: Huấn luyện trên **dữ liệu gốc (chưa SMOTE)** để tránh hiện tượng quá khớp (overfitting) các mẫu nhân tạo, kết hợp kỹ thuật **Cost-Sensitive Learning** (tham số phạt lỗi `class_weight='balanced'` cho RF và `scale_pos_weight` cho XGBoost) giúp tối đa hóa khả năng tổng quát hóa trên tập kiểm tra thực tế.

**Kết quả chính:**
Mô hình XGBoost đạt **recall 49%** và **AUC-ROC 0.6475** trên tập kiểm tra, giúp phát hiện gần một nửa số bệnh nhân có nguy cơ tái nhập viện mà vẫn giữ được tính tổng quát hóa tốt nhất.

---

## 🛠 Công nghệ sử dụng

- **Ngôn ngữ:** Python 3.9
- **Môi trường:** Miniconda / Anaconda
- **Thư viện chính:**
  - `pandas`, `numpy` – xử lý dữ liệu
  - `scikit-learn` – tiền xử lý, mô hình Logistic Regression & Random Forest, đánh giá
  - `xgboost` – mô hình XGBoost
  - `imbalanced-learn` – SMOTE
  - `matplotlib`, `seaborn` – vẽ biểu đồ
  - `streamlit` – xây dựng ứng dụng demo
  - `joblib` – lưu/tải mô hình

---

## 📁 Cấu trúc dự án

```text
readmission-prediction/
│
├── data/                         # Dữ liệu (được .gitignore, chỉ giữ cấu trúc)
│   ├── raw/                      # Đặt file diabetic_data.csv tại đây
│   └── processed/                # CSV sau tiền xử lý (tự sinh)
│
├── src/                          # Mã nguồn chính
│   ├── __init__.py
│   ├── config.py                 # Đường dẫn, tham số
│   ├── data_loader.py            # Đọc dữ liệu thô
│   ├── preprocessing.py          # Tiền xử lý, SMOTE, chia train/test
│   ├── train_model.py            # Huấn luyện các mô hình
│   └── evaluate.py               # Thư viện vẽ biểu đồ phân tích và so sánh
│
├── models/                       # Mô hình đã huấn luyện (.pkl) và scaler
├── app/                          # Ứng dụng Streamlit
│   └── app.py
│
├── reports/figures/              # Biểu đồ (được tự động sinh ra phục vụ viết báo cáo)
├── preprocess_only.py            # Chạy tiền xử lý dữ liệu (chạy một lần)
├── main.py                       # Chạy huấn luyện lại và đánh giá mô hình
├── generate_report_charts.py     # Tạo tự động 13 biểu đồ phân tích phục vụ viết báo cáo
├── environment.yml               # Môi trường Conda
├── requirements.txt              # Pip dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Cài đặt và chạy

### 1. Clone repository
```bash
git clone https://github.com/nghia12rf/readmission-prediction.git
cd readmission-prediction
```

### 2. Tạo môi trường Conda
```bash
conda env create -f environment.yml
conda activate readmission_env
```
Hoặc dùng pip:
```bash
pip install -r requirements.txt
```

### 3. Đặt dữ liệu
Tải file `diabetic_data.csv` từ [UCI](https://archive.ics.uci.edu/ml/machine-learning-databases/00296/) và đặt vào thư mục `data/raw/`.

### 4. Tiền xử lý (chạy một lần)
```bash
python preprocess_only.py
```
- Sinh các file CSV trong `data/processed/`
- Lưu `scaler.pkl` và `label_encoders.pkl` vào `models/`

### 5. Huấn luyện và đánh giá mô hình
```bash
python main.py
```
- Huấn luyện Logistic Regression (với SMOTE), Random Forest (với class weights) và XGBoost (với scale_pos_weight).
- Đánh giá hiệu suất và in ra báo cáo phân loại (precision, recall, f1) cùng AUC-ROC trên tập kiểm tra thực tế.

### 6. Tự động xuất biểu đồ phục vụ viết báo cáo
```bash
python generate_report_charts.py
```
- Tự động tạo và lưu **13 biểu đồ phân tích** lâm sàng (EDA), phân bố SMOTE, ma trận nhầm lẫn (confusion matrix) của từng mô hình, so sánh hiệu suất và đường cong ROC vào thư mục `reports/figures/`.

### 7. Chạy ứng dụng demo
```bash
streamlit run app/app.py
```
Trình duyệt sẽ mở ra giao diện, cho phép nhập thông tin bệnh nhân và nhận kết quả dự đoán.

---

## 📊 Kết quả

### Hiệu suất các mô hình (trên tập kiểm tra Test Set)

| Mô hình             | Recall (tái nhập) | Precision (tái nhập) | F1-score | AUC-ROC |
|---------------------|-------------------|----------------------|----------|---------|
| Logistic Regression | 0.53              | 0.12                 | 0.20     | 0.6128  |
| Random Forest       | 0.16              | 0.17                 | 0.17     | 0.6248  |
| **XGBoost**         | **0.49**          | **0.15**             | **0.22** | **0.6475** |

- **XGBoost** được chọn làm mô hình triển khai thực tế nhờ đạt chỉ số **AUC-ROC (0.6475)** và **F1-score (0.22)** tối ưu nhất, đồng thời giữ được **Recall đạt 49%** giúp bệnh viện chủ động khoanh vùng một nửa số bệnh nhân có nguy cơ tái nhập viện cao.

### Danh sách biểu đồ trong thư mục `reports/figures/`:
* `class_distribution_comparison.png` – Phân bố mẫu trước/sau SMOTE.
* `correlation_matrix.png` – Ma trận tương quan các đặc trưng số.
* `eda_time_in_hospital.png`, `eda_num_medications.png`, `eda_number_inpatient.png`, `eda_age_distribution.png` – Biểu đồ EDA đặc trưng quan trọng.
* `confusion_matrix_LogisticRegression.png`, `confusion_matrix_RandomForest.png`, `confusion_matrix_XGBoost.png` – Ma trận nhầm lẫn của từng mô hình.
* `metrics_comparison.png` – So sánh trực quan các chỉ số đánh giá.
* `roc_comparison.png` – So sánh đường cong ROC.
* `feature_importance_RandomForest.png`, `feature_importance_XGBoost.png` – Top 20 đặc trưng quan trọng của các mô hình cây.

---

## 📄 Giấy phép

Dự án được phát triển cho mục đích học tập, sử dụng bộ dữ liệu công khai. Mọi quyền truy cập và sử dụng dữ liệu phải tuân theo giấy phép của UCI Machine Learning Repository.

---

## 👥 Tác giả

Nhóm sinh viên – **Trường Đại học Công Thương Thành phố Hồ Chí Minh**
Giảng viên hướng dẫn: **Th.S Nguyễn Thị Huyền Trang**

```