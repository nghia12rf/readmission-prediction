
```markdown
# 🏥 Dự đoán nguy cơ tái nhập viện ở bệnh nhân đái tháo đường

Đồ án môn **Khai phá dữ liệu** – Xây dựng mô hình phân lớp dự đoán khả năng bệnh nhân đái tháo đường tái nhập viện trong vòng 30 ngày sau xuất viện, từ đó hỗ trợ các cơ sở y tế chủ động lập kế hoạch chăm sóc.

## 📌 Mục lục
- [Tổng quan](#-tổng-quan)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cài đặt và chạy](#-cài-đặt-và-chạy)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Kết quả](#-kết-quả)
- [Giấy phép](#-giấy-phép)

---

## 📖 Tổng quan

**Bài toán:**
Dự đoán nhị phân – bệnh nhân có tái nhập viện trong vòng 30 ngày hay không.

**Dữ liệu:**
Bộ dữ liệu công khai [Diabetes 130-US Hospitals (1999-2008)](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) từ UCI Machine Learning Repository, bao gồm hơn 100.000 lượt nhập viện với 50 đặc trưng (nhân khẩu, lâm sàng, thuốc, lịch sử khám chữa bệnh,…).

**Quy trình:**
Tiền xử lý → SMOTE cân bằng lớp → Huấn luyện các mô hình (Logistic Regression, Random Forest, XGBoost) → Đánh giá → Xây dựng ứng dụng demo.

**Kết quả chính:**
Mô hình XGBoost đạt **recall 49%** và **AUC-ROC 0.6475** trên tập kiểm tra, giúp phát hiện gần một nửa số bệnh nhân có nguy cơ tái nhập viện.

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
  - `joblib` – lưu/mô hình

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
│   ├── train_model.py            # Huấn luyện 3 mô hình
│   ├── evaluate.py               # Đánh giá, vẽ ROC, feature importance
│   └── utils.py                  # (không dùng)
│
├── models/                       # Mô hình đã huấn luyện (.pkl) và scaler
├── app/                          # Ứng dụng Streamlit
│   └── app.py
│
├── reports/figures/              # Biểu đồ (ROC, feature importance)
├── preprocess_only.py            # Chạy tiền xử lý (một lần)
├── main.py                       # Huấn luyện + đánh giá (chạy sau preprocess)
├── environment.yml               # Môi trường Conda
├── requirements.txt              # Pip dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Cài đặt và chạy

### 1. Clone repository
```bash
git clone [https://github.com/nghia12rf/readmission-prediction.git](https://github.com/nghia12rf/readmission-prediction.git)
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

### 5. Huấn luyện và đánh giá
```bash
python main.py
```
- Huấn luyện Logistic Regression, Random Forest, XGBoost (có SMOTE)
- In báo cáo phân loại (precision, recall, f1) và AUC-ROC
- Vẽ ROC curves và feature importance (lưu vào `reports/figures/`)

### 6. Chạy ứng dụng demo
```bash
streamlit run app/app.py
```
Trình duyệt sẽ mở ra giao diện, cho phép nhập thông tin bệnh nhân và nhận kết quả dự đoán.

---

## 🖥 Hướng dẫn sử dụng ứng dụng

1. Nhập các thông tin cơ bản ở sidebar:
   - Tuổi, giới tính
   - Số ngày nằm viện, số xét nghiệm, số thuốc
   - Số lần nhập viện nội trú trước đó
   - Thay đổi thuốc, thuốc đái tháo đường, insulin
2. Nhấn nút **"Dự đoán nguy cơ"**.
3. Kết quả hiển thị:
   - **Có nguy cơ** hoặc **Không có nguy cơ**
   - Xác suất tái nhập viện
   - Khuyến nghị chăm sóc tương ứng.

> **Lưu ý:** Ứng dụng chỉ mang tính chất tham khảo, không thay thế chẩn đoán lâm sàng.

---

## 📊 Kết quả

### Hiệu suất các mô hình (trên tập test)

| Mô hình             | Recall (tái nhập) | Precision (tái nhập) | F1-score | AUC-ROC |
|---------------------|-------------------|----------------------|----------|---------|
| Logistic Regression | 0.50              | 0.13                 | 0.20     | 0.6208  |
| Random Forest       | 0.16              | 0.17                 | 0.17     | 0.6248  |
| **XGBoost**         | **0.49**          | **0.15**             | **0.22** | **0.6475** |

- **XGBoost** được chọn cho ứng dụng do recall cao nhất, ưu tiên phát hiện bệnh nhân có nguy cơ.

### Biểu đồ ROC và Feature importance
Các hình ảnh được lưu trong thư mục `reports/figures/`:
- `roc_comparison.png` – so sánh ROC của 3 mô hình.
- `feature_importance_RandomForest.png` và `feature_importance_XGBoost.png` – top 20 đặc trưng quan trọng.

---

## 📄 Giấy phép

Dự án được phát triển cho mục đích học tập, sử dụng bộ dữ liệu công khai. Mọi quyền truy cập và sử dụng dữ liệu phải tuân theo giấy phép của UCI Machine Learning Repository.

---

## 👥 Tác giả

Nhóm sinh viên – **Trường Đại học Công Thương Thành phố Hồ Chí Minh**
Giảng viên hướng dẫn: ****

---

## 🙏 Cảm ơn

- UCI Machine Learning Repository vì đã cung cấp bộ dữ liệu.
- Cộng đồng mã nguồn mở Python vì các thư viện tuyệt vời.
```