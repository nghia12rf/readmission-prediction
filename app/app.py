import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Dự đoán tái nhập viện", layout="wide")
st.title("🏥 Dự đoán nguy cơ tái nhập viện 30 ngày")
st.markdown("### Ứng dụng dành cho bệnh nhân đái tháo đường")

# Load models
@st.cache_resource
def load_models():
    model = joblib.load("models/xgboost.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/label_encoders.pkl")
    return model, scaler, encoders

model, scaler, encoders = load_models()

# Lấy danh sách cột từ scaler (chính xác)
all_features = list(scaler.feature_names_in_)
st.sidebar.write(f"DEBUG: Số cột = {len(all_features)}")  # có thể xóa sau

# Sidebar form (chỉ hiển thị một số trường chính)
st.sidebar.header("📝 Thông tin bệnh nhân")
age = st.sidebar.selectbox("Tuổi", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                                     '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])
gender = st.sidebar.selectbox("Giới tính", ['Female', 'Male'])
time_in_hospital = st.sidebar.slider("Số ngày nằm viện", 1, 14, 4)
num_lab_procedures = st.sidebar.slider("Số xét nghiệm", 0, 100, 50)
num_medications = st.sidebar.slider("Số loại thuốc", 1, 50, 10)
number_inpatient = st.sidebar.slider("Số lần nhập viện trước (nội trú)", 0, 20, 1)
change = st.sidebar.selectbox("Thay đổi thuốc", ['No', 'Ch'])
diabetesMed = st.sidebar.selectbox("Thuốc đái tháo đường", ['No', 'Yes'])
insulin = st.sidebar.selectbox("Insulin", ['No', 'Steady', 'Up', 'Down'])

# Tạo DataFrame với đầy đủ cột, gán giá trị mặc định
default_row = {col: 0 for col in all_features}  # mặc định là 0

# Ghi đè các giá trị người dùng nhập (cần map đúng tên cột)
default_row['age'] = age
default_row['gender'] = gender
default_row['time_in_hospital'] = time_in_hospital
default_row['num_lab_procedures'] = num_lab_procedures
default_row['num_medications'] = num_medications
default_row['number_inpatient'] = number_inpatient
default_row['change'] = change
default_row['diabetesMed'] = diabetesMed
default_row['insulin'] = insulin

# Các cột còn lại giữ nguyên 0 hoặc cần giá trị mặc định khác (vd: 'No')
# Đối với cột phân loại, encoder sẽ xử lý; nhưng nếu giá trị 0 không nằm trong classes thì sẽ lỗi.
# Vì vậy, ta cần gán giá trị phổ biến (mode) cho các cột phân loại không được nhập.
# Cách đơn giản: lấy giá trị đầu tiên từ encoder.classes_ (nếu có)
for col in all_features:
    if col in encoders and col not in default_row:
        # Lấy giá trị phổ biến nhất (class đầu tiên)
        default_row[col] = encoders[col].classes_[0]

input_data = pd.DataFrame([default_row])

# Mã hóa các cột phân loại
for col in all_features:
    if col in encoders:
        le = encoders[col]
        val = input_data[col].iloc[0]
        if val in le.classes_:
            input_data[col] = le.transform([val])
        else:
            # Nếu không có, gán giá trị phổ biến nhất (class đầu tiên)
            input_data[col] = le.transform([le.classes_[0]])

# Chuyển tất cả về số (phòng trường hợp còn object)
for col in all_features:
    input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)

# Chuẩn hóa
input_data_scaled = scaler.transform(input_data)
input_data = pd.DataFrame(input_data_scaled, columns=all_features)

# Dự đoán
if st.sidebar.button("🔍 Dự đoán nguy cơ"):
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0][1]
    
    st.subheader("📊 Kết quả dự đoán")
    if prediction[0] == 1:
        st.error(f"⚠️ **Có nguy cơ tái nhập viện trong vòng 30 ngày** (Xác suất: {proba:.2%})")
        st.markdown("> **Khuyến nghị:** Cần theo dõi chặt chẽ, lập kế hoạch chăm sóc sau xuất viện.")
    else:
        st.success(f"✅ **Không có nguy cơ tái nhập viện** (Xác suất: {proba:.2%})")
        st.markdown("> **Khuyến nghị:** Tiếp tục duy trì phác đồ điều trị hiện tại.")
    
    with st.expander("ℹ️ Thông tin mô hình"):
        st.write("**Mô hình sử dụng:** XGBoost")
        st.write("**Độ chính xác:** 70%")
        st.write("**Recall (tái nhập):** 49%")
        st.write("**AUC-ROC:** 0.6475")