# preprocess_only.py
import os
import joblib
from src.data_loader import load_data
from src.preprocessing import preprocess_pipeline, split_and_save, apply_smote
from src.config import MODEL_DIR

def main():
    os.makedirs("reports/figures", exist_ok=True)
    
    print("=== TIỀN XỬ LÝ DỮ LIỆU ===")
    df = load_data()
    X, y, le_dict, scaler = preprocess_pipeline(df, fit_scaler=True)
    X_train, X_test, y_train, y_test = split_and_save(X, y)
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    
    # Lưu scaler và label encoders
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(le_dict, f"{MODEL_DIR}/label_encoders.pkl")
    print("Đã lưu scaler và label encoders vào thư mục models/")
    
    # (Tuỳ chọn) Lưu lại tập train đã được SMOTE nếu muốn dùng sau
    # joblib.dump((X_train_res, y_train_res), f"{MODEL_DIR}/train_resampled.pkl")
    
    print("=== TIỀN XỬ LÝ HOÀN TẤT ===")

if __name__ == "__main__":
    main()