# rebuild_models.py
import os
import joblib
from src.data_loader import load_data
from src.preprocessing import preprocess_pipeline, split_and_save, apply_smote
from src.train_model import train_xgboost, save_model
from src.config import MODEL_DIR

def main():
    print("=== REBUILD MODELS ===")
    
    # 1. Load dữ liệu gốc
    df = load_data()
    
    # 2. Tiền xử lý (tạo scaler mới)
    X, y, le_dict, scaler = preprocess_pipeline(df, fit_scaler=True)
    print(f"Shape của X sau tiền xử lý: {X.shape}")
    
    # 3. Chia train/test và lưu CSV (có thể ghi đè)
    X_train, X_test, y_train, y_test = split_and_save(X, y)
    
    # 4. SMOTE
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    
    # 5. Lưu scaler và encoders (ghi đè)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(le_dict, f"{MODEL_DIR}/label_encoders.pkl")
    print("Đã lưu scaler và encoders mới")
    
    # 6. Huấn luyện XGBoost mới
    xgb = train_xgboost(X_train_res, y_train_res)
    save_model(xgb, "xgboost.pkl")
    print("Đã lưu XGBoost mới")
    
    # 7. Kiểm tra số features
    scaler_check = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    xgb_check = joblib.load(f"{MODEL_DIR}/xgboost.pkl")
    print(f"Scaler features: {len(scaler_check.feature_names_in_)}")
    print(f"Model features: {len(xgb_check.get_booster().feature_names)}")
    
    print("=== HOÀN TẤT ===")

if __name__ == "__main__":
    main()