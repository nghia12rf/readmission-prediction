# main.py
import os
import sys
import io
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from src.train_model import train_logistic_regression, train_random_forest, train_xgboost, save_model, evaluate_model, cross_validate_models
from src.evaluate import plot_roc_curves, plot_feature_importance
from src.config import RANDOM_STATE

# Thiet lap stdout ra UTF-8 de tranh loi ma hoa tren Windows Console
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except AttributeError:
    pass

def load_processed_data():
    """Đọc dữ liệu đã được tiền xử lý từ thư mục data/processed/"""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def main():
    os.makedirs("reports/figures", exist_ok=True)
    
    # === ĐỌC DỮ LIỆU ĐÃ TIỀN XỬ LÝ ===
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # === THỰC HIỆN STRATIFIED 5-FOLD CROSS VALIDATION ===
    cross_validate_models(X_train, y_train)
    
    # === ÁP DỤNG SMOTE (chỉ dùng cho Logistic Regression) ===
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE (for LR): {X_train.shape} -> {X_train_res.shape}")
    
    # === HUẤN LUYỆN MÔ HÌNH ===
    print("\n=== HUAN LUYEN MO HINH ===")
    
    # 1. Logistic Regression: huan luyen tren du lieu SMOTE
    print("\n[1/3] Huan luyen Logistic Regression tren du lieu SMOTE...")
    lr = train_logistic_regression(X_train_res, y_train_res)
    save_model(lr, "logistic_regression.pkl")
    
    # 2. Random Forest: huan luyen tren du lieu goc (cost-sensitive)
    print("\n[2/3] Huan luyen Random Forest tren du lieu goc...")
    rf = train_random_forest(X_train, y_train)
    save_model(rf, "random_forest.pkl")
    
    # 3. XGBoost: huan luyen tren du lieu goc (scale_pos_weight tu dong)
    print("\n[3/3] Huan luyen XGBoost tren du lieu goc...")
    xgb = train_xgboost(X_train, y_train)
    save_model(xgb, "xgboost.pkl")
    
    # === ĐÁNH GIÁ ===
    print("\n=== DANH GIA MO HINH ===")
    _, lr_proba = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    _, rf_proba = evaluate_model(rf, X_test, y_test, "Random Forest")
    _, xgb_proba = evaluate_model(xgb, X_test, y_test, "XGBoost")
    
    print("\n=== VE BIEU DO DANH GIA ===")
    plot_roc_curves([lr_proba, rf_proba, xgb_proba], y_test,
                    ["Logistic Regression", "Random Forest", "XGBoost"])
    
    feature_names = X_train.columns.tolist()
    plot_feature_importance(rf, feature_names, "RandomForest", top_n=20)
    plot_feature_importance(xgb, feature_names, "XGBoost", top_n=20)
    
    print("\n=== HOAN TAT ===")

if __name__ == "__main__":
    main()