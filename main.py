# main.py
import os
from src.data_loader import load_data
from src.preprocessing import preprocess_pipeline, split_and_save, apply_smote
from src.train_model import train_logistic_regression, train_random_forest, train_xgboost, save_model, evaluate_model
from src.evaluate import plot_roc_curves, plot_feature_importance
import joblib
from src.config import MODEL_DIR

def main():
    os.makedirs("reports/figures", exist_ok=True)
    
    # === TIỀN XỬ LÝ ===
    print("=== TIỀN XỬ LÝ DỮ LIỆU ===")
    df = load_data()
    X, y, le_dict, scaler = preprocess_pipeline(df, fit_scaler=True)
    X_train, X_test, y_train, y_test = split_and_save(X, y)
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    
    # Lưu scaler và label encoders
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(le_dict, f"{MODEL_DIR}/label_encoders.pkl")
    
    # === HUẤN LUYỆN MÔ HÌNH ===
    print("\n=== HUẤN LUYỆN MÔ HÌNH ===")
    lr = train_logistic_regression(X_train_res, y_train_res)
    save_model(lr, "logistic_regression.pkl")
    
    rf = train_random_forest(X_train_res, y_train_res)
    save_model(rf, "random_forest.pkl")
    
    xgb = train_xgboost(X_train_res, y_train_res)
    save_model(xgb, "xgboost.pkl")
    
    # === ĐÁNH GIÁ ===
    print("\n=== ĐÁNH GIÁ MÔ HÌNH ===")
    _, lr_proba = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    _, rf_proba = evaluate_model(rf, X_test, y_test, "Random Forest")
    _, xgb_proba = evaluate_model(xgb, X_test, y_test, "XGBoost")
    
    plot_roc_curves([lr_proba, rf_proba, xgb_proba], y_test,
                    ["Logistic Regression", "Random Forest", "XGBoost"])
    
    feature_names = X_train.columns.tolist()
    plot_feature_importance(rf, feature_names, "RandomForest", top_n=20)
    plot_feature_importance(xgb, feature_names, "XGBoost", top_n=20)
    
    print("\n=== HOÀN TẤT ===")

if __name__ == "__main__":
    main()