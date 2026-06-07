
# generate_report_charts.py
import os
import sys
import io
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from src.data_loader import load_data
from src.preprocessing import clean_data, create_target
from src.evaluate import (
    plot_roc_curves,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_class_distribution_comparison,
    plot_metrics_comparison,
    plot_correlation_matrix,
    plot_eda_features
)
from src.config import RANDOM_STATE, MODEL_DIR

# Thiet lap stdout ra UTF-8 de tranh loi ma hoa tren Windows Console
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except AttributeError:
    pass

def load_processed_data():
    """Doc du lieu da duoc tien xu ly"""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test

def main():
    os.makedirs("reports/figures", exist_ok=True)
    print("=== BAT DAU VE CAC BIEU DO BAO CAO ===")
    
    # 1. Doc du lieu tho va lam sach de ve bieu do EDA
    print("\n[1/5] Dang xu ly du lieu tho phuc vu ve EDA...")
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    df_clean = create_target(df_clean)
    
    # Ve cac bieu do EDA
    plot_eda_features(df_clean)
    print("  -> Da luu cac bieu do phan tich EDA:")
    print("     - reports/figures/eda_time_in_hospital.png")
    print("     - reports/figures/eda_num_medications.png")
    print("     - reports/figures/eda_number_inpatient.png")
    print("     - reports/figures/eda_age_distribution.png")
    
    # Ve ma tran tuong quan
    plot_correlation_matrix(df_clean)
    print("     - reports/figures/correlation_matrix.png")
    
    # 2. Doc du lieu da tien xu ly
    print("\n[2/5] Doc du lieu da tien xu ly...")
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Ap dung SMOTE de ve phan bo truoc/sau SMOTE
    print("  -> Tinh toan du lieu SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    _, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Ve phan bo lop truoc/sau SMOTE
    plot_class_distribution_comparison(y_train, y_train_res)
    print("     - reports/figures/class_distribution_comparison.png")
    
    # 3. Tai cac mo hinh da huan luyen
    print("\n[3/5] Dang tai cac mo hinh da luu tu thu muc models/...")
    try:
        lr = joblib.load(f"{MODEL_DIR}/logistic_regression.pkl")
        rf = joblib.load(f"{MODEL_DIR}/random_forest.pkl")
        xgb = joblib.load(f"{MODEL_DIR}/xgboost.pkl")
    except FileNotFoundError as e:
        print(f"Loi: Khong tim thay file mo hinh. Hay chay main.py truoc. Chi tiet: {e}")
        return

    models_dict = {
        'Logistic Regression': lr,
        'Random Forest': rf,
        'XGBoost': xgb
    }
    
    # 4. Ve ma tran nham lan (Confusion Matrix) cho tung mo hinh
    print("\n[4/5] Ve ma tran nham lan...")
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, name)
        print(f"     - reports/figures/confusion_matrix_{name.replace(' ', '')}.png")
        
    # 5. So sanh hieu suat mo hinh va ve ROC Curves, Feature Importance
    print("\n[5/5] Dang ve cac bieu do so sanh hieu suat...")
    # So sanh cac chi so (Accuracy, Precision, Recall, F1, AUC)
    plot_metrics_comparison(models_dict, X_test, y_test)
    print("     - reports/figures/metrics_comparison.png")
    
    # So sanh ROC Curves
    lr_proba = lr.predict_proba(X_test)[:, 1]
    rf_proba = rf.predict_proba(X_test)[:, 1]
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    plot_roc_curves([lr_proba, rf_proba, xgb_proba], y_test, list(models_dict.keys()))
    print("     - reports/figures/roc_comparison.png")
    
    # Feature Importance cua RF va XGBoost
    feature_names = X_train.columns.tolist()
    plot_feature_importance(rf, feature_names, "RandomForest", top_n=20)
    plot_feature_importance(xgb, feature_names, "XGBoost", top_n=20)
    print("     - reports/figures/feature_importance_RandomForest.png")
    print("     - reports/figures/feature_importance_XGBoost.png")
    
    print("\n=== HOAN TAT! TAT CA BIEU DO DA DUOC LUU VAO reports/figures/ ===")

if __name__ == "__main__":
    main()
