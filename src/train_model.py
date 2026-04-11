# src/train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.config import MODEL_DIR, RANDOM_STATE

def load_processed_data():
    """Đọc dữ liệu đã tiền xử lý từ CSV"""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    """Huấn luyện Logistic Regression (baseline)"""
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Huấn luyện Random Forest với cấu hình cân bằng hơn"""
    # Tính trọng số cho lớp tái nhập
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {classes[0]: weights[0], classes[1]: weights[1]}
    
    model = RandomForestClassifier(
        n_estimators=200,           # tăng số cây
        max_depth=15,               # tăng độ sâu để học được lớp thiểu số
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weight_dict,  # hoặc dùng 'balanced'
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """Huấn luyện XGBoost với scale_pos_weight tự động"""
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale = neg_count / pos_count if pos_count > 0 else 1
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                          scale_pos_weight=scale, random_state=RANDOM_STATE,
                          use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    """Lưu mô hình vào thư mục models/"""
    joblib.dump(model, f"{MODEL_DIR}/{filename}")
    print(f"Đã lưu: {filename}")

def evaluate_model(model, X_test, y_test, model_name):
    """Đánh giá và in kết quả"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred, target_names=['Không tái nhập', 'Tái nhập']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        print(f"AUC-ROC: {auc:.4f}")
    return y_pred, y_proba