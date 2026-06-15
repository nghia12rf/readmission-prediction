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

def cross_validate_models(X_train, y_train):
    """Thực hiện Stratified 5-Fold Cross Validation trên tập Train"""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
    from imblearn.over_sampling import SMOTE
    from sklearn.utils.class_weight import compute_class_weight
    
    print("\n" + "="*50)
    print("BAT DAU THU HIEN STRATIFIED 5-FOLD CROSS VALIDATION")
    print("="*50)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    results = {
        'Logistic Regression': {'recall': [], 'precision': [], 'f1': [], 'auc': []},
        'Random Forest': {'recall': [], 'precision': [], 'f1': [], 'auc': []},
        'XGBoost': {'recall': [], 'precision': [], 'f1': [], 'auc': []}
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n--- Fold {fold}/5 ---")
        
        # Tách dữ liệu
        X_tr, X_val = X_train.iloc[train_idx].copy(), X_train.iloc[val_idx].copy()
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # 1. Logistic Regression: SMOTE tren tập train của fold
        print("Evaluating Logistic Regression (SMOTE)...")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
        
        lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
        lr_model.fit(X_tr_res, y_tr_res)
        y_pred_lr = lr_model.predict(X_val)
        y_proba_lr = lr_model.predict_proba(X_val)[:, 1]
        
        results['Logistic Regression']['recall'].append(recall_score(y_val, y_pred_lr))
        results['Logistic Regression']['precision'].append(precision_score(y_val, y_pred_lr))
        results['Logistic Regression']['f1'].append(f1_score(y_val, y_pred_lr))
        results['Logistic Regression']['auc'].append(roc_auc_score(y_val, y_proba_lr))
        
        # 2. Random Forest: Tính class_weight trên tập train của fold
        print("Evaluating Random Forest (Cost-Sensitive)...")
        classes = np.unique(y_tr)
        weights = compute_class_weight('balanced', classes=classes, y=y_tr)
        class_weight_dict = {classes[0]: weights[0], classes[1]: weights[1]}
        
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
            class_weight=class_weight_dict, random_state=RANDOM_STATE, n_jobs=-1
        )
        rf_model.fit(X_tr, y_tr)
        y_pred_rf = rf_model.predict(X_val)
        y_proba_rf = rf_model.predict_proba(X_val)[:, 1]
        
        results['Random Forest']['recall'].append(recall_score(y_val, y_pred_rf))
        results['Random Forest']['precision'].append(precision_score(y_val, y_pred_rf))
        results['Random Forest']['f1'].append(f1_score(y_val, y_pred_rf))
        results['Random Forest']['auc'].append(roc_auc_score(y_val, y_proba_rf))
        
        # 3. XGBoost: Tính scale_pos_weight trên tập train của fold
        print("Evaluating XGBoost (Cost-Sensitive)...")
        neg_count = (y_tr == 0).sum()
        pos_count = (y_tr == 1).sum()
        scale = neg_count / pos_count if pos_count > 0 else 1
        
        xgb_model = XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            scale_pos_weight=scale, random_state=RANDOM_STATE,
            use_label_encoder=False, eval_metric='logloss'
        )
        xgb_model.fit(X_tr, y_tr)
        y_pred_xgb = xgb_model.predict(X_val)
        y_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
        
        results['XGBoost']['recall'].append(recall_score(y_val, y_pred_xgb))
        results['XGBoost']['precision'].append(precision_score(y_val, y_pred_xgb))
        results['XGBoost']['f1'].append(f1_score(y_val, y_pred_xgb))
        results['XGBoost']['auc'].append(roc_auc_score(y_val, y_proba_xgb))
        
    print("\n" + "="*55)
    print("KET QUA STRATIFIED 5-FOLD CROSS-VALIDATION (TRUNG BINH)")
    print("="*55)
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        for metric_name, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {metric_name.upper():<10}: {mean_val:.4f} ± {std_val:.4f}")
    print("="*55 + "\n")