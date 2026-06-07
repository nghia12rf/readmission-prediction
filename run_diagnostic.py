# run_diagnostic.py
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.config import RANDOM_STATE

def load_processed_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_processed_data()
    print(f"Original train size: {X_train.shape}, test size: {X_test.shape}")
    print(f"Train label counts: {np.bincount(y_train)}")
    print(f"Test label counts: {np.bincount(y_test)}")
    
    # Apply SMOTE
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE train size: {X_train_res.shape}")
    print(f"SMOTE train label counts: {np.bincount(y_train_res)}")
    
    # 1. Logistic Regression
    print("\n--- Training Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
    lr.fit(X_train_res, y_train_res)
    y_pred_lr = lr.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr, target_names=['No', 'Yes']))
    
    # 2. Random Forest (with class weight balanced)
    print("\n--- Training Random Forest (SMOTE + weight balanced) ---")
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train_res)
    weights = compute_class_weight('balanced', classes=classes, y=y_train_res)
    class_weight_dict = {classes[0]: weights[0], classes[1]: weights[1]}
    print(f"RF weights: {class_weight_dict}")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weight_dict,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train_res, y_train_res)
    y_pred_rf = rf.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf, target_names=['No', 'Yes']))

    # 3. XGBoost
    print("\n--- Training XGBoost (SMOTE, scale_pos_weight calculated from SMOTE) ---")
    neg_count = (y_train_res == 0).sum()
    pos_count = (y_train_res == 1).sum()
    scale = neg_count / pos_count
    print(f"XGBoost scale_pos_weight: {scale}")
    xgb = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6,
        scale_pos_weight=scale, random_state=RANDOM_STATE,
        use_label_encoder=False, eval_metric='logloss'
    )
    xgb.fit(X_train_res, y_train_res)
    y_pred_xgb = xgb.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_xgb))
    print(classification_report(y_test, y_pred_xgb, target_names=['No', 'Yes']))

    # 4. XGBoost (Original data + scale_pos_weight)
    print("\n--- Training XGBoost (Original data + scale_pos_weight) ---")
    neg_count_orig = (y_train == 0).sum()
    pos_count_orig = (y_train == 1).sum()
    scale_orig = neg_count_orig / pos_count_orig
    print(f"XGBoost original scale_pos_weight: {scale_orig}")
    xgb_orig = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6,
        scale_pos_weight=scale_orig, random_state=RANDOM_STATE,
        use_label_encoder=False, eval_metric='logloss'
    )
    xgb_orig.fit(X_train, y_train)
    y_pred_xgb_orig = xgb_orig.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_xgb_orig))
    print(classification_report(y_test, y_pred_xgb_orig, target_names=['No', 'Yes']))

if __name__ == "__main__":
    main()
