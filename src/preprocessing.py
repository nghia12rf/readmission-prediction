# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from src.config import RANDOM_STATE, TEST_SIZE, TARGET_COLUMN

def clean_data(df):
    """Làm sạch dữ liệu: thay ? bằng NaN, loại bỏ hàng không hợp lệ"""
    df = df.replace('?', np.nan)
    
    # Loại bỏ các bệnh nhân tử vong hoặc xuất viện trái ý muốn
    invalid_discharge = [11, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    df = df[~df['discharge_disposition_id'].isin(invalid_discharge)]
    
    # Loại bỏ cột weight (quá nhiều missing)
    df = df.drop(columns=['weight'], errors='ignore')
    
    # Chỉ giữ lại một lượt nhập viện đầu tiên cho mỗi bệnh nhân
    df = df.sort_values('encounter_id').groupby('patient_nbr').first().reset_index()
    
    return df

def create_target(df):
    """Tạo nhãn nhị phân: 1 nếu readmitted == '<30', ngược lại 0"""
    df['target'] = df[TARGET_COLUMN].apply(lambda x: 1 if x == '<30' else 0)
    return df

def encode_categorical(df):
    """Mã hóa tất cả các biến phân loại (object) thành số"""
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Loại trừ cột target (đã tạo) và cột readmitted gốc
    if 'target' in cat_cols:
        cat_cols.remove('target')
    if TARGET_COLUMN in cat_cols:
        cat_cols.remove(TARGET_COLUMN)
    
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Xử lý NaN: thay bằng 'Unknown'
        df[col] = df[col].fillna('Unknown')
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    return df, le_dict

def scale_numerical(df, scaler=None):
    """Chuẩn hóa các biến số (int, float) nhưng không bao gồm cột target"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Loại trừ cột target
    if 'target' in num_cols:
        num_cols.remove('target')
    if scaler is None:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    return df, scaler

def select_features(df):
    """
    Chọn đặc trưng: chỉ giữ lại 33 cột đã được xác định.
    Đây là cách an toàn nhất để đảm bảo số lượng cột đúng.
    """
    keep_cols = [
        'race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
        'admission_source_id', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
        'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult',
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
        'insulin', 'change', 'diabetesMed'
    ]
    # Lọc các cột có trong df
    keep_cols = [c for c in keep_cols if c in df.columns]
    X = df[keep_cols]
    y = df['target']
    print(f"[select_features] Đã giữ lại {X.shape[1]} cột")
    return X, y

def preprocess_pipeline(df, fit_scaler=True):
    """Chạy toàn bộ pipeline tiền xử lý, trả về X, y, le_dict, scaler"""
    df = clean_data(df)
    df = create_target(df)
    df, le_dict = encode_categorical(df)
    
    # Tách X, y trước khi scale (chỉ giữ 33 cột)
    X, y = select_features(df)
    
    if fit_scaler:
        X, scaler = scale_numerical(X, scaler=None)
    else:
        scaler = None
    
    return X, y, le_dict, scaler

def split_and_save(X, y):
    """Chia train/test và lưu lại CSV"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    """Áp dụng SMOTE để cân bằng lớp"""
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE: {X_train.shape} -> {X_train_res.shape}")
    return X_train_res, y_train_res