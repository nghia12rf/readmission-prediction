from src.data_loader import load_data
from src.preprocessing import preprocess_pipeline, split_and_save, apply_smote
import joblib
from src.config import MODEL_DIR

df = load_data()
X, y, le_dict, scaler = preprocess_pipeline(df, fit_scaler=True)
X_train, X_test, y_train, y_test = split_and_save(X, y)
X_train_res, y_train_res = apply_smote(X_train, y_train)

joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(le_dict, f"{MODEL_DIR}/label_encoders.pkl")
print("Tiền xử lý hoàn tất!")