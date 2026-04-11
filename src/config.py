import os

# Đường dẫn
RAW_DATA_PATH = "data/raw/diabetic_data.csv"
PROCESSED_DIR = "data/processed/"
MODEL_DIR = "models/"

# Tạo thư mục nếu chưa có
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Tham số
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "readmitted"