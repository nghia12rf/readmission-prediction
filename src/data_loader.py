import pandas as pd
from src.config import RAW_DATA_PATH

def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Đã đọc {len(df)} dòng, {df.shape[1]} cột")
    return df