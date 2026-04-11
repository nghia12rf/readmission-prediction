import pandas as pd

# Đọc dữ liệu
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

print("=== KIỂM TRA RÒ RỈ DỮ LIỆU ===\n")
print("Các cột trong X_train:")
print(X_train.columns.tolist())
print("\n" + "="*50 + "\n")

# Kiểm tra từng cột
for col in X_train.columns:
    print(f"Cột: {col}")
    # Nếu cột là số
    if X_train[col].dtype in ['int64', 'float64']:
        # Tính tương quan với y_train
        if X_train[col].std() > 0:
            corr = X_train[col].corr(pd.Series(y_train))
            print(f"  - Tương quan Pearson: {corr:.6f}")
            if abs(corr) > 0.99:
                print(f"  *** CẢNH BÁO: Tương quan quá cao! ***")
        else:
            print("  - Cột là hằng số, bỏ qua")
    else:
        # Cột phân loại: kiểm tra bảng chéo
        cross = pd.crosstab(X_train[col], y_train)
        # Kiểm tra xem có giá trị nào chỉ xuất hiện ở một lớp không
        if (cross.iloc[:,0] == 0).any() or (cross.iloc[:,1] == 0).any():
            print("  - Có giá trị chỉ xuất hiện ở một lớp (gây rò rỉ):")
            print(cross)
        else:
            print("  - Không phát hiện rò rỉ")
    print("-" * 50)