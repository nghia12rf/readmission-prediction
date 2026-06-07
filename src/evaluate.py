# src/evaluate.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd

# Thiết lập font chữ hỗ trợ tiếng Việt trên Windows
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']

def plot_roc_curves(models_proba, y_test, model_names):
    """Vẽ ROC curve so sánh các mô hình"""
    plt.figure(figsize=(8, 6))
    for name, y_proba in zip(model_names, models_proba):
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('So sánh ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/figures/roc_comparison.png', dpi=150)
    plt.close()

def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """Vẽ biểu đồ độ quan trọng của đặc trưng"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} đặc trưng quan trọng - {model_name}')
        plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.gca().invert_yaxis()
        plt.xlabel('Mức độ quan trọng')
        plt.tight_layout()
        plt.savefig(f'reports/figures/feature_importance_{model_name}.png', dpi=150)
        plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Vẽ ma trận nhầm lẫn dưới dạng heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Không tái nhập', 'Tái nhập'],
                yticklabels=['Không tái nhập', 'Tái nhập'])
    plt.ylabel('Thực tế')
    plt.xlabel('Dự đoán')
    plt.title(f'Ma trận nhầm lẫn - {model_name}')
    plt.tight_layout()
    plt.savefig(f'reports/figures/confusion_matrix_{model_name.replace(" ", "")}.png', dpi=150)
    plt.close()

def plot_class_distribution_comparison(y_before, y_after):
    """Vẽ phân bố lớp trước và sau khi áp dụng SMOTE"""
    before_counts = pd.Series(y_before).value_counts()
    after_counts = pd.Series(y_after).value_counts()
    
    df_plot = pd.DataFrame({
        'Lớp': ['Không tái nhập', 'Tái nhập'] * 2,
        'Số lượng': [before_counts.get(0, 0), before_counts.get(1, 0), 
                     after_counts.get(0, 0), after_counts.get(1, 0)],
        'Thời điểm': ['Trước SMOTE', 'Trước SMOTE', 'Sau SMOTE', 'Sau SMOTE']
    })
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Thời điểm', y='Số lượng', hue='Lớp', data=df_plot, palette='Set2')
    plt.title('Phân bố các lớp trước và sau khi áp dụng SMOTE')
    plt.ylabel('Số lượng mẫu')
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig('reports/figures/class_distribution_comparison.png', dpi=150)
    plt.close()

def plot_metrics_comparison(models_dict, X_test, y_test):
    """Vẽ biểu đồ so sánh các chỉ số đánh giá của các mô hình"""
    metrics = []
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        auc_val = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
        
        metrics.append({
            'Mô hình': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1,
            'AUC-ROC': auc_val
        })
        
    df_metrics = pd.DataFrame(metrics)
    df_melted = df_metrics.melt(id_vars='Mô hình', var_name='Chỉ số', value_name='Giá trị')
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Chỉ số', y='Giá trị', hue='Mô hình', data=df_melted, palette='viridis')
    plt.title('So sánh hiệu suất các mô hình trên tập kiểm tra')
    plt.ylim(0, 1.1)
    
    # Ghi giá trị trên đầu cột
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f'{val:.2f}', (p.get_x() + p.get_width() / 2., val),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
            
    plt.ylabel('Giá trị')
    plt.xlabel('Chỉ số đánh giá')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('reports/figures/metrics_comparison.png', dpi=150)
    plt.close()

def plot_correlation_matrix(df_clean):
    """Vẽ ma trận tương quan giữa các đặc trưng số và nhãn target"""
    num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                'num_medications', 'number_outpatient', 'number_emergency', 
                'number_inpatient', 'number_diagnoses', 'target']
    
    # Chỉ lấy các cột tồn tại trong dataframe
    num_cols = [c for c in num_cols if c in df_clean.columns]
    
    corr_matrix = df_clean[num_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Ma trận tương quan giữa các đặc trưng số và nhãn Target')
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_matrix.png', dpi=150)
    plt.close()

def plot_eda_features(df_clean):
    """Vẽ các biểu đồ phân tích khám phá dữ liệu (EDA) các đặc trưng quan trọng"""
    os.makedirs('reports/figures', exist_ok=True)
    
    # 1. Số ngày nằm viện (time_in_hospital)
    if 'time_in_hospital' in df_clean.columns and 'target' in df_clean.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='target', y='time_in_hospital', data=df_clean, palette='Set2')
        plt.xticks([0, 1], ['Không tái nhập', 'Tái nhập'])
        plt.xlabel('Trạng thái tái nhập viện')
        plt.ylabel('Số ngày nằm viện (time_in_hospital)')
        plt.title('Phân bố số ngày nằm viện theo trạng thái tái nhập viện')
        plt.tight_layout()
        plt.savefig('reports/figures/eda_time_in_hospital.png', dpi=150)
        plt.close()
        
    # 2. Số lượng thuốc (num_medications)
    if 'num_medications' in df_clean.columns and 'target' in df_clean.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='target', y='num_medications', data=df_clean, palette='Set2')
        plt.xticks([0, 1], ['Không tái nhập', 'Tái nhập'])
        plt.xlabel('Trạng thái tái nhập viện')
        plt.ylabel('Số lượng thuốc được kê (num_medications)')
        plt.title('Phân bố số lượng thuốc được kê theo trạng thái tái nhập viện')
        plt.tight_layout()
        plt.savefig('reports/figures/eda_num_medications.png', dpi=150)
        plt.close()
        
    # 3. Số lần nhập viện nội trú trước đó (number_inpatient)
    if 'number_inpatient' in df_clean.columns and 'target' in df_clean.columns:
        plt.figure(figsize=(8, 5))
        mean_inpatient = df_clean.groupby('target')['number_inpatient'].mean().reset_index()
        sns.barplot(x='target', y='number_inpatient', data=mean_inpatient, palette='Set2')
        plt.xticks([0, 1], ['Không tái nhập', 'Tái nhập'])
        plt.xlabel('Trạng thái tái nhập viện')
        plt.ylabel('Số lần nhập viện nội trú trung bình trước đó')
        plt.title('Số lần nhập viện nội trú trung bình theo trạng thái tái nhập viện')
        plt.tight_layout()
        plt.savefig('reports/figures/eda_number_inpatient.png', dpi=150)
        plt.close()
        
    # 4. Phân bố nhóm tuổi (age)
    if 'age' in df_clean.columns and 'target' in df_clean.columns:
        plt.figure(figsize=(12, 6))
        age_order = sorted(df_clean['age'].unique())
        sns.countplot(x='age', hue='target', data=df_clean, order=age_order, palette='Set2')
        plt.xlabel('Nhóm tuổi')
        plt.ylabel('Số lượng bệnh nhân')
        plt.title('Phân bố nhóm tuổi theo trạng thái tái nhập viện')
        plt.legend(['Không tái nhập', 'Tái nhập'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('reports/figures/eda_age_distribution.png', dpi=150)
        plt.close()