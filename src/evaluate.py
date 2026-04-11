# src/evaluate.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import numpy as np

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
    plt.savefig('reports/figures/roc_comparison.png', dpi=150)
    plt.show()

def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """Vẽ biểu đồ độ quan trọng của đặc trưng"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} đặc trưng quan trọng - {model_name}')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'reports/figures/feature_importance_{model_name}.png', dpi=150)
        plt.show()