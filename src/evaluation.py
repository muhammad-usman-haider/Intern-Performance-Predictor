import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report

def evaluate_model(clf, X_test, y_test, proba_supported=True):
    y_pred = clf.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, digits=4)
    }
    if proba_supported:
        try:
            y_proba = clf.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        except Exception:
            pass
    return metrics

def compare_metrics(rf_metrics, xgb_metrics):
    keys = sorted(set(rf_metrics.keys()) | set(xgb_metrics.keys()))
    rows = []
    for k in keys:
        rf_val = rf_metrics.get(k, None)
        xgb_val = xgb_metrics.get(k, None)
        rows.append((k, rf_val, xgb_val))
    return rows