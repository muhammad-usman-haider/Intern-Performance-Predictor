import os
import joblib
import pandas as pd

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def save_metrics_comparison(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = ["| Metric | RandomForest | XGBoost |", "|---|---:|---:|"]
    for k, rf_val, xgb_val in rows:
        rf_str = f"{rf_val:.4f}" if isinstance(rf_val, (int, float)) else str(rf_val)
        xgb_str = f"{xgb_val:.4f}" if isinstance(xgb_val, (int, float)) else str(xgb_val)
        lines.append(f"| {k} | {rf_str} | {xgb_str} |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))