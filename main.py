import os
import numpy as np
import pandas as pd

from src.config import CSV_PATH, FEATURES, TARGET, TEST_SIZE, RANDOM_STATE, N_SPLITS, FIG_DIR, MODEL_DIR
from src.data_loader import load_data
from src.preprocessing import split_features_target, train_test_split_data, build_numeric_pipeline, X_cols_placeholder
from src.visualization import plot_correlation_heatmap, plot_distributions, plot_pairplot, plot_feature_importance
from src.modeling import build_rf_pipeline, build_xgb_pipeline, cross_validate
from src.evaluation import evaluate_model, compare_metrics
from src.utils import save_model, save_metrics_comparison

def main():
    # Load
    df = load_data(CSV_PATH)

    # Basic cleaning (optional, robust defaults)
    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    # EDA plots
    plot_correlation_heatmap(df)
    plot_distributions(df)
    plot_pairplot(df, TARGET)

    # Split
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Preprocessor
    # Set placeholder column indices based on X columns
    from src import preprocessing as pre
    pre.X_cols_placeholder = list(range(X_train.shape[1]))
    preprocessor = build_numeric_pipeline()

    # Build models + CV
    rf_pipe, rf_grid = build_rf_pipeline(preprocessor)
    xgb_pipe, xgb_grid = build_xgb_pipeline(preprocessor)

    rf_cv = cross_validate(rf_pipe, rf_grid, X_train, y_train, n_splits=N_SPLITS)
    xgb_cv = cross_validate(xgb_pipe, xgb_grid, X_train, y_train, n_splits=N_SPLITS)

    rf_best = rf_cv.best_estimator_
    xgb_best = xgb_cv.best_estimator_

    # Fit best on full training set
    rf_best.fit(X_train, y_train)
    xgb_best.fit(X_train, y_train)

    # Evaluate
    rf_metrics = evaluate_model(rf_best, X_test, y_test, proba_supported=True)
    xgb_metrics = evaluate_model(xgb_best, X_test, y_test, proba_supported=True)

    # Feature importance (post-fit)
    # Extract model from pipelines
    rf_model = rf_best.named_steps['model']
    xgb_model = xgb_best.named_steps['model']

    # Use original feature names; importances aligned with transformed order because scaling preserves column order
    plot_feature_importance(
        importances=rf_model.feature_importances_,
        feature_names=FEATURES,
        title="Random Forest Feature Importance",
        filename="rf_feature_importance.png"
    )
    plot_feature_importance(
        importances=xgb_model.feature_importances_,
        feature_names=FEATURES,
        title="XGBoost Feature Importance",
        filename="xgb_feature_importance.png"
    )

    # Save models
    save_model(rf_best, os.path.join(MODEL_DIR, "rf_best.pkl"))
    save_model(xgb_best, os.path.join(MODEL_DIR, "xgb_best.pkl"))

    # Compare and save metrics
    rows = compare_metrics(rf_metrics, xgb_metrics)
    save_metrics_comparison(rows, "reports/metrics_comparison.md")

    # Print concise comparison
    print("Best RF params:", rf_cv.best_params_)
    print("Best XGB params:", xgb_cv.best_params_)
    print("RandomForest metrics:", rf_metrics)
    print("XGBoost metrics:", xgb_metrics)

    # Quick verdict
    rf_f1 = rf_metrics.get('f1', np.nan)
    xgb_f1 = xgb_metrics.get('f1', np.nan)
    winner = "XGBoost" if (xgb_f1 > rf_f1) else "RandomForest"
    print(f"Quick verdict (by F1): {winner}")

if __name__ == "__main__":
    main()