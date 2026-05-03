import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from .config import FIG_DIR, FEATURES

def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)

def plot_correlation_heatmap(df: pd.DataFrame):
    ensure_fig_dir()
    corr = df[FEATURES + [col for col in df.columns if col not in FEATURES][:0]].corr()  # corr among features
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/correlation_heatmap.png")
    plt.close()

def plot_distributions(df: pd.DataFrame):
    ensure_fig_dir()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for i, col in enumerate(FEATURES):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#2b8a3e")
        axes[i].set_title(f"Distribution: {col}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/distributions.png")
    plt.close()

def plot_pairplot(df: pd.DataFrame, target_col: str):
    ensure_fig_dir()
    sns.pairplot(df[FEATURES + [target_col]], hue=target_col, diag_kind="kde")
    plt.savefig(f"{FIG_DIR}/pairplot.png")
    plt.close()

def plot_feature_importance(importances, feature_names, title, filename):
    ensure_fig_dir()
    importances = importances
    order = importances.argsort()[::-1]
    sorted_feats = [feature_names[i] for i in order]
    sorted_imps = importances[order]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=sorted_imps, y=sorted_feats, palette="viridis")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{filename}")
    plt.close()