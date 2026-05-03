import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .config import FEATURES, TARGET, TEST_SIZE, RANDOM_STATE

def split_features_target(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y

def train_test_split_data(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

def build_numeric_pipeline():
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[('num', num_transformer, list(range(len(X_cols_placeholder))))],
        remainder='drop'
    )
    return preprocessor

# Helper to avoid circular dependency; will be set in main after columns known
X_cols_placeholder = None