from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

def build_rf_pipeline(preprocessor):
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    pipe = Pipeline(steps=[('preprocess', preprocessor), ('model', rf)])
    param_grid = {
        'model__n_estimators': [200, 400],
        'model__max_depth': [None, 6, 10],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2],
        'model__max_features': ['sqrt', 'log2']
    }
    return pipe, param_grid

def build_xgb_pipeline(preprocessor):
    xgb = XGBClassifier(
        random_state=42,
        n_estimators=400,
        n_jobs=-1,
        eval_metric='logloss',
        tree_method='hist'  # fast, good default
    )
    pipe = Pipeline(steps=[('preprocess', preprocessor), ('model', xgb)])
    param_grid = {
        'model__n_estimators': [300, 500],
        'model__max_depth': [3, 6, 9],
        'model__learning_rate': [0.05, 0.1],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0],
        'model__reg_lambda': [1.0, 2.0]
    }
    return pipe, param_grid

def cross_validate(pipe, param_grid, X_train, y_train, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='f1',  # change to 'roc_auc' if binary and you prefer
        cv=cv,
        n_jobs=-1,
        return_train_score=True
    )
    grid.fit(X_train, y_train)
    return grid