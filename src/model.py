"""Model building utilities."""

from typing import Any

import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Models that don't need scaling (tree-based)
NO_SCALE_MODELS = {'rf', 'randomforest', 'gb', 'gradientboosting', 'xgb', 'xgboost'}

# Model aliases
MODEL_ALIASES = {
    'rf': RandomForestClassifier,
    'randomforest': RandomForestClassifier,
    'gb': GradientBoostingClassifier,
    'gradientboosting': GradientBoostingClassifier,
    'xgb': xgb.XGBClassifier,
    'xgboost': xgb.XGBClassifier,
    'svm': SVC,
    'svc': SVC,
}


def build_model_pipeline(
    model_name: str,
    X: Any,
    **model_kwargs
) -> Pipeline:
    """Build a model pipeline with preprocessing and classifier.

    Args:
        model_name: Model name or alias (rf, xgb, gb)
        X: Feature data (for fitting)
        **model_kwargs: Additional arguments to pass to the classifier

    Returns:
        sklearn Pipeline with preprocessing + model
    """
    from src.preprocessing import build_preprocessor

    # Resolve alias
    model_name_lower = model_name.lower()
    if model_name_lower not in MODEL_ALIASES:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_ALIASES.keys())}")

    model_cls = MODEL_ALIASES[model_name_lower]

    # SVM needs scaling; tree models don't
    use_scaler = model_name_lower not in NO_SCALE_MODELS

    # Build preprocessor
    preprocessor = build_preprocessor(X, use_scaler=use_scaler)

    # Build model
    if model_kwargs:
        model = model_cls(**model_kwargs)
    else:
        model = model_cls()

    # Create pipeline
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])

    return pipeline


def get_default_param_grid(model_name: str) -> dict:
    """Fallback to use the default parameter grid for GridSearchCV.

    Args:
        model_name: Model name or alias

    Returns:
        Parameter grid dict for GridSearchCV
    """
    model_name_lower = model_name.lower()

    if model_name_lower in ['rf', 'randomforest']:
        return {
            'model__n_estimators': [100, 200],
            'model__max_depth': [10, 20, None],
            'model__min_samples_split': [2, 5],
            'model__n_jobs': [-1]
        }
    elif model_name_lower in ['xgb', 'xgboost']:
        return {
            'model__n_estimators': [100, 200],
            'model__max_depth': [5, 10],
            'model__learning_rate': [0.05, 0.1],
            'model__n_jobs': [-1]
        }
    elif model_name_lower in ['gb', 'gradientboosting']:
        return {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 5],
            'model__learning_rate': [0.05, 0.1]
        }
    elif model_name_lower in ['svm', 'svc']:
        return {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['rbf', 'linear'],
            'model__gamma': ['scale', 'auto']
        }
    else:
        return {}
