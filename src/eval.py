"""Evaluation utilities."""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def validate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cv: int = 5,
    scoring: str = 'f1_weighted'
) -> dict:
    """Validate model with cross-validation and test set evaluation.

    Args:
        model: Fitted sklearn model or pipeline
        X_train: Training features
        y_train: Training labels
        X_val: Validation/test features
        y_val: Validation/test labels
        cv: Number of cross-validation folds
        scoring: Scoring metric for CV

    Returns:
        Dictionary with evaluation metrics
    """
    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)

    # Fit on full training set and predict on validation
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # Calculate metrics
    metrics = {
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision_macro': precision_score(y_val, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_val, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_val, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_val, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_val, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_val, y_pred, average='weighted', zero_division=0),
    }

    return metrics, y_pred


def print_evaluation(metrics: dict, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print evaluation results.

    Args:
        metrics: Dictionary of metrics from validate_model
        y_true: True labels
        y_pred: Predicted labels
    """
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")

    print(f"\nCross-Validation (5-fold, F1 weighted):")
    print(f"  Mean: {metrics['cv_f1_mean']:.4f}")
    print(f"  Std:  {metrics['cv_f1_std']:.4f}")

    print(f"\nValidation Set:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):         {metrics['f1_macro']:.4f}")
    print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted):   {metrics['recall_weighted']:.4f}")
    print(f"  F1 (weighted):       {metrics['f1_weighted']:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    print(f"Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def run_grid_search(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict,
    cv: int = 3,
    scoring: str = 'f1_weighted',
    n_jobs: int = -1
) -> GridSearchCV:
    """Run grid search for hyperparameter tuning.

    Args:
        model: sklearn model or pipeline
        X_train: Training features
        y_train: Training labels
        param_grid: Parameter grid for GridSearchCV
        cv: Number of CV folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs

    Returns:
        Fitted GridSearchCV object
    """
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score:  {grid_search.best_score_:.4f}")

    return grid_search
