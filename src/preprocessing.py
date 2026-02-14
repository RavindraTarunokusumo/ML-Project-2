"""Preprocessing utilities."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class SplitConfig:
    """Configuration for data splits."""

    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: SplitConfig | None = None
) -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
    """Split data into train, validation, and test sets.

    Args:
        X: Feature DataFrame
        y: Target Series
        config: Split configuration

    Returns:
        ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    if config is None:
        config = SplitConfig()

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y  # Stratify for classification
    )

    # Second split: train vs val
    val_adjusted = config.val_size / (1 - config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_adjusted,
        random_state=config.random_state,
        stratify=y_trainval
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def identify_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical columns.

    Args:
        df: Feature DataFrame

    Returns:
        (numeric_columns, categorical_columns)
    """
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric, categorical


def build_preprocessor(
    X: pd.DataFrame,
    use_scaler: bool = True,
    use_pca: bool = False,
    pca_components: int | None = None
) -> ColumnTransformer:
    """Build preprocessing pipeline.

    Args:
        X: Feature DataFrame (for column identification)
        use_scaler: Whether to apply StandardScaler to numeric features
        use_pca: Whether to apply PCA (not typically needed for tree models)
        pca_components: Number of PCA components if use_pca=True

    Returns:
        Fitted ColumnTransformer
    """
    numeric_cols, categorical_cols = identify_column_types(X)

    transformers = []

    # Numeric: impute + optional scale
    if numeric_cols:
        numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
        if use_scaler:
            numeric_steps.append(('scaler', StandardScaler()))
        transformers.append(('num', numeric_steps, numeric_cols))

    # Categorical: impute + one-hot
    if categorical_cols:
        transformers.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )
