"""Preprocessing utilities."""

from dataclasses import dataclass
from typing import Any

from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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


class IndicatorCollapseTransformer(BaseEstimator, TransformerMixin):
    """Collapse one-hot indicator groups into a single categorical column."""

    def __init__(
        self,
        prefix: str,
        output_column: str,
        unknown_label: str = "Unknown"
    ) -> None:
        self.prefix = prefix
        self.output_column = output_column
        self.unknown_label = unknown_label
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Remember the indicator columns present during fitting."""
        df = self._to_dataframe(X)
        self.columns_ = self._indicator_columns(df)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace indicator columns with a single categorical column."""
        df = self._to_dataframe(X).copy()
        columns = [
            col for col in self.columns_
            if col in df.columns
        ] or self._indicator_columns(df)

        if not columns:
            return df

        encoded = df[columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        winning_col = encoded.idxmax(axis=1)
        has_indicator = encoded.sum(axis=1) > 0
        suffix = winning_col.str.extract(r"(\d+)$", expand=False)
        labels = (
            f"{self.output_column}_"
            + suffix.fillna(self.unknown_label)
        )
        df[self.output_column] = labels.where(
            has_indicator,
            other=self.unknown_label,
        ).astype("category")
        return df.drop(columns=columns)

    def _indicator_columns(self, df: pd.DataFrame) -> list[str]:
        return sorted(
            [col for col in df.columns if col.startswith(self.prefix)],
            key=lambda col: int(col.removeprefix(self.prefix)),
        )

    @staticmethod
    def _to_dataframe(X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


class FeatureGroupCollapser(BaseEstimator, TransformerMixin):
    """Collapse multiple indicator groups into categorical columns."""

    def __init__(
        self,
        groups: list[tuple[str, str]] | None = None,
        unknown_label: str = "Unknown"
    ) -> None:
        if groups is None:
            groups = [
                ("Soil_Type", "Soil_Type"),
                ("Wilderness_Area", "Wilderness_Area"),
            ]
        self.groups = groups
        self.unknown_label = unknown_label
        self.transformers_: list[IndicatorCollapseTransformer] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit one collapse transformer per configured indicator group."""
        df = self._to_dataframe(X)
        self.transformers_ = []
        for prefix, output_column in self.groups:
            transformer = IndicatorCollapseTransformer(
                prefix=prefix,
                output_column=output_column,
                unknown_label=self.unknown_label,
            )
            transformer.fit(df)
            self.transformers_.append(transformer)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all configured group collapses in sequence."""
        df = self._to_dataframe(X).copy()
        for transformer in self.transformers_:
            df = transformer.transform(df)
        return df

    @staticmethod
    def _to_dataframe(X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


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


def build_feature_collapser() -> FeatureGroupCollapser:
    """Create the shared feature-group collapsing transformer."""
    return FeatureGroupCollapser()


def _column_selector(
    df: pd.DataFrame,
    columns: list[str],
    use_column_names: bool,
) -> list[str] | list[int]:
    if use_column_names:
        return columns
    return [df.columns.get_loc(col) for col in columns]


def _build_column_transformer(
    X: pd.DataFrame,
    use_scaler: bool = True,
    use_column_names: bool = True,
) -> ColumnTransformer:
    """Build the numeric/categorical column transformer."""
    numeric_cols, categorical_cols = identify_column_types(X)

    transformers = []

    if numeric_cols:
        numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
        if use_scaler:
            numeric_steps.append(('scaler', StandardScaler()))
        transformers.append((
            'num',
            Pipeline(numeric_steps),
            _column_selector(X, numeric_cols, use_column_names),
        ))

    if categorical_cols:
        transformers.append((
            'cat',
            Pipeline([
                ('imputer', SimpleImputer(
                    strategy='constant',
                    fill_value='missing',
                )),
                ('onehot', OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False,
                )),
            ]),
            _column_selector(X, categorical_cols, use_column_names),
        ))

    return ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )


def build_resampling_steps(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> list[tuple[str, Any]]:
    """Build undersampling + SMOTENC steps for the training data."""
    y_series = pd.Series(y).reset_index(drop=True)
    class_counts = y_series.astype(str).value_counts().sort_index()
    if class_counts.empty:
        return []

    target_count = int(round(len(y_series) / class_counts.size))
    majority_classes = {'1', '2'}

    under_strategy = {
        label: target_count
        for label, count in class_counts.items()
        if label in majority_classes and count > target_count
    }
    over_strategy = {
        label: target_count
        for label, count in class_counts.items()
        if label not in majority_classes and count < target_count
    }

    steps: list[tuple[str, Any]] = []
    if under_strategy:
        steps.append((
            'undersample_majority',
            RandomUnderSampler(
                sampling_strategy=under_strategy,
                random_state=random_state,
            ),
        ))

    if over_strategy:
        _, categorical_cols = identify_column_types(X)
        categorical_indices = [
            X.columns.get_loc(col) for col in categorical_cols
        ]
        minority_floor = min(
            int(class_counts[label]) for label in over_strategy
        )
        # Keep SMOTE stable inside CV folds when the rarest class is small.
        k_neighbors = max(1, min(5, minority_floor // 5))
        steps.append((
            'smote_minority',
            SMOTENC(
                categorical_features=categorical_indices,
                sampling_strategy=over_strategy,
                random_state=random_state,
                k_neighbors=k_neighbors,
            ),
        ))

    return steps


def build_preprocessor(
    X: pd.DataFrame,
    use_scaler: bool = True,
    use_pca: bool = False,
    pca_components: int | None = None,
    collapse_feature_groups: bool = True,
    use_column_names: bool = True,
) -> Pipeline | ColumnTransformer:
    """Build preprocessing pipeline.

    Args:
        X: Feature DataFrame (for column identification)
        use_scaler: Whether to apply StandardScaler to numeric features
        use_pca: Whether to apply PCA (not typically needed for tree models)
        pca_components: Number of PCA components if use_pca=True

    Returns:
        sklearn Pipeline with feature collapsing + column preprocessing
    """
    if collapse_feature_groups:
        collapse = build_feature_collapser()
        collapsed_X = collapse.fit_transform(X)
        column_transformer = _build_column_transformer(
            collapsed_X,
            use_scaler=use_scaler,
            use_column_names=use_column_names,
        )
        return Pipeline([
            ('collapse_feature_groups', collapse),
            ('columns', column_transformer),
        ])

    column_transformer = _build_column_transformer(
        X,
        use_scaler=use_scaler,
        use_column_names=use_column_names,
    )
    return column_transformer
