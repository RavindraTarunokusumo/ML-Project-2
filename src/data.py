"""Data loading utilities."""

import pandas as pd
import openml
from pathlib import Path


class DataLoader:
    """Handles loading data from various sources."""

    @staticmethod
    def load_openml(dataset_id: int = 159, dataset_name: str = "covertype") -> pd.DataFrame:
        """Load dataset from OpenML.

        Args:
            dataset_id: OpenML dataset ID (159 = Covertype)
            dataset_name: Name for caching

        Returns:
            pandas DataFrame
        """
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        df = X.copy()
        df[dataset.default_target_attribute] = y
        return df

    @staticmethod
    def load_csv(path: str | Path) -> pd.DataFrame:
        """Load dataset from CSV file.

        Args:
            path: Path to CSV file

        Returns:
            pandas DataFrame
        """
        return pd.read_csv(path)
