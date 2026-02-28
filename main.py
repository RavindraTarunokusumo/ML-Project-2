"""Main entry point for ML-Project-2 classification pipeline."""

import argparse

from src.data import DataLoader
from src.eval import print_evaluation, run_grid_search, validate_model
from src.model import build_model_pipeline, get_default_param_grid
from src.preprocessing import SplitConfig, split_data

# Default target column for Covertype v3
DEFAULT_TARGET = "class"


PARAM_GRID = {
    'rf': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 20, None],
        'model__min_samples_split': [2, 5],
    },
    'xgb': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [3, 6],
        'model__learning_rate': [0.01, 0.1],
    },
    'gb': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [3, 6],
        'model__learning_rate': [0.01, 0.1],
    },
    'svm': {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf'],
    }
}


def main(args):
    """Run the classification pipeline."""

    print(f"\n{'='*60}")
    print("ML-Project-2: Classification Pipeline")
    print(f"{'='*60}")
    print(f"Dataset ID: {args.dataset}")
    print(f"Target: {args.target_col}")
    print(f"Splits: val={args.val_size}, test={args.test_size}")
    print(f"Random state: {args.random_state}")
    print(f"Optimize: {args.optimize}")
    print(f"Resampling: {not args.no_resample}")
    print(f"Models: {args.models}")

    # Load data
    print(f"\nLoading dataset {args.dataset} from OpenML...")
    df = DataLoader.load_openml(args.dataset)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    # Separate features and target
    X = df.drop(columns=[args.target_col])
    y = df[args.target_col]

    print(f"Features: {X.shape[1]}")
    print(f"Classes: {y.nunique()}")

    # Split data
    split_config = SplitConfig(
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state
    )
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y, split_config)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train models
    results = {}
    for model_name in args.models:
        print(f"\n{'='*40}")
        print(f"Training {model_name.upper()}...")
        print(f"{'='*40}")

        # Build model
        pipeline = build_model_pipeline(
            model_name,
            X_train,
            y_train,
            resample_training=not args.no_resample,
            random_state=args.random_state,
        )

        if args.optimize:
            # Run grid search on training set
            if PARAM_GRID.get(model_name.lower()):
                param_grid = PARAM_GRID[model_name.lower()]
            else:
                param_grid = get_default_param_grid(model_name)
            print(f"Running GridSearchCV with param grid: {param_grid}")
            grid_search = run_grid_search(
                pipeline, X_train, y_train, param_grid, cv=3
            )
            best_model = grid_search.best_estimator_
            # Evaluate on validation
            metrics, y_pred = validate_model(
                best_model, X_train, y_train, X_val, y_val
            )
        else:
            # Basic training
            metrics, y_pred = validate_model(
                pipeline, X_train, y_train, X_val, y_val
            )
            best_model = pipeline

        print_evaluation(metrics, y_val, y_pred)
        results[model_name] = {'metrics': metrics, 'model': best_model}

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_name, result in results.items():
        m = result['metrics']
        print(f"{model_name.upper():8} | F1 (macro): {m['f1_macro']:.4f} | F1 (weighted): {m['f1_weighted']:.4f}")

    print(f"\n{'='*60}")
    print("Pipeline complete!")

if __name__ == "__main__":
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ML-Project-2: Classification Pipeline")
    parser.add_argument(
        "--dataset",
        type=int,
        default=159,
        help="OpenML dataset ID (default: 159 = Covertype)"
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default=DEFAULT_TARGET,
        help=f"Target column name (default: {DEFAULT_TARGET})"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Validation set proportion (default: 0.15)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Test set proportion (default: 0.15)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run GridSearchCV hyperparameter tuning"
    )
    parser.add_argument(
        "--no_resample",
        action="store_true",
        help="Disable undersampling and SMOTE on the training folds"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rf", "xgb", "gb", "svm"],
        help="Models to train (default: rf xgb gb svm)"
    )
    args = parser.parse_args()  
    main(args)
