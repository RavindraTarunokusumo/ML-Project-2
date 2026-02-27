# ML-Project-2: Covertype Classification Pipeline

Multi-class classification pipeline for the [Covertype dataset](https://www.openml.org/d/180) from OpenML. Predicts forest cover type (7 classes) from cartographic features.

## Dataset

- **Source**: OpenML — Covertype (dataset ID 159)
- **Samples**: ~581K
- **Features**: 54 (elevation, slope, aspect, wilderness areas, soil types)
- **Classes**: 7 forest cover types

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Train all models with default settings
python main.py

# Train specific models
python main.py --models rf xgb

# Enable hyperparameter tuning
python main.py --models rf xgb --optimize

# Custom dataset and split sizes
python main.py --dataset 159 --val_size 0.15 --test_size 0.15 --random_state 42
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `159` | OpenML dataset ID |
| `--target_col` | `Cover_Type` | Target column name |
| `--val_size` | `0.15` | Validation set proportion |
| `--test_size` | `0.15` | Test set proportion |
| `--random_state` | `42` | Random seed |
| `--optimize` | `False` | Enable GridSearchCV tuning |
| `--models` | `rf xgb gb svm` | Models to train |

## Models

| Key | Model |
|---|---|
| `rf` | RandomForestClassifier |
| `xgb` | XGBClassifier |
| `gb` | GradientBoostingClassifier |
| `svm` | SVC (with StandardScaler) |

## Project Structure

```
main.py              # CLI entry point
src/
  data.py            # DataLoader (OpenML + CSV)
  preprocessing.py   # Data splitting and feature preprocessing
  model.py           # Model pipelines and hyperparameter grids
  eval.py            # Validation, metrics, and grid search
pyproject.toml       # Dependencies and linter config
```

## Metrics Reported

- Cross-validation F1 (weighted, 5-fold)
- Accuracy, Precision, Recall, F1 — macro and weighted averages
- Per-class classification report
- Confusion matrix
