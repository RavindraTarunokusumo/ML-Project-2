# ML-Project-2

## What This Is

Classification pipeline for the Covertype dataset from OpenML. Predicts forest cover type (7 classes) from cartographic features.

## Core Value

Production-ready multi-class classification pipeline with hyperparameter optimization and comprehensive metrics.

## Dataset

- **Source**: OpenML (Covertype / dataset_159_covertype)
- **Samples**: ~581K
- **Features**: 54 (elevation, slope, aspect, wilderness areas, soil types)
- **Classes**: 7 (forest cover types)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Multi-class classification | Covertype has 7 classes | — |
| Precision/Recall/F1 | Toy datasets use accuracy; production needs per-class metrics | — |
| GridSearchCV | Consistent with ML-Project-1 | — |
| Classifier models | Different from ML-Project-1 regression | — |

## Requirements

### Active

- [x] Load Covertype from OpenML
- [x] Train/val/test split
- [x] Preprocessing pipeline
- [x] RandomForest classifier
- [x] XGBoost classifier (if available)
- [x] GradientBoosting classifier
- [x] SVM classifier
- [x] GridSearchCV hyperparameter tuning
- [x] Evaluation: precision, recall, F1 (macro/weighted)
- [x] CLI with args: --dataset, --target_col, --val_size, --test_size, --random_state, --optimize

### Out of Scope

- Neural networks — save for exploration phase
- Deployment/API — future milestone
- Experiment logging (JSON) — future milestone

---
*Last updated: 2026-02-27 — all initial requirements complete*
