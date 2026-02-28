"""
Exploratory analysis for the OpenML covertype v3 dataset.

Generates a markdown report with chart images in report/.
Run from the ML-Project-2 directory:
    ..\.venv311\Scripts\python.exe -m src.explore
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import mutual_info_classif

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "report"
IMG_DIR = REPORT_DIR / "images"
REPORT_PATH = REPORT_DIR / "exploration_report.md"

DATASET_NAME = "covertype"
DATASET_VERSION = 3
TARGET = "class"
RANDOM_STATE = 42

sns.set_theme(style="whitegrid")


def load_covertype_v3() -> pd.DataFrame:
    """Load the Covertype v3 dataset from OpenML."""
    dataset = fetch_openml(
        DATASET_NAME,
        version=DATASET_VERSION,
        as_frame=True,
    )
    if dataset.frame is None:
        msg = "OpenML did not return a tabular frame for covertype v3."
        raise RuntimeError(msg)
    return dataset.frame.copy()


def _sorted_indicator_columns(
    df: pd.DataFrame, prefix: str
) -> list[str]:
    return sorted(
        [col for col in df.columns if col.startswith(prefix)],
        key=lambda col: int(col.removeprefix(prefix)),
    )


def get_feature_groups(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """Split the schema into continuous and one-hot indicator groups."""
    continuous_cols = [
        col
        for col in df.columns
        if col != TARGET
        and not col.startswith("Wilderness_Area")
        and not col.startswith("Soil_Type")
    ]
    wilderness_cols = _sorted_indicator_columns(df, "Wilderness_Area")
    soil_cols = _sorted_indicator_columns(df, "Soil_Type")
    return continuous_cols, wilderness_cols, soil_cols


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(series.astype(str), errors="coerce")


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[bool]]:
    """Build a numeric feature matrix for feature scoring."""
    continuous_cols, wilderness_cols, soil_cols = get_feature_groups(df)
    feature_cols = continuous_cols + wilderness_cols + soil_cols
    X = pd.DataFrame(index=df.index)
    for col in feature_cols:
        X[col] = _coerce_numeric(df[col])
    discrete_mask = [col not in continuous_cols for col in feature_cols]
    return X, discrete_mask


def _decode_indicator_group(
    df: pd.DataFrame,
    columns: list[str],
    label_prefix: str,
) -> pd.Series:
    encoded = pd.DataFrame(index=df.index)
    for col in columns:
        encoded[col] = _coerce_numeric(df[col]).fillna(0)

    winning_col = encoded.idxmax(axis=1)
    has_indicator = encoded.sum(axis=1) > 0
    suffix = winning_col.str.extract(r"(\d+)$", expand=False)
    labels = label_prefix + " " + suffix.fillna("?")
    return labels.where(has_indicator, other="Unknown")


def build_indicator_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Derive readable wilderness-area and soil-type labels."""
    _, wilderness_cols, soil_cols = get_feature_groups(df)
    return pd.DataFrame(
        {
            "Wilderness_Area": _decode_indicator_group(
                df, wilderness_cols, "Area"
            ),
            "Soil_Type": _decode_indicator_group(
                df, soil_cols, "Type"
            ),
        },
        index=df.index,
    )


def _sample_frame(df: pd.DataFrame, size: int) -> pd.DataFrame:
    if len(df) <= size:
        return df.copy()
    return df.sample(size, random_state=RANDOM_STATE)


def _save(fig: plt.Figure, name: str) -> str:
    path = IMG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"images/{name}.png"


def _md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    display_df = df.copy()
    if max_rows is not None:
        display_df = display_df.head(max_rows)

    lines = [
        "| " + " | ".join(str(col) for col in display_df.columns) + " |",
        "| " + " | ".join("---" for _ in display_df.columns) + " |",
    ]
    for _, row in display_df.iterrows():
        lines.append(
            "| " + " | ".join(str(value) for value in row) + " |"
        )

    if max_rows is not None and len(df) > max_rows:
        lines.append("")
        lines.append(
            f"_Showing first {max_rows} of {len(df)} rows._"
        )
    return "\n".join(lines)


def _label_sort_key(label: str) -> tuple[int, int, str]:
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    if digits:
        return (0, int(digits), str(label))
    return (1, 0, str(label))


def dataset_overview(df: pd.DataFrame) -> str:
    """Summarize the Covertype v3 schema and data size."""
    continuous_cols, wilderness_cols, soil_cols = get_feature_groups(df)
    class_count = df[TARGET].astype(str).nunique()
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2

    overview = pd.DataFrame(
        {
            "Metric": [
                "Rows",
                "Columns",
                "Continuous features",
                "Wilderness indicators",
                "Soil indicators",
                "Target column",
                "Target classes",
                "Duplicate rows",
                "Missing cells",
                "Memory usage (MB)",
            ],
            "Value": [
                len(df),
                len(df.columns),
                len(continuous_cols),
                len(wilderness_cols),
                len(soil_cols),
                TARGET,
                class_count,
                int(df.duplicated().sum()),
                int(df.isna().sum().sum()),
                round(memory_mb, 2),
            ],
        }
    )

    return f"""## 1. Dataset Overview

{_md_table(overview)}
"""


def column_catalogue(df: pd.DataFrame) -> str:
    """List all columns with dtypes, null counts, and unique counts."""
    info = pd.DataFrame(
        {
            "Column": df.columns,
            "Dtype": [str(df[col].dtype) for col in df.columns],
            "Non-Null": [int(df[col].notna().sum()) for col in df.columns],
            "Null%": [
                f"{df[col].isna().mean() * 100:.2f}%"
                for col in df.columns
            ],
            "Unique": [int(df[col].nunique(dropna=False)) for col in df.columns],
        }
    )

    return f"""## 2. Column Catalogue

{_md_table(info)}
"""


def missing_values(df: pd.DataFrame) -> str:
    """Report missingness. Covertype v3 is typically complete."""
    missing = (
        df.isna().mean().mul(100).sort_values(ascending=False)
    )
    missing = missing[missing > 0]

    if missing.empty:
        return """## 3. Missing Values

No missing values were found in Covertype v3.
"""

    missing_df = missing.reset_index()
    missing_df.columns = ["Feature", "Missing%"]
    missing_df["Missing%"] = missing_df["Missing%"].map(
        lambda value: f"{value:.2f}%"
    )

    fig, ax = plt.subplots(figsize=(10, max(4, len(missing_df) * 0.25)))
    ax.barh(
        missing_df["Feature"][::-1],
        missing.iloc[::-1],
        color="#d97757",
    )
    ax.set_xlabel("Missing %")
    ax.set_title("Missingness by Feature")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    img = _save(fig, "missing_values")

    return f"""## 3. Missing Values

{_md_table(missing_df)}

![Missing values]({img})
"""


def class_distribution(df: pd.DataFrame) -> str:
    """Describe the multiclass target distribution."""
    counts = (
        df[TARGET]
        .astype(str)
        .value_counts()
        .sort_index(key=lambda idx: idx.astype(int))
    )
    pct = counts.div(len(df)).mul(100)
    imbalance_ratio = counts.max() / counts.min()

    summary = pd.DataFrame(
        {
            "Class": counts.index,
            "Count": counts.values,
            "Share%": pct.round(2).values,
        }
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        summary["Class"],
        summary["Count"],
        color=sns.color_palette("crest", n_colors=len(summary)),
    )
    ax.set_title("Target Class Distribution")
    ax.set_xlabel(TARGET)
    ax.set_ylabel("Count")
    ax.bar_label(bars, padding=3, fontsize=8)
    img = _save(fig, "class_distribution")

    return f"""## 4. Target Distribution

- Largest class share: {pct.max():.2f}%
- Smallest class share: {pct.min():.2f}%
- Imbalance ratio (largest / smallest): {imbalance_ratio:.2f}

{_md_table(summary)}

![Class distribution]({img})
"""


def training_policy() -> str:
    """Document the preprocessing and resampling policy used in training."""
    return """## 5. Training Preprocessing Policy

- `Soil_Type1` to `Soil_Type40` are collapsed into one categorical `Soil_Type` feature before encoding.
- `Wilderness_Area1` to `Wilderness_Area4` are collapsed into one categorical `Wilderness_Area` feature before encoding.
- Class rebalancing is applied only on the training split and inside cross-validation folds. Validation and test sets remain untouched.
- Majority classes `1` and `2` are randomly undersampled down to the per-class mean target count.
- Minority classes `3` to `7` are oversampled with `SMOTENC` up to the same target count.
- The rebalance target is `round(training_rows / 7)`, which keeps the final training set size approximately unchanged while flattening the class distribution.
"""


def continuous_feature_summary(
    df: pd.DataFrame, plot_sample: int
) -> str:
    """Summarize the ten continuous terrain features."""
    continuous_cols, _, _ = get_feature_groups(df)
    numeric = df[continuous_cols].apply(_coerce_numeric)

    desc = numeric.describe().T.round(4)
    desc["skewness"] = numeric.skew().round(4)
    desc.insert(0, "Feature", desc.index)
    desc = desc.reset_index(drop=True)

    scaled_to_unit = int(
        ((numeric.min() >= 0) & (numeric.max() <= 1)).sum()
    )

    sample = _sample_frame(numeric, min(plot_sample, 100_000))
    fig, axes = plt.subplots(5, 2, figsize=(14, 18))
    axes = axes.flatten()
    for i, col in enumerate(continuous_cols):
        axes[i].hist(
            sample[col].dropna(),
            bins=30,
            color="#4c8da8",
            edgecolor="white",
        )
        axes[i].set_title(col, fontsize=10)
        axes[i].tick_params(labelsize=8)
    fig.suptitle("Continuous Feature Distributions", fontsize=14, y=1.0)
    fig.tight_layout()
    img = _save(fig, "continuous_feature_distributions")

    return f"""## 6. Continuous Feature Summary

- Continuous features detected: {len(continuous_cols)}
- Features bounded to [0, 1]: {scaled_to_unit} of {len(continuous_cols)}
- Histogram sample size: {len(sample):,}

{_md_table(desc)}

![Continuous feature distributions]({img})
"""


def compute_mutual_information(
    df: pd.DataFrame, mi_sample: int
) -> pd.Series:
    """Score features against the multiclass target."""
    X, discrete_mask = build_feature_matrix(df)
    sampled = _sample_frame(X.join(df[TARGET].astype(str)), mi_sample)
    X_sample = sampled.drop(columns=[TARGET])
    y_sample = sampled[TARGET]

    mi_values = mutual_info_classif(
        X_sample,
        y_sample,
        discrete_features=discrete_mask,
        random_state=RANDOM_STATE,
    )
    return pd.Series(mi_values, index=X_sample.columns).sort_values(
        ascending=False
    )


def mutual_information_section(mi_scores: pd.Series, mi_sample: int) -> str:
    """Format feature-signal output and save a ranking chart."""
    top = mi_scores.head(15).round(4).reset_index()
    top.columns = ["Feature", "Mutual Information"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        top["Feature"][::-1],
        top["Mutual Information"][::-1],
        color="#2a9d8f",
    )
    ax.set_title("Top Features by Mutual Information")
    ax.set_xlabel("Mutual information")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    img = _save(fig, "mutual_information")

    return f"""## 7. Feature Signal

- Mutual information computed on a sample of {mi_sample:,} rows.
- Higher scores indicate more information about the target classes.

{_md_table(top)}

![Top features by mutual information]({img})
"""


def feature_relationships(
    df: pd.DataFrame,
    mi_scores: pd.Series,
    plot_sample: int,
) -> str:
    """Analyze correlations and class-specific continuous feature patterns."""
    continuous_cols, _, _ = get_feature_groups(df)
    numeric = df[continuous_cols].apply(_coerce_numeric)
    corr = numeric.corr().round(3)
    mask = np.tril(np.ones(corr.shape, dtype=bool))

    pairs = (
        corr.where(~mask)
        .stack()
        .sort_values(key=lambda series: series.abs(), ascending=False)
        .head(10)
        .reset_index()
    )
    pairs.columns = ["Feature A", "Feature B", "Correlation"]
    pairs["Correlation"] = pairs["Correlation"].round(3)

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        square=True,
        ax=ax1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax1.set_title("Continuous Feature Correlation Heatmap")
    fig1.tight_layout()
    corr_img = _save(fig1, "continuous_correlation_heatmap")

    top_continuous = [
        feature for feature in mi_scores.index if feature in continuous_cols
    ][:4]
    sampled = _sample_frame(
        df[top_continuous + [TARGET]].copy(), plot_sample
    )

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    axes2 = axes2.flatten()
    for i, col in enumerate(top_continuous):
        sns.boxplot(
            data=sampled,
            x=TARGET,
            y=col,
            hue=TARGET,
            legend=False,
            ax=axes2[i],
            palette="Set2",
        )
        axes2[i].set_title(f"{col} by class")
        axes2[i].tick_params(axis="x", labelsize=8)
    fig2.tight_layout()
    box_img = _save(fig2, "top_continuous_by_class")

    class_means = (
        numeric.join(df[TARGET].astype(str))
        .groupby(TARGET)
        .mean()
        .sort_index(key=lambda idx: idx.astype(int))
    )
    scaled_profiles = (
        class_means - numeric.mean()
    ).div(numeric.std(ddof=0).replace(0, 1))

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        scaled_profiles,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax3,
        cbar_kws={"shrink": 0.8},
    )
    ax3.set_title("Class-wise Standardized Mean Profiles")
    ax3.set_xlabel("Continuous feature")
    ax3.set_ylabel(TARGET)
    fig3.tight_layout()
    profile_img = _save(fig3, "class_profiles")

    return f"""## 8. Feature Relationships

### Strongest continuous-feature correlations

{_md_table(pairs)}

### Continuous-feature correlations

![Correlation heatmap]({corr_img})

### Top continuous features by class

![Top continuous features by class]({box_img})

### Standardized class profiles

![Class profiles]({profile_img})
"""


def terrain_indicator_analysis(df: pd.DataFrame) -> str:
    """Analyze the encoded wilderness-area and soil-type indicators."""
    derived = build_indicator_labels(df)
    terrain = derived.join(df[TARGET].astype(str))

    wilderness_counts = (
        terrain["Wilderness_Area"]
        .value_counts()
        .reindex(
            sorted(
                terrain["Wilderness_Area"].unique(),
                key=_label_sort_key,
            )
        )
    )
    wilderness_df = pd.DataFrame(
        {
            "Wilderness_Area": wilderness_counts.index,
            "Count": wilderness_counts.values,
            "Share%": wilderness_counts.div(len(df)).mul(100).round(2).values,
        }
    )

    soil_counts = terrain["Soil_Type"].value_counts()
    top_soils = soil_counts.head(15)
    soil_df = pd.DataFrame(
        {
            "Soil_Type": top_soils.index,
            "Count": top_soils.values,
            "Share%": top_soils.div(len(df)).mul(100).round(2).values,
        }
    )

    wilderness_share = (
        pd.crosstab(
            terrain["Wilderness_Area"],
            terrain[TARGET],
            normalize="index",
        )
        .mul(100)
        .reindex(
            sorted(
                terrain["Wilderness_Area"].unique(),
                key=_label_sort_key,
            )
        )
    )
    soil_share = (
        pd.crosstab(
            terrain["Soil_Type"],
            terrain[TARGET],
            normalize="index",
        )
        .mul(100)
        .loc[top_soils.index]
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    sns.heatmap(
        wilderness_share,
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title("Class mix within wilderness areas (%)")
    axes[0].set_xlabel(TARGET)
    axes[0].set_ylabel("Wilderness area")

    sns.heatmap(
        soil_share,
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        ax=axes[1],
        cbar=False,
    )
    axes[1].set_title("Class mix within top soil types (%)")
    axes[1].set_xlabel(TARGET)
    axes[1].set_ylabel("Soil type")
    fig.tight_layout()
    img = _save(fig, "terrain_indicator_heatmaps")

    return f"""## 9. Encoded Terrain Indicators

### Wilderness area distribution

{_md_table(wilderness_df)}

### Top soil types by frequency

{_md_table(soil_df)}

### Class mix by derived indicator groups

![Terrain indicator heatmaps]({img})
"""


def key_findings(df: pd.DataFrame, mi_scores: pd.Series) -> str:
    """Summarize the main takeaways for downstream work."""
    continuous_cols, _, _ = get_feature_groups(df)
    class_counts = (
        df[TARGET]
        .astype(str)
        .value_counts()
        .sort_index(key=lambda idx: idx.astype(int))
    )
    derived = build_indicator_labels(df)
    top_wilderness = derived["Wilderness_Area"].value_counts().idxmax()
    top_soil = derived["Soil_Type"].value_counts().idxmax()
    top_continuous = [
        feature for feature in mi_scores.index if feature in continuous_cols
    ][:3]
    top_indicators = [
        feature for feature in mi_scores.index if feature not in continuous_cols
    ][:3]

    lines = [
        "## 10. Key Findings",
        "",
        f"- Covertype v3 contains {len(df):,} rows and "
        f"{len(df.columns) - 1} input features.",
        f"- The target `{TARGET}` has {class_counts.size} classes with a "
        f"{class_counts.max() / class_counts.min():.2f}x "
        "largest-to-smallest class ratio.",
        "- All ten continuous terrain measurements appear to be scaled to "
        "[0, 1], so the exploration reflects the normalized version of "
        "the dataset.",
        "- Top continuous signal features by mutual information: "
        + ", ".join(f"`{feature}`" for feature in top_continuous)
        + ".",
        "- Top binary indicator features by mutual information: "
        + ", ".join(f"`{feature}`" for feature in top_indicators)
        + ".",
        f"- Most common wilderness area: `{top_wilderness}`.",
        f"- Most common soil type: `{top_soil}`.",
        "- No missing values were detected, so preprocessing can focus on "
        "class imbalance handling, feature scaling policy, and model "
        "selection rather than imputation.",
        "",
    ]
    return "\n".join(lines)


def build_report(
    df: pd.DataFrame,
    mi_sample: int = 50_000,
    plot_sample: int = 20_000,
) -> str:
    """Run the full exploration workflow and assemble markdown output."""
    mi_scores = compute_mutual_information(df, mi_sample)

    sections = [
        "# Covertype v3 Data Exploration Report\n",
        (
            "Dataset: [OpenML Covertype v3 (dataset id 159)]"
            "(https://www.openml.org/search?type=data&sort=runs&id=159"
            "&status=active)  \n"
            "Generated by `src/explore.py` using the root `.venv311` "
            "environment.\n"
        ),
        "---\n",
        dataset_overview(df),
        column_catalogue(df),
        missing_values(df),
        class_distribution(df),
        training_policy(),
        continuous_feature_summary(df, plot_sample),
        mutual_information_section(mi_scores, min(mi_sample, len(df))),
        feature_relationships(df, mi_scores, plot_sample),
        terrain_indicator_analysis(df),
        key_findings(df, mi_scores),
    ]
    return "\n".join(sections)


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Explore the OpenML Covertype v3 dataset."
    )
    parser.add_argument(
        "--mi-sample",
        type=int,
        default=50_000,
        help="Rows to sample for mutual-information scoring.",
    )
    parser.add_argument(
        "--plot-sample",
        type=int,
        default=20_000,
        help="Rows to sample for class-wise boxplots and histograms.",
    )
    return parser.parse_args()


def main(args: Namespace | None = None) -> None:
    """Load Covertype v3, run exploration, and write the report."""
    if args is None:
        args = parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading OpenML covertype v3...")
    df = load_covertype_v3()
    print(f"Dataset shape: {df.shape}")
    print("Building exploration report...")

    report = build_report(
        df,
        mi_sample=args.mi_sample,
        plot_sample=args.plot_sample,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Report written to {REPORT_PATH}")
    print(f"Images saved to {IMG_DIR}")


if __name__ == "__main__":
    main()
