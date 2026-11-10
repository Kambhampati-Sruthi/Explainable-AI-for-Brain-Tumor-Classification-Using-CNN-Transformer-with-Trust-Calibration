
"""
Statistical analysis of stratified segmentation XAI results.

Input:
    report/segmentation_xai_spatial/stratified_xai_metrics.csv

Outputs:
    report/segmentation_xai_spatial/statistical_analysis/
        descriptive_statistics.csv
        bootstrap_confidence_intervals.csv
        paired_method_comparisons.csv
        dice_group_comparisons.csv
        statistical_summary.txt

Important:
    - The independent sampling unit is the CASE, not the CSV row.
    - Predicted-region and ground-truth-oracle rows are paired by case.
    - Empty predicted ROIs are excluded from predicted-region metric
      comparisons, and their counts are reported.
    - Ground-truth-oracle explanations are offline analyses only.
    - Bootstrap confidence intervals are case-level percentile CIs.
    - P-values use paired permutation/sign-flip tests.
    - Holm correction is applied within each comparison family.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_CSV = (
    PROJECT_ROOT
    / "report"
    / "segmentation_xai_spatial"
    / "stratified_xai_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "report"
    / "segmentation_xai_spatial"
    / "statistical_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 10000
N_PERMUTATIONS = 20000
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# 2. METRIC COLUMN CONFIGURATION
# ============================================================

# These are the metric column names used by the spatial XAI audit.
# If your CSV has different names, the script will report the
# available columns and stop rather than silently using the wrong data.

METRICS = {
    "Grad-CAM": "gradcam_rank_auc",
    "IG absolute": "ig_abs_rank_auc",
    "IG positive": "ig_positive_rank_auc",
    "IG negative magnitude": "ig_negative_rank_auc",
    "Occlusion absolute": "occlusion_abs_rank_auc",
    "Occlusion positive": "occlusion_positive_rank_auc",
}

TARGET_COLUMN = "target"
CASE_COLUMN = "case"
DICE_COLUMN = "dice"

PREDICTED_TARGET = "predicted_region"
ORACLE_TARGET = "ground_truth_oracle_OFFLINE"


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def find_column(df, expected, alternatives=()):
    """Find a column using exact or case-insensitive matching."""

    candidates = [expected] + list(alternatives)

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    lower_map = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def holm_adjust(p_values):
    """Holm-Bonferroni adjustment for multiple comparisons."""

    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)

    valid_indices = np.where(np.isfinite(p_values))[0]

    if len(valid_indices) == 0:
        return adjusted

    valid_p = p_values[valid_indices]
    order = np.argsort(valid_p)

    sorted_p = valid_p[order]
    m = len(sorted_p)

    sorted_adjusted = np.maximum.accumulate(
        [(m - i) * p for i, p in enumerate(sorted_p)]
    )

    sorted_adjusted = np.clip(sorted_adjusted, 0, 1)

    restored = np.empty(m)
    restored[order] = sorted_adjusted

    adjusted[valid_indices] = restored

    return adjusted


def bootstrap_mean_ci(values, n_bootstrap=N_BOOTSTRAP):
    """
    Case-level percentile bootstrap CI for a mean.

    Each element of values represents one independent case.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    observed_mean = float(np.mean(values))

    if len(values) == 1:
        return observed_mean, np.nan, np.nan

    samples = rng.choice(
        values,
        size=(n_bootstrap, len(values)),
        replace=True,
    )

    bootstrap_means = np.mean(samples, axis=1)

    alpha = 1 - CONFIDENCE_LEVEL

    lower = np.quantile(bootstrap_means, alpha / 2)
    upper = np.quantile(bootstrap_means, 1 - alpha / 2)

    return observed_mean, float(lower), float(upper)


def paired_bootstrap_difference_ci(
    first,
    second,
    n_bootstrap=N_BOOTSTRAP,
):
    """
    Paired bootstrap CI for mean(first - second).

    Resampling is performed over paired cases.
    """

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    valid = np.isfinite(first) & np.isfinite(second)

    first = first[valid]
    second = second[valid]

    if len(first) == 0:
        return np.nan, np.nan, np.nan

    differences = first - second
    observed = float(np.mean(differences))

    if len(differences) == 1:
        return observed, np.nan, np.nan

    samples = rng.choice(
        differences,
        size=(n_bootstrap, len(differences)),
        replace=True,
    )

    bootstrap_means = np.mean(samples, axis=1)

    alpha = 1 - CONFIDENCE_LEVEL

    lower = np.quantile(bootstrap_means, alpha / 2)
    upper = np.quantile(bootstrap_means, 1 - alpha / 2)

    return observed, float(lower), float(upper)


def paired_permutation_test(
    first,
    second,
    n_permutations=N_PERMUTATIONS,
):
    """
    Two-sided paired sign-flip permutation test.

    Tests whether the mean paired difference is zero.
    """

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    valid = np.isfinite(first) & np.isfinite(second)

    differences = first[valid] - second[valid]

    n = len(differences)

    if n == 0:
        return np.nan

    observed = abs(np.mean(differences))

    if n == 1:
        return 1.0

    if n <= 15:
        # Exact enumeration of all sign combinations.
        number = 2 ** n

        signs = np.arange(number)[:, None]
        bits = (
            (signs >> np.arange(n)) & 1
        )

        sign_matrix = 2 * bits - 1

        permuted = np.abs(
            np.mean(
                sign_matrix * differences,
                axis=1,
            )
        )

        return float(
            np.mean(permuted >= observed - 1e-12)
        )

    # Monte Carlo sign-flip permutation.
    signs = rng.choice(
        [-1, 1],
        size=(n_permutations, n),
    )

    permuted = np.abs(
        np.mean(
            signs * differences,
            axis=1,
        )
    )

    extreme = np.sum(
        permuted >= observed - 1e-12
    )

    # Plus-one correction prevents zero p-values.
    return float(
        (extreme + 1) / (n_permutations + 1)
    )


def cliffs_delta_paired(differences):
    """
    Simple paired effect-size estimate.

    Reports the proportion of positive minus negative
    paired differences. Zero differences are omitted.
    """

    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]

    differences = differences[differences != 0]

    if len(differences) == 0:
        return np.nan

    return float(
        (np.sum(differences > 0) - np.sum(differences < 0))
        / len(differences)
    )


# ============================================================
# 4. LOAD AND VALIDATE DATA
# ============================================================

print("\n" + "=" * 65)
print("STRATIFIED SEGMENTATION XAI STATISTICAL ANALYSIS")
print("=" * 65)

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"\nInput CSV not found:\n{INPUT_CSV}\n\n"
        "Make sure stratified_xai_metrics.csv exists."
    )

df = pd.read_csv(INPUT_CSV)

print(f"\nInput: {INPUT_CSV}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

case_col = find_column(
    df,
    CASE_COLUMN,
    alternatives=("case_id", "image", "filename"),
)

target_col = find_column(
    df,
    TARGET_COLUMN,
    alternatives=("target_type", "explanation_target"),
)

dice_col = find_column(
    df,
    DICE_COLUMN,
    alternatives=("pt_dice", "segmentation_dice", "dice_score"),
)

if case_col is None or target_col is None or dice_col is None:
    print("\nCould not identify required columns.")
    print("\nAvailable columns:")
    print(df.columns.tolist())

    raise ValueError(
        "Please check case, target, and Dice column names."
    )

resolved_metrics = {}

for metric_name, expected_column in METRICS.items():

    actual_column = find_column(
        df,
        expected_column,
    )

    if actual_column is not None:
        resolved_metrics[metric_name] = actual_column

if len(resolved_metrics) == 0:
    print("\nNo configured metric columns were found.")
    print("\nAvailable columns:")
    print(df.columns.tolist())

    raise ValueError(
        "Update the METRICS dictionary to match your CSV."
    )

print("\nResolved columns:")
print(f"Case: {case_col}")
print(f"Target: {target_col}")
print(f"Dice: {dice_col}")

for name, column in resolved_metrics.items():
    print(f"{name}: {column}")


# ============================================================
# 5. NORMALISE TARGET LABELS AND CREATE CASE-LEVEL DATA
# ============================================================

df[target_col] = (
    df[target_col]
    .astype(str)
    .str.strip()
)

df[case_col] = (
    df[case_col]
    .astype(str)
    .str.strip()
)

df[dice_col] = pd.to_numeric(
    df[dice_col],
    errors="coerce",
)

for column in resolved_metrics.values():
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

# Retain only recognised target rows.
valid_targets = [
    PREDICTED_TARGET,
    ORACLE_TARGET,
]

unknown_targets = sorted(
    set(df[target_col].dropna()) - set(valid_targets)
)

if unknown_targets:
    print("\nWarning: unrecognised target labels:")
    print(unknown_targets)

df = df[
    df[target_col].isin(valid_targets)
].copy()

# One Dice value per case. Dice is repeated across target rows.
case_dice_counts = (
    df.groupby(case_col)[dice_col]
    .nunique(dropna=True)
)

if (case_dice_counts > 1).any():
    raise ValueError(
        "Some cases have inconsistent Dice values across target rows."
    )

case_dice = (
    df.groupby(case_col)[dice_col]
    .first()
    .rename("dice")
)

# Filename-derived class grouping.
def infer_class(case_name):
    name = str(case_name).lower()

    if "_gl_" in name or "glioma" in name:
        return "glioma"

    if "_me_" in name or "meningioma" in name:
        return "meningioma"

    if "_pi_" in name or "pituitary" in name:
        return "pituitary"

    return "unknown"


case_metadata = pd.DataFrame({
    "case": case_dice.index,
    "dice": case_dice.values,
})

case_metadata["inferred_class"] = (
    case_metadata["case"].map(infer_class)
)

case_metadata["dice_group"] = pd.cut(
    case_metadata["dice"],
    bins=[-np.inf, 0, 0.50, 0.85, 1.000001],
    labels=["zero", "low", "medium", "high"],
    right=True,
)

# Ensure exact zero is grouped as zero.
case_metadata.loc[
    case_metadata["dice"] == 0,
    "dice_group",
] = "zero"

print("\nCase-level dataset:")
print(f"Unique cases: {case_metadata['case'].nunique()}")
print(f"Mean Dice: {case_metadata['dice'].mean():.4f}")
print(f"Median Dice: {case_metadata['dice'].median():.4f}")

print("\nDice-group counts:")
print(
    case_metadata["dice_group"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nTarget-row counts:")
print(
    df[target_col]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# 6. DESCRIPTIVE STATISTICS
# ============================================================

descriptive_rows = []

for target in valid_targets:

    subset = df[
        df[target_col] == target
    ]

    for metric_name, metric_column in resolved_metrics.items():

        values = subset[metric_column].dropna().to_numpy()

        if len(values) == 0:
            continue

        mean, ci_low, ci_high = bootstrap_mean_ci(values)

        descriptive_rows.append({
            "target": target,
            "metric": metric_name,
            "n_cases": len(values),
            "mean": mean,
            "std": float(np.std(values, ddof=1))
            if len(values) > 1 else np.nan,
            "median": float(np.median(values)),
            "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "bootstrap_ci_lower": ci_low,
            "bootstrap_ci_upper": ci_high,
        })

descriptive_df = pd.DataFrame(descriptive_rows)

descriptive_path = OUTPUT_DIR / "descriptive_statistics.csv"

descriptive_df.to_csv(
    descriptive_path,
    index=False,
)

print(f"\nSaved: {descriptive_path}")


# ============================================================
# 7. PAIRED BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

# Pivot to one row per case, preserving pairing.
metric_long = df[
    [case_col, target_col] + list(resolved_metrics.values())
].copy()

paired = metric_long.pivot(
    index=case_col,
    columns=target_col,
    values=list(resolved_metrics.values()),
)

ci_rows = []

for metric_name, metric_column in resolved_metrics.items():

    for target in valid_targets:

        if (metric_column, target) not in paired.columns:
            continue

        values = paired[(metric_column, target)].to_numpy()

        mean, lower, upper = bootstrap_mean_ci(values)

        ci_rows.append({
            "comparison": "target_mean",
            "metric": metric_name,
            "target": target,
            "n_cases": int(np.isfinite(values).sum()),
            "mean": mean,
            "ci_lower": lower,
            "ci_upper": upper,
        })

    if (
        (metric_column, PREDICTED_TARGET) in paired.columns
        and (metric_column, ORACLE_TARGET) in paired.columns
    ):

        predicted = paired[
            (metric_column, PREDICTED_TARGET)
        ].to_numpy()

        oracle = paired[
            (metric_column, ORACLE_TARGET)
        ].to_numpy()

        difference, lower, upper = paired_bootstrap_difference_ci(
            predicted,
            oracle,
        )

        ci_rows.append({
            "comparison": "predicted_minus_oracle",
            "metric": metric_name,
            "target": "paired_difference",
            "n_cases": int(
                (
                    np.isfinite(predicted)
                    & np.isfinite(oracle)
                ).sum()
            ),
            "mean": difference,
            "ci_lower": lower,
            "ci_upper": upper,
        })

ci_df = pd.DataFrame(ci_rows)

ci_path = OUTPUT_DIR / "bootstrap_confidence_intervals.csv"

ci_df.to_csv(
    ci_path,
    index=False,
)

print(f"Saved: {ci_path}")


# ============================================================
# 8. PAIRED METHOD COMPARISONS
# ============================================================

# Comparisons are performed separately for predicted and oracle
# targets. The same cases are paired across XAI methods.

method_comparisons = [
    ("Occlusion absolute", "Grad-CAM"),
    ("IG absolute", "Grad-CAM"),
    ("Occlusion absolute", "IG absolute"),
    ("IG positive", "Grad-CAM"),
    ("Occlusion positive", "Grad-CAM"),
]

comparison_rows = []

for target in valid_targets:

    target_df = df[
        df[target_col] == target
    ]

    # One row per case for the selected target.
    target_df = target_df.drop_duplicates(
        subset=[case_col]
    ).set_index(case_col)

    for first_name, second_name in method_comparisons:

        if (
            first_name not in resolved_metrics
            or second_name not in resolved_metrics
        ):
            continue

        first_column = resolved_metrics[first_name]
        second_column = resolved_metrics[second_name]

        first = target_df[first_column].to_numpy()
        second = target_df[second_column].to_numpy()

        valid = np.isfinite(first) & np.isfinite(second)

        first_valid = first[valid]
        second_valid = second[valid]

        if len(first_valid) == 0:
            continue

        difference = first_valid - second_valid

        mean_difference, ci_low, ci_high = (
            paired_bootstrap_difference_ci(
                first_valid,
                second_valid,
            )
        )

        p_value = paired_permutation_test(
            first_valid,
            second_valid,
        )

        comparison_rows.append({
            "target": target,
            "method_1": first_name,
            "method_2": second_name,
            "difference_definition": (
                f"{first_name} minus {second_name}"
            ),
            "n_paired_cases": len(first_valid),
            "mean_method_1": float(np.mean(first_valid)),
            "mean_method_2": float(np.mean(second_valid)),
            "mean_paired_difference": mean_difference,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "paired_effect_size": cliffs_delta_paired(difference),
            "p_value": p_value,
        })

comparison_df = pd.DataFrame(comparison_rows)

if not comparison_df.empty:

    comparison_df["holm_adjusted_p"] = np.nan

    for target in comparison_df["target"].unique():

        mask = comparison_df["target"] == target

        comparison_df.loc[
            mask,
            "holm_adjusted_p",
        ] = holm_adjust(
            comparison_df.loc[mask, "p_value"].to_numpy()
        )

    comparison_df["significant_holm_0_05"] = (
        comparison_df["holm_adjusted_p"] < 0.05
    )

comparison_path = OUTPUT_DIR / "paired_method_comparisons.csv"

comparison_df.to_csv(
    comparison_path,
    index=False,
)

print(f"Saved: {comparison_path}")


# ============================================================
# 9. PREDICTED-REGION VS ORACLE COMPARISONS
# ============================================================

target_comparison_rows = []

for metric_name, metric_column in resolved_metrics.items():

    if (
        (metric_column, PREDICTED_TARGET) not in paired.columns
        or (metric_column, ORACLE_TARGET) not in paired.columns
    ):
        continue

    predicted = paired[
        (metric_column, PREDICTED_TARGET)
    ].to_numpy()

    oracle = paired[
        (metric_column, ORACLE_TARGET)
    ].to_numpy()

    valid = np.isfinite(predicted) & np.isfinite(oracle)

    predicted_valid = predicted[valid]
    oracle_valid = oracle[valid]

    if len(predicted_valid) == 0:
        continue

    difference = predicted_valid - oracle_valid

    mean_diff, lower, upper = paired_bootstrap_difference_ci(
        predicted_valid,
        oracle_valid,
    )

    p_value = paired_permutation_test(
        predicted_valid,
        oracle_valid,
    )

    target_comparison_rows.append({
        "metric": metric_name,
        "n_paired_cases": len(predicted_valid),
        "predicted_mean": float(np.mean(predicted_valid)),
        "oracle_mean": float(np.mean(oracle_valid)),
        "predicted_minus_oracle": mean_diff,
        "ci_lower": lower,
        "ci_upper": upper,
        "paired_effect_size": cliffs_delta_paired(difference),
        "p_value": p_value,
    })

target_comparison_df = pd.DataFrame(
    target_comparison_rows
)

if not target_comparison_df.empty:

    target_comparison_df["holm_adjusted_p"] = holm_adjust(
        target_comparison_df["p_value"].to_numpy()
    )

    target_comparison_df["significant_holm_0_05"] = (
        target_comparison_df["holm_adjusted_p"] < 0.05
    )

target_comparison_path = (
    OUTPUT_DIR / "predicted_vs_oracle_comparisons.csv"
)

target_comparison_df.to_csv(
    target_comparison_path,
    index=False,
)

print(f"Saved: {target_comparison_path}")


# ============================================================
# 10. DICE-GROUP COMPARISONS
# ============================================================

# Descriptive AUC summaries within Dice groups.
# These are exploratory subgroup analyses, not independent
# full-dataset performance estimates.

group_rows = []

for target in valid_targets:

    subset = df[
        df[target_col] == target
    ].copy()

    subset = subset.merge(
        case_metadata[[ "case", "dice_group", "inferred_class" ]],
        left_on=case_col,
        right_on="case",
        how="left",
    )

    for group in ["zero", "low", "medium", "high"]:

        group_df = subset[
            subset["dice_group"].astype(str) == group
        ]

        for metric_name, metric_column in resolved_metrics.items():

            values = group_df[metric_column].dropna().to_numpy()

            if len(values) == 0:
                continue

            group_rows.append({
                "target": target,
                "dice_group": group,
                "metric": metric_name,
                "n_cases": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1))
                if len(values) > 1 else np.nan,
                "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
            })

group_df = pd.DataFrame(group_rows)

group_path = OUTPUT_DIR / "dice_group_comparisons.csv"

group_df.to_csv(
    group_path,
    index=False,
)

print(f"Saved: {group_path}")


# ============================================================
# 11. EXPLORATORY DICE CORRELATIONS
# ============================================================

correlation_rows = []

for target in valid_targets:

    subset = df[
        df[target_col] == target
    ].drop_duplicates(
        subset=[case_col]
    )

    for metric_name, metric_column in resolved_metrics.items():

        valid = subset[
            [dice_col, metric_column]
        ].dropna()

        if len(valid) < 3:
            continue

        rho, p_value = spearmanr(
            valid[dice_col],
            valid[metric_column],
        )

        correlation_rows.append({
            "target": target,
            "metric": metric_name,
            "n_cases": len(valid),
            "spearman_rho": float(rho),
            "p_value_unadjusted": float(p_value),
        })

correlation_df = pd.DataFrame(correlation_rows)

if not correlation_df.empty:

    correlation_df["holm_adjusted_p"] = np.nan

    for target in correlation_df["target"].unique():

        mask = correlation_df["target"] == target

        correlation_df.loc[
            mask,
            "holm_adjusted_p",
        ] = holm_adjust(
            correlation_df.loc[
                mask,
                "p_value_unadjusted",
            ].to_numpy()
        )

correlation_path = OUTPUT_DIR / "dice_auc_correlations.csv"

correlation_df.to_csv(
    correlation_path,
    index=False,
)

print(f"Saved: {correlation_path}")


# ============================================================
# 12. CLASS-LEVEL DESCRIPTIVE SUMMARIES
# ============================================================

class_rows = []

for target in valid_targets:

    subset = df[
        df[target_col] == target
    ].drop_duplicates(
        subset=[case_col]
    )

    subset = subset.merge(
        case_metadata[["case", "inferred_class"]],
        left_on=case_col,
        right_on="case",
        how="left",
    )

    for class_name in [
        "glioma",
        "meningioma",
        "pituitary",
        "unknown",
    ]:

        class_df = subset[
            subset["inferred_class"] == class_name
        ]

        for metric_name, metric_column in resolved_metrics.items():

            values = class_df[metric_column].dropna().to_numpy()

            if len(values) == 0:
                continue

            class_rows.append({
                "target": target,
                "inferred_class": class_name,
                "metric": metric_name,
                "n_cases": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1))
                if len(values) > 1 else np.nan,
                "median": float(np.median(values)),
            })

class_df = pd.DataFrame(class_rows)

class_path = OUTPUT_DIR / "class_level_descriptive_statistics.csv"

class_df.to_csv(
    class_path,
    index=False,
)

print(f"Saved: {class_path}")


# ============================================================
# 13. EMPTY ROI AND DATA COMPLETENESS AUDIT
# ============================================================

status_col = find_column(
    df,
    "status",
    alternatives=("explanation_status",),
)

if status_col is not None:

    status_counts = (
        df[status_col]
        .fillna("missing")
        .value_counts()
        .to_dict()
    )

else:
    status_counts = {
        "status_column_not_available": len(df)
    }

target_case_counts = (
    df.groupby(target_col)[case_col]
    .nunique()
    .to_dict()
)

missing_metric_counts = {
    name: int(df[column].isna().sum())
    for name, column in resolved_metrics.items()
}

# Cases with missing predicted rows or oracle rows.
target_presence = (
    df.groupby(case_col)[target_col]
    .apply(set)
)

missing_predicted = [
    case for case, targets in target_presence.items()
    if PREDICTED_TARGET not in targets
]

missing_oracle = [
    case for case, targets in target_presence.items()
    if ORACLE_TARGET not in targets
]


# ============================================================
# 14. WRITE HUMAN-READABLE SUMMARY
# ============================================================

summary_path = OUTPUT_DIR / "statistical_summary.txt"

with open(summary_path, "w", encoding="utf-8") as f:

    f.write("STRATIFIED SEGMENTATION XAI STATISTICAL ANALYSIS\n")
    f.write("=" * 65 + "\n\n")

    f.write(f"Input CSV: {INPUT_CSV}\n")
    f.write(f"Total input rows: {len(df)}\n")
    f.write(
        f"Unique cases: {df[case_col].nunique()}\n\n"
    )

    f.write("CASE-LEVEL DICE SUMMARY\n")
    f.write("-" * 65 + "\n")

    f.write(
        f"Mean Dice: {case_metadata['dice'].mean():.6f}\n"
    )

    f.write(
        f"Median Dice: {case_metadata['dice'].median():.6f}\n"
    )

    f.write(
        f"SD Dice: {case_metadata['dice'].std():.6f}\n\n"
    )

    f.write("TARGET COUNTS\n")
    f.write("-" * 65 + "\n")

    for target, count in target_case_counts.items():
        f.write(f"{target}: {count} cases\n")

    f.write("\nDICE GROUP COUNTS\n")
    f.write("-" * 65 + "\n")

    for group, count in (
        case_metadata["dice_group"]
        .value_counts(dropna=False)
        .items()
    ):
        f.write(f"{group}: {count}\n")

    f.write("\nDATA COMPLETENESS\n")
    f.write("-" * 65 + "\n")

    f.write(f"Status counts: {status_counts}\n")
    f.write(f"Cases missing predicted row: {len(missing_predicted)}\n")
    f.write(f"Cases missing oracle row: {len(missing_oracle)}\n")

    if missing_predicted:
        f.write(
            "Missing predicted cases: "
            + ", ".join(missing_predicted)
            + "\n"
        )

    if missing_oracle:
        f.write(
            "Missing oracle cases: "
            + ", ".join(missing_oracle)
            + "\n"
        )

    f.write("\nMissing metric values:\n")

    for metric, count in missing_metric_counts.items():
        f.write(f"{metric}: {count}\n")

    f.write("\nDESCRIPTIVE XAI RESULTS\n")
    f.write("-" * 65 + "\n")

    for _, row in descriptive_df.iterrows():

        f.write(
            f"{row['target']} | {row['metric']} | "
            f"n={row['n_cases']} | "
            f"mean={row['mean']:.4f} | "
            f"95% CI=[{row['bootstrap_ci_lower']:.4f}, "
            f"{row['bootstrap_ci_upper']:.4f}]\n"
        )

    f.write("\nPAIRED METHOD COMPARISONS\n")
    f.write("-" * 65 + "\n")

    if comparison_df.empty:
        f.write("No paired method comparisons available.\n")

    else:
        for _, row in comparison_df.iterrows():

            f.write(
                f"{row['target']} | "
                f"{row['method_1']} minus {row['method_2']} | "
                f"n={row['n_paired_cases']} | "
                f"mean difference={row['mean_paired_difference']:.4f} | "
                f"95% CI=[{row['ci_lower']:.4f}, "
                f"{row['ci_upper']:.4f}] | "
                f"Holm p={row['holm_adjusted_p']:.5f}\n"
            )

    f.write("\nPREDICTED VS ORACLE COMPARISONS\n")
    f.write("-" * 65 + "\n")

    if target_comparison_df.empty:
        f.write("No paired target comparisons available.\n")

    else:
        for _, row in target_comparison_df.iterrows():

            f.write(
                f"{row['metric']} | "
                f"predicted minus oracle="
                f"{row['predicted_minus_oracle']:.4f} | "
                f"95% CI=[{row['ci_lower']:.4f}, "
                f"{row['ci_upper']:.4f}] | "
                f"Holm p={row['holm_adjusted_p']:.5f}\n"
            )

    f.write("\nINTERPRETATION AND LIMITATIONS\n")
    f.write("-" * 65 + "\n")

    f.write(
        "1. The independent sampling unit is the case. "
        "Repeated target rows are not treated as independent cases.\n"
    )

    f.write(
        "2. Predicted-region and ground-truth-oracle comparisons "
        "are paired by case.\n"
    )

    f.write(
        "3. Oracle explanations use ground-truth regions and are "
        "offline analyses, not deployment-time explanations.\n"
    )

    f.write(
        "4. High spatial ranking AUC does not establish explanation "
        "faithfulness, model correctness, or clinical validity.\n"
    )

    f.write(
        "5. Absolute attribution scores combine positive and negative "
        "contributions. They do not imply that all attribution "
        "supports tumour prediction.\n"
    )

    f.write(
        "6. The 39 cases are a stratified selected sample, not a "
        "replacement for the full 860-image segmentation evaluation.\n"
    )

    f.write(
        "7. Class labels are inferred from filenames and should be "
        "verified against dataset metadata before publication.\n"
    )

    f.write(
        "8. Bootstrap intervals and permutation tests quantify "
        "uncertainty within this selected sample; external validation "
        "is still required.\n"
    )

print(f"Saved: {summary_path}")


# ============================================================
# 15. FINAL CONSOLE REPORT
# ============================================================

print("\n" + "=" * 65)
print("STATISTICAL ANALYSIS COMPLETED")
print("=" * 65)

print(f"\nUnique cases analysed: {case_metadata['case'].nunique()}")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for path in sorted(OUTPUT_DIR.glob("*")):
    if path.is_file():
        print(f"  {path.name}")

print("\nImportant:")
print("- Interpret p-values together with effect sizes and CIs.")
print("- A non-significant result does not prove equivalence.")
print("- This analysis does not establish XAI faithfulness.")
print("- Filename-inferred class groups require metadata verification.")

print("\nDone.")
