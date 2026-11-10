from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

METRICS_CSV = ROOT / "report" / "segmentation_full_testset" / "per_image_metrics.csv"
OUT_DIR = ROOT / "report" / "segmentation_xai_spatial"

SAMPLES_PER_STRATUM = 4
RANDOM_SEED = 2026

def dice_band(dice):
    if dice <= 1e-12:
        return "zero"
    if dice <= 0.50:
        return "low_0_to_050"
    if dice <= 0.85:
        return "medium_050_to_085"
    return "high_085_to_1"

def main():
    if not METRICS_CSV.is_file():
        raise FileNotFoundError(
            f"Existing 860-image evaluation CSV not found:\n{METRICS_CSV}\n"
            "Do not rerun the evaluation automatically. Check the report folder."
        )

    df = pd.read_csv(METRICS_CSV)

    required = {"image", "class", "pt_dice"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Evaluation CSV is missing columns: {sorted(missing)}")

    df["pt_dice"] = pd.to_numeric(df["pt_dice"], errors="coerce")
    df = df.dropna(subset=["image", "class", "pt_dice"]).copy()
    df = df[df["class"].isin(["glioma", "meningioma", "pituitary"])].copy()

    if df.empty:
        raise RuntimeError("No eligible tumour cases found in the evaluation CSV.")

    df["case"] = df["image"].map(lambda x: Path(str(x)).stem)
    df["dice_band"] = df["pt_dice"].map(dice_band)

    # One deterministic sample from each class x Dice-performance stratum.
    selected = []
    for cls in ["glioma", "meningioma", "pituitary"]:
        for band in ["zero", "low_0_to_050", "medium_050_to_085", "high_085_to_1"]:
            group = df[(df["class"] == cls) & (df["dice_band"] == band)]
            if group.empty:
                continue

            take = min(SAMPLES_PER_STRATUM, len(group))
            sample = group.sample(n=take, random_state=RANDOM_SEED)
            selected.append(sample)

    result = pd.concat(selected, ignore_index=True)
    result = result.drop_duplicates(subset=["case"])
    result = result.sort_values(["class", "dice_band", "pt_dice", "case"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    output_csv = OUT_DIR / "stratified_xai_cases.csv"
    output_txt = OUT_DIR / "stratified_xai_cases.txt"
    summary_csv = OUT_DIR / "stratified_xai_sampling_summary.csv"

    result[["case", "image", "class", "pt_dice", "dice_band"]].to_csv(
        output_csv, index=False
    )
    output_txt.write_text(
        "\n".join(result["case"].tolist()) + "\n",
        encoding="utf-8"
    )

    summary = (
        result.groupby(["class", "dice_band"], observed=True)
        .agg(
            selected_cases=("case", "count"),
            mean_dice=("pt_dice", "mean"),
            min_dice=("pt_dice", "min"),
            max_dice=("pt_dice", "max"),
        )
        .reset_index()
    )
    summary.to_csv(summary_csv, index=False)

    print("\nSTRATIFIED SEGMENTATION XAI SAMPLE")
    print(f"Source evaluation: {METRICS_CSV}")
    print(f"Eligible tumour images: {len(df)}")
    print(f"Selected cases: {len(result)}")
    print("\nSelection by class and Dice band:")
    print(summary.to_string(index=False))
    print(f"\nCase list: {output_txt}")
    print(f"Case details: {output_csv}")
    print(f"Sampling summary: {summary_csv}")
    print("\nThe original 860-image evaluation was not rerun or modified.")

if __name__ == "__main__":
    main()