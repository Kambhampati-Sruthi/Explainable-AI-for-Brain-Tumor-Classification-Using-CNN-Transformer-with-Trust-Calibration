from pathlib import Path
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent

CASE_LIST = ROOT / "report" / "segmentation_xai_spatial" / "stratified_xai_cases.txt"
AUDIT_SCRIPT = ROOT / "audit_segmentation_xai_spatial.py"
AUDIT_CSV = ROOT / "report" / "segmentation_xai_spatial" / "segmentation_xai_spatial_metrics.csv"

OUT_DIR = ROOT / "report" / "segmentation_xai_spatial"
SNAPSHOT_DIR = OUT_DIR / "stratified_per_case_csv"

def main():
    for path in [CASE_LIST, AUDIT_SCRIPT]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    cases = [
        line.strip()
        for line in CASE_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not cases:
        raise RuntimeError("The selected case list is empty.")

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    collected = []
    failed = []

    print("=" * 72)
    print("STRATIFIED SEGMENTATION XAI RUN")
    print(f"Cases selected: {len(cases)}")
    print("Using the existing segmentation XAI audit script.")
    print("No training, threshold tuning, or checkpoint modification.")
    print("=" * 72)

    for index, case in enumerate(cases, start=1):
        print(f"\n[{index}/{len(cases)}] Running {case}")

        try:
            subprocess.run(
                [
                    sys.executable,
                    str(AUDIT_SCRIPT),
                    "--case",
                    case,
                ],
                cwd=ROOT,
                check=True,
            )

            if not AUDIT_CSV.is_file():
                raise FileNotFoundError(
                    f"Expected audit output was not created: {AUDIT_CSV}"
                )

            current = pd.read_csv(AUDIT_CSV)

            if "case" not in current.columns:
                raise ValueError("Audit CSV does not contain the expected 'case' column.")

            current = current[current["case"].astype(str) == case].copy()

            if current.empty:
                raise RuntimeError(
                    f"No result row for {case} was found in the audit CSV."
                )

            # Preserve a separate copy before the next single-case run overwrites it.
            snapshot = SNAPSHOT_DIR / f"{case}_metrics.csv"
            current.to_csv(snapshot, index=False)

            collected.append(current)

            combined = pd.concat(collected, ignore_index=True)
            combined.to_csv(
                OUT_DIR / "stratified_xai_metrics_partial.csv",
                index=False,
            )

            print(f"Saved case metrics: {snapshot.name}")

        except Exception as exc:
            failed.append({"case": case, "error": repr(exc)})
            print(f"FAILED: {case}: {exc}")

    if collected:
        combined = pd.concat(collected, ignore_index=True)
        combined.to_csv(
            OUT_DIR / "stratified_xai_metrics.csv",
            index=False,
        )

    if failed:
        pd.DataFrame(failed).to_csv(
            OUT_DIR / "stratified_xai_failed_cases.csv",
            index=False,
        )

    print("\n" + "=" * 72)
    print("STRATIFIED SEGMENTATION XAI FINISHED")
    print(f"Selected: {len(cases)}")
    print(f"Completed: {len(collected)}")
    print(f"Failed: {len(failed)}")
    if collected:
        print(f"Combined metrics: {OUT_DIR / 'stratified_xai_metrics.csv'}")
    if failed:
        print(f"Failure list: {OUT_DIR / 'stratified_xai_failed_cases.csv'}")
    print(f"Individual CSV snapshots: {SNAPSHOT_DIR}")
    print("=" * 72)

if __name__ == "__main__":
    main()