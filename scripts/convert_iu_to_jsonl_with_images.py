# scripts/convert_iu_to_jsonl_with_images.py

import pandas as pd
import json
from pathlib import Path
import os

# ==== 1. SET THESE PATHS FOR YOUR ENVIRONMENT =====================
# If you are on COLAB and downloaded via Kaggle API, something like:
# REPORTS_CSV = "/content/iu_xray/indiana_reports.csv"
# PROJ_CSV    = "/content/iu_xray/indiana_projections.csv"
# IMAGE_ROOT  = "/content/iu_xray/images"   # folder that has the PNGs

# If you are on KAGGLE:
# REPORTS_CSV = "/kaggle/input/chest-xrays-indiana-university/indiana_reports.csv"
# PROJ_CSV    = "/kaggle/input/chest-xrays-indiana-university/indiana_projections.csv"
# IMAGE_ROOT  = "/kaggle/input/chest-xrays-indiana-university/images"  # check name with !ls

REPORTS_CSV = "/kaggle/input/chest-xrays-indiana-university/indiana_reports.csv"
PROJ_CSV    = "/kaggle/input/chest-xrays-indiana-university/indiana_projections.csv"
IMAGE_ROOT  = "/kaggle/input/chest-xrays-indiana-university/images"

# Where to write the JSONL (inside your project)
OUT_PATH = Path("data/iu_reports_with_images.jsonl")
# ==================================================================


def main():
    print("Loading CSVs...")
    reports_df = pd.read_csv(REPORTS_CSV)
    proj_df = pd.read_csv(PROJ_CSV)

    print("Reports columns:", reports_df.columns.tolist())
    print("Projections columns:", proj_df.columns.tolist())

    # 1) Keep only frontal images (most works in literature do this)
    frontal_df = proj_df[proj_df["projection"] == "Frontal"].copy()

    # If there are multiple frontal images per uid, keep the first
    frontal_df = frontal_df.sort_values(["uid", "filename"])
    frontal_df = frontal_df.drop_duplicates(subset=["uid"], keep="first")

    # 2) Merge with reports on uid
    merged = pd.merge(
        reports_df,
        frontal_df[["uid", "filename"]],
        on="uid",
        how="inner",
    )

    print("Merged rows:", len(merged))

    # 3) Build JSONL with image path + FINDINGS text
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    num_written = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for _, row in merged.iterrows():
            findings = str(row.get("findings", "") or "").strip()
            impression = str(row.get("impression", "") or "").strip()

            reference = findings if findings else impression
            if len(reference) < 10:
                continue

            # image file on disk
            filename = row["filename"]
            # store relative path from IMAGE_ROOT
            # (or full path if you prefer)
            image_path = os.path.join(IMAGE_ROOT, filename)

            sample = {
                "image_path": image_path,
                "prompt": "FINDINGS: ",
                "reference": reference,
            }
            f.write(json.dumps(sample) + "\n")
            num_written += 1

    print(f"Wrote {num_written} samples to {OUT_PATH}")


if __name__ == "__main__":
    main()
