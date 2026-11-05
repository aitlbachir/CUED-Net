"""Radiologist annotations per lesion, from the CBIS-DDSM mass tables. Used for descriptive checks only."""

import argparse

import pandas as pd

FIELDS = {"subtlety": "subtlety", "assessment": "assessment", "mass_shape": "shape", "mass_margins": "margins"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.concat(pd.read_csv(p) for p in (args.train_csv, args.test_csv))
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df["key"] = df.patient_id + "_" + df.left_or_right_breast + "_" + df.abnormality_id.astype(int).astype(str)
    df = df.drop_duplicates(["key", "image_view"])
    wide = df.pivot(index="key", columns="image_view", values=list(FIELDS))
    wide.columns = [f"{FIELDS[a]}_{b}" for a, b in wide.columns]
    wide["density"] = df.groupby("key").breast_density.first()
    wide.reset_index().to_csv(args.out, index=False)
    print(len(wide), "lesions")


if __name__ == "__main__":
    main()
