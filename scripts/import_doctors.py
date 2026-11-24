"""Validate and import a REAL doctor directory.

Usage:  python scripts/import_doctors.py my_directory.csv

Required columns: name, specialty, city, hospital, experience_years, rating, reviews, fee_inr, languages
                  (';'-separated), teleconsult (true/false), registration_no, source
`source` must say where each row came from (e.g. "NMC Indian Medical Register", "hospital website"); rows without a
source or registration number are rejected. Only import data you are allowed to use - do not scrape sites whose
terms forbid it. Output: data/doctors.csv (used automatically instead of the synthetic sample).
"""
import sys

import pandas as pd

from src.config import ROOT
from src.knowledge_base import SPECIALTIES

REQUIRED = ["name", "specialty", "city", "hospital", "experience_years", "rating", "reviews", "fee_inr",
            "languages", "teleconsult", "registration_no", "source"]


def main(path: str):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns: {missing}")
    canon = {s.lower(): s for s in SPECIALTIES}
    df["specialty"] = df["specialty"].astype(str).str.strip().str.lower().map(canon)
    n0 = len(df)
    df = df.dropna(subset=["specialty", "name", "city", "registration_no", "source"])
    df = df.drop_duplicates(subset="registration_no")
    for c, lo, hi in [("rating", 0, 5), ("experience_years", 0, 70), ("fee_inr", 0, 100000)]:
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(lo, hi)
    df["rating"] = df["rating"].fillna(df["rating"].median())
    df["reviews"] = pd.to_numeric(df["reviews"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["experience_years", "fee_inr"])
    df.insert(0, "doctor_id", [f"D{i:05d}" for i in range(1, len(df) + 1)])
    df["is_sample"] = False
    out = ROOT / "data" / "doctors.csv"
    df.to_csv(out, index=False)
    print(f"Imported {len(df)} of {n0} rows -> {out}")
    unknown = sorted(set(df["city"]) - set(__import__("src.doctors", fromlist=["x"]).CITY_COORDS))
    if unknown:
        print("Note: add coordinates to CITY_COORDS in src/doctors.py for:", unknown, "(distance scoring falls back to a neutral value)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__))
