"""Doctor finder: specialty match + location + quality ranking.

The bundled directory (data/doctors_sample.csv) is SYNTHETIC demo data with placeholder names - never present it
as real. Load a verified directory with `python scripts/import_doctors.py your_file.csv`; it is then used instead.
"""
from __future__ import annotations

import math

import pandas as pd

from .config import ROOT
from .knowledge_base import get_info, specialties_for

REAL_PATH = ROOT / "data" / "doctors.csv"
SAMPLE_PATH = ROOT / "data" / "doctors_sample.csv"

# Approximate city-centre coordinates (lat, lon) for distance scoring.
CITY_COORDS = {
    "Ludhiana": (30.9010, 75.8573), "Chandigarh": (30.7333, 76.7794), "Amritsar": (31.6340, 74.8723),
    "Jalandhar": (31.3260, 75.5762), "Patiala": (30.3398, 76.3869), "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777), "Bengaluru": (12.9716, 77.5946), "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707), "Kolkata": (22.5726, 88.3639), "Pune": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873), "Lucknow": (26.8467, 80.9462),
}
WEIGHTS = {"Specialty match": 0.40, "Location": 0.25, "Rating": 0.20, "Experience": 0.10, "Availability": 0.05}


def load_doctors() -> tuple[pd.DataFrame, bool]:
    """Return (directory, is_sample)."""
    path, sample = (REAL_PATH, False) if REAL_PATH.exists() else (SAMPLE_PATH, True)
    df = pd.read_csv(path)
    df["teleconsult"] = df["teleconsult"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df, sample


def _km(a: tuple[float, float], b: tuple[float, float]) -> float:
    (la1, lo1), (la2, lo2) = map(lambda p: tuple(map(math.radians, p)), (a, b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def rank_doctors(df: pd.DataFrame, disease: str, city: str, *, max_fee: int | None = None, language: str | None = None,
                 teleconsult_only: bool = False, k: int = 8) -> pd.DataFrame:
    """Rank doctors for a disease. Columns: score components (0-1), final `score`, `distance_km`."""
    specs = specialties_for(disease)
    d = df[df["specialty"].isin(specs)].copy()
    if max_fee:
        d = d[d["fee_inr"] <= max_fee]
    if language:
        d = d[d["languages"].str.contains(language, case=False, na=False)]
    if teleconsult_only:
        d = d[d["teleconsult"]]
    if d.empty:
        return d
    here = CITY_COORDS.get(city)
    d["distance_km"] = d["city"].map(lambda c: 0.0 if c == city else (_km(here, CITY_COORDS[c]) if here and c in CITY_COORDS else float("nan")))
    spec_rank = {s: i for i, s in enumerate(specs)}
    d["Specialty match"] = d["specialty"].map(lambda s: 1.0 if spec_rank[s] == 0 else max(0.35, 0.7 - 0.15 * (spec_rank[s] - 1)))
    loc = d["distance_km"].map(lambda x: 0.3 if math.isnan(x) else math.exp(-x / 250))
    d["Location"] = [max(l, 0.55) if tele else l for l, tele in zip(loc, d["teleconsult"])]  # tele-visits soften distance
    d["Rating"] = ((d["rating"] - 3.0) / 2.0).clip(0, 1) * (d["reviews"].clip(upper=100) / 100) ** 0.25  # shrink low-review ratings
    d["Experience"] = (d["experience_years"].clip(upper=25) / 25)
    d["Availability"] = d["teleconsult"].astype(float) * 0.5 + 0.5
    d["score"] = sum(d[c] * w for c, w in WEIGHTS.items())
    return d.sort_values("score", ascending=False).head(k).reset_index(drop=True)


def emergency_advice(disease: str) -> str | None:
    if get_info(disease)["severity"] != "emergency":
        return None
    return ("🚨 This can be a medical emergency. Do not wait for an appointment: call your local emergency number "
            "(112 in India) or go to the nearest hospital emergency department now.")
