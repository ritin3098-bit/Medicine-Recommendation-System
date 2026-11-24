"""Loader for the open symptom / disease-info datasets in data/external/.

Source: the widely used "Disease Prediction Using Machine Learning" bundle (Kaggle-origin; mirrored on GitHub as
sobhan2204/HealthEdu). Please check the original licence before redistributing.
Files used: Training.csv, Symptom-severity.csv, description.csv, precautions_df.csv, diets.csv, workout_df.csv.
medications.csv is deliberately NOT surfaced: it is an unfiltered list that would bypass the safety layer.
"""
from __future__ import annotations

import ast
from functools import lru_cache

import pandas as pd

from .config import ROOT
from .knowledge_base import all_diseases

EXT = ROOT / "data" / "external"

_ALIASES = {
    "(vertigo) paroymsal positional vertigo": "Vertigo (BPPV)", "aids": "HIV/AIDS", "bronchial asthma": "Asthma",
    "cervical spondylosis": "Cervical Spondylosis", "chicken pox": "Chickenpox", "chronic cholestasis": "Chronic Cholestasis",
    "dengue": "Dengue Fever", "dimorphic hemmorhoids(piles)": "Hemorrhoids", "fungal infection": "Fungal Infection",
    "heart attack": "Myocardial Infarction (Heart Attack)", "hepatitis a": "Hepatitis A", "osteoarthristis": "Osteoarthritis",
    "paralysis (brain hemorrhage)": "Stroke", "peptic ulcer diseae": "Peptic Ulcer Disease", "typhoid": "Typhoid Fever",
    "alcoholic hepatitis": "Alcoholic Hepatitis", "varicose veins": "Varicose Veins",
}


def _norm(s: str) -> str:
    return " ".join(str(s).split()).lower()


@lru_cache(maxsize=1)
def _kb_lookup() -> dict[str, str]:
    return {_norm(d): d for d in all_diseases()}


def to_kb(name: str) -> str:
    """Map a dataset disease label to the canonical knowledge-base name (raises if unknown)."""
    n = _norm(name)
    if n in _ALIASES:
        return _ALIASES[n]
    if n in _kb_lookup():
        return _kb_lookup()[n]
    raise KeyError(f"Dataset disease '{name}' has no knowledge-base mapping")


def pretty_symptom(col: str) -> str:
    return col.replace("_", " ").replace(".1", "").strip().capitalize()


def _as_list(v) -> list[str]:
    try:
        out = ast.literal_eval(v) if isinstance(v, str) else list(v)
        return [str(x).strip() for x in out]
    except Exception:
        return [str(v)]


@lru_cache(maxsize=1)
def load_external() -> dict:
    """Return {'description': {}, 'precautions': {}, 'diet': {}, 'tips': {}} keyed by KB disease name."""
    out = {"description": {}, "precautions": {}, "diet": {}, "tips": {}}
    if not (EXT / "description.csv").exists():
        return out
    for _, r in pd.read_csv(EXT / "description.csv").iterrows():
        out["description"][to_kb(r["Disease"])] = r["Description"]
    for _, r in pd.read_csv(EXT / "precautions_df.csv").iterrows():
        vals = [str(r[c]).strip().capitalize() for c in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"] if pd.notna(r[c])]
        out["precautions"][to_kb(r["Disease"])] = vals
    for _, r in pd.read_csv(EXT / "diets.csv").iterrows():
        out["diet"][to_kb(r["Disease"])] = _as_list(r["Diet"])
    for d, g in pd.read_csv(EXT / "workout_df.csv").groupby("disease"):
        out["tips"][to_kb(d)] = list(dict.fromkeys(g["workout"].astype(str).str.strip()))
    return out


def get_external(disease: str) -> dict | None:
    ext = load_external()
    if disease not in ext["description"]:
        return None
    return {k: ext[k].get(disease) for k in ext}


def dataset_overview() -> pd.DataFrame:
    """Provenance + quality table for the Datasets page."""
    rows = []
    def add(name, df, use, caveat):
        rows.append({"Dataset": name, "Rows": len(df), "Columns": df.shape[1], "Used for": use, "Caveat": caveat})
    from .config import DATA_PATH
    add("Cleaned_Dataset.csv (your medicare.zip)", pd.read_csv(DATA_PATH), "Risk model + 4-symptom disease model, EDA",
        "Tiny: 349 rows, 116 diseases, only 4 symptoms")
    if (EXT / "Training.csv").exists():
        t = pd.read_csv(EXT / "Training.csv")
        add("Training.csv (132 symptoms, 41 diseases)", t, "Detailed symptom model",
            f"Only {len(t.drop_duplicates())} unique rows of {len(t)} - duplicates removed before evaluation; synthetic-style data")
        add("Symptom-severity.csv", pd.read_csv(EXT / "Symptom-severity.csv"), "Symptom burden score", "Weights are heuristic, not clinical")
        add("description.csv", pd.read_csv(EXT / "description.csv"), "Disease descriptions", "One-line summaries")
        add("precautions_df.csv", pd.read_csv(EXT / "precautions_df.csv"), "Precautions shown in the app", "Generic advice")
        add("diets.csv", pd.read_csv(EXT / "diets.csv"), "Diet reference list", "General foods list, not a diet plan")
        add("workout_df.csv", pd.read_csv(EXT / "workout_df.csv"), "Lifestyle tips", "Mostly lifestyle/diet tips, not exercises")
        add("medications.csv", pd.read_csv(EXT / "medications.csv"), "Not shown (reference only)", "Unfiltered drug lists - bypass the safety layer")
    return pd.DataFrame(rows)
