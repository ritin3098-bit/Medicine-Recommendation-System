"""Data loading, cleaning and feature engineering."""
from __future__ import annotations

import pandas as pd

from .config import DATA_PATH, SYMPTOMS
from .knowledge_base import CATEGORY_LIST, SEVERITY_RANK, get_info

# The source CSV has truncated / duplicated disease labels - map them to canonical names.
NAME_FIXES = {
    "Chronic Obstructive Pulmonary...": "Chronic Obstructive Pulmonary Disease (COPD)",
    "Eating Disorders (Anorexia": "Eating Disorders (Anorexia/Bulimia)",
    "Myocardial Infarction (Heart...": "Myocardial Infarction (Heart Attack)",
    "Obsessive-Compulsive Disorde...": "Obsessive-Compulsive Disorder (OCD)",
    "Systemic Lupus Erythematosus...": "Systemic Lupus Erythematosus (SLE)",
    "Urinary Tract Infection (UTI)": "Urinary Tract Infection",
}

BASE_FEATURES = SYMPTOMS + ["age", "male", "blood_pressure", "cholesterol_level", "symptom_count"]
RISK_FEATURES = BASE_FEATURES + [f"cat_{c}" for c in CATEGORY_LIST] + ["severity_rank"]


def load_clean() -> pd.DataFrame:
    """Return the cleaned dataset with binary symptoms and canonical disease names."""
    df = pd.read_csv(DATA_PATH)
    df["disease"] = df["disease"].replace(NAME_FIXES)
    for s in SYMPTOMS:
        df[s] = (df[s].str.lower() == "yes").astype(int)
    df["male"] = (df["gender"].str.lower() == "male").astype(int)
    df["symptom_count"] = df[SYMPTOMS].sum(axis=1)
    df["category"] = df["disease"].map(lambda d: get_info(d)["category"])
    df["severity"] = df["disease"].map(lambda d: get_info(d)["severity"])
    df["bp_label"] = df["blood_pressure"].map({0: "Low", 1: "Normal", 2: "High"})
    df["chol_label"] = df["cholesterol_level"].map({0: "Low", 1: "Normal", 2: "High"})
    return df


def disease_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[BASE_FEATURES].astype(float)


def risk_matrix(df: pd.DataFrame, disease: pd.Series | None = None) -> pd.DataFrame:
    """Features for the risk model = patient features + disease category/severity."""
    X = df[BASE_FEATURES].astype(float).copy()
    diseases = df["disease"] if disease is None else disease
    cats = diseases.map(lambda d: get_info(d)["category"])
    for c in CATEGORY_LIST:
        X[f"cat_{c}"] = (cats == c).astype(float).values
    X["severity_rank"] = diseases.map(lambda d: SEVERITY_RANK[get_info(d)["severity"]]).astype(float).values
    return X[RISK_FEATURES]
