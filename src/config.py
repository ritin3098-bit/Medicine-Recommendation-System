"""Central configuration for the Medicine & Exercise Recommendation System."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Cleaned_Dataset.csv"
MODEL_DIR = ROOT / "models"

RANDOM_STATE = 42
SYMPTOMS = ["fever", "cough", "fatigue", "difficulty_breathing"]
SYMPTOM_LABELS = {
    "fever": "Fever",
    "cough": "Cough",
    "fatigue": "Fatigue",
    "difficulty_breathing": "Difficulty breathing",
}
# In Cleaned_Dataset.csv blood pressure / cholesterol are ordinal codes.
LEVELS = {0: "Low", 1: "Normal", 2: "High"}
RISK_ORDER = ["Low", "Medium", "High"]
