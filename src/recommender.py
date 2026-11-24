"""Recommendation engine: disease -> medicines + exercise, tailored to the patient profile.

Components
----------
* Disease prediction  (ML)      : top-k diseases from symptoms / vitals.
* Risk prediction     (ML)      : Low / Medium / High, used to cap exercise intensity.
* Medicine safety filter (rules): allergies, pregnancy, age, comorbidities vs. risk tags.
* Exercise personalisation      : intensity cap = min(disease limit, ML risk, age, vitals, symptoms).
* Similar conditions (hybrid content-based): TF-IDF over treatment text + symptom-profile similarity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import MODEL_DIR, SYMPTOMS
from .data import BASE_FEATURES, load_clean, risk_matrix
from .external import pretty_symptom
from .symptom_model import MODEL_PATH as SYMPTOM_MODEL_PATH
from .knowledge_base import SEVERITY_TEXT, all_diseases, get_info

ALLERGY_OPTIONS = {
    "Penicillin / amoxicillin": "allergy:penicillin",
    "Sulfa drugs": "allergy:sulfa",
    "NSAIDs / aspirin": "allergy:nsaid",
    "Paracetamol": "allergy:paracetamol",
    "Macrolides (azithromycin)": "allergy:macrolide",
    "Fluoroquinolones (ciprofloxacin)": "allergy:fluoroquinolone",
}
CONDITION_OPTIONS = {
    "Kidney disease": "kidney",
    "Liver disease": "liver",
    "Stomach ulcer / GI bleeding": "ulcer",
    "Asthma": "asthma",
    "Diabetes": "diabetes",
    "High blood pressure": "hypertension",
}
TAG_TEXT = {
    "pregnancy": "not advised in pregnancy",
    "age_lt_12": "not suitable for young children",
    "age_gt_65": "use extra caution over age 65",
    "kidney": "caution with kidney disease",
    "liver": "caution with liver disease",
    "ulcer": "can irritate the stomach / ulcers",
    "asthma": "can worsen asthma",
    "diabetes": "can affect blood sugar",
    "hypertension": "can raise blood pressure",
}
HARD_TAGS = {"pregnancy", "age_lt_12"}  # plus every allergy:* tag
INTENSITY_NAME = {0: "Rest", 1: "Light", 2: "Moderate", 3: "Vigorous"}


@dataclass
class Profile:
    age: int = 35
    gender: str = "female"
    fever: bool = False
    cough: bool = False
    fatigue: bool = False
    difficulty_breathing: bool = False
    blood_pressure: int = 1       # 0 low, 1 normal, 2 high
    cholesterol_level: int = 1
    pregnant: bool = False
    allergies: tuple = ()          # tags such as "allergy:nsaid"
    conditions: tuple = ()         # tags such as "kidney"

    def to_row(self) -> pd.DataFrame:
        row = {s: int(getattr(self, s)) for s in SYMPTOMS}
        row.update(age=self.age, male=int(self.gender == "male"), blood_pressure=self.blood_pressure,
                   cholesterol_level=self.cholesterol_level)
        row["symptom_count"] = sum(row[s] for s in SYMPTOMS)
        return pd.DataFrame([row])[BASE_FEATURES].astype(float)

    def active_tags(self) -> set[str]:
        tags = set(self.allergies) | set(self.conditions)
        if self.pregnant and self.gender == "female":
            tags.add("pregnancy")
        if self.age < 12:
            tags.add("age_lt_12")
        if self.age >= 65:
            tags.add("age_gt_65")
        if self.blood_pressure == 2:
            tags.add("hypertension")
        return tags


def assess_medicine(med: dict, profile: Profile) -> tuple[str, list[str]]:
    """Return ('ok' | 'caution' | 'avoid', reasons) for one medicine and patient."""
    hits = [t for t in med["avoid"] if t in profile.active_tags()]
    reasons = []
    for t in hits:
        if t.startswith("allergy:"):
            reasons.append(f"you reported an allergy ({t.split(':')[1]})")
        else:
            reasons.append(TAG_TEXT[t])
    if any(t.startswith("allergy:") or t in HARD_TAGS for t in hits):
        return "avoid", reasons
    return ("caution" if hits else "ok"), reasons


class Recommender:
    def __init__(self):
        self.df = load_clean()
        d = joblib.load(MODEL_DIR / "disease_model.joblib")
        self.disease_model, self.encoder = d["model"], d["encoder"]
        r = joblib.load(MODEL_DIR / "risk_model.joblib")
        self.risk_model = r["model"]
        self.symptom_bundle = joblib.load(SYMPTOM_MODEL_PATH) if SYMPTOM_MODEL_PATH.exists() else None
        self._build_similarity()

    # ---------------- ML predictions ----------------
    def predict_disease(self, profile: Profile, k: int = 5) -> pd.DataFrame:
        proba = self.disease_model.predict_proba(profile.to_row())[0]
        classes = self.disease_model.classes_ if hasattr(self.disease_model, "classes_") else self.disease_model[-1].classes_
        order = np.argsort(-proba)[:k]
        return pd.DataFrame({
            "disease": self.encoder.inverse_transform(classes[order]),
            "probability": proba[order],
        })

    @property
    def symptom_options(self) -> dict[str, str]:
        """{column: friendly label} for the detailed 132-symptom model (empty if not trained)."""
        if not self.symptom_bundle:
            return {}
        return {c: pretty_symptom(c) for c in self.symptom_bundle["symptoms"]}

    def predict_from_symptoms(self, selected: list[str], k: int = 5) -> pd.DataFrame:
        """Detailed model: any subset of the 132 symptoms -> top-k of 41 diseases."""
        b = self.symptom_bundle
        x = np.array([[1.0 if c in set(selected) else 0.0 for c in b["symptoms"]]])
        proba = b["model"].predict_proba(x)[0]
        order = np.argsort(-proba)[:k]
        return pd.DataFrame({"disease": b["encoder"].inverse_transform(b["model"].classes_[order]), "probability": proba[order]})

    def symptom_burden(self, selected: list[str]) -> float:
        w = self.symptom_bundle["weights"] if self.symptom_bundle else {}
        return float(sum(w.get(c, 1.0) for c in selected))

    def predict_risk(self, profile: Profile, disease: str) -> dict[str, float]:
        row = profile.to_row().copy()
        row["disease"] = disease
        X = risk_matrix(row, row["disease"])
        proba = self.risk_model.predict_proba(X)[0]
        return dict(zip(self.risk_model.classes_, map(float, proba)))

    # ---------------- Medicines ----------------
    def recommend_medicines(self, disease: str, profile: Profile) -> list[dict]:
        meds = []
        for m in get_info(disease)["medicines"]:
            status, reasons = assess_medicine(m, profile)
            meds.append({**m, "status": status, "reasons": reasons})
        order = {"ok": 0, "caution": 1, "avoid": 2}
        return sorted(meds, key=lambda m: (order[m["status"]], m["rx"]))

    # ---------------- Exercise ----------------
    def exercise_cap(self, disease: str, profile: Profile, risk_probs: dict[str, float]) -> tuple[int, list[str]]:
        info = get_info(disease)
        cap, why = info["max_intensity"], [f"Condition limit: {INTENSITY_NAME[info['max_intensity']].lower()} activity at most"]
        risk = max(risk_probs, key=risk_probs.get)
        for limit, cond, text in [
            ({"High": 1, "Medium": 2, "Low": 3}[risk], True, f"ML risk level is {risk}"),
            (2, profile.age >= 65, "age 65 or older"),
            (1, profile.age >= 75, "age 75 or older"),
            (2, profile.blood_pressure == 2, "high blood pressure"),
            (1, profile.difficulty_breathing, "difficulty breathing"),
            (0, profile.fever, "fever - rest until fever-free for 24 h"),
        ]:
            if cond and limit < cap:
                cap = limit
                why.append(f"Lowered to {INTENSITY_NAME[limit].lower()}: {text}")
        return cap, why

    def recommend_exercises(self, disease: str, cap: int) -> list[dict]:
        info = get_info(disease)
        if cap == 0:
            return [e for e in info["exercises"] if e["intensity"] == 0] or [
                dict(key="rest", name="Rest, fluids and sleep", intensity=0, minutes=0,
                     benefit="Exercise can wait until you are cleared or feel better")]
        allowed = [e for e in info["exercises"] if 1 <= e["intensity"] <= cap]
        return allowed or [e for e in info["exercises"] if e["intensity"] == 1][:2]

    @staticmethod
    def weekly_plan(exercises: list[dict], cap: int) -> pd.DataFrame:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        active_days = {0: [], 1: [0, 2, 4, 5], 2: [0, 1, 3, 4, 5], 3: [0, 1, 3, 4, 5]}[cap]
        rows, i = [], 0
        for d, name in enumerate(days):
            if d in active_days and exercises and exercises[0]["intensity"] > 0:
                ex = exercises[i % len(exercises)]
                i += 1
                rows.append((name, ex["name"], ex["minutes"], INTENSITY_NAME[ex["intensity"]]))
            else:
                rows.append((name, "Rest / gentle movement", 0, "Rest"))
        return pd.DataFrame(rows, columns=["Day", "Activity", "Minutes", "Intensity"])

    # ---------------- Similar conditions (hybrid content-based) ----------------
    def _build_similarity(self):
        self.diseases = all_diseases()
        docs = []
        for d in self.diseases:
            i = get_info(d)
            docs.append(" ".join([i["category"], i["severity"]] + [f"{m['name']} {m['drug_class']} {m['use']}" for m in i["medicines"]]
                                 + [e["name"] for e in i["exercises"]]))
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(docs)
        self._text_sim = cosine_similarity(tfidf)
        prof = self.df.groupby("disease")[SYMPTOMS + ["blood_pressure", "cholesterol_level"]].mean()
        prof["age"] = self.df.groupby("disease")["age"].mean() / 90
        prof = prof.reindex(self.diseases).fillna(prof.mean())
        self._sym_sim = cosine_similarity(prof.values)

    def similar_conditions(self, disease: str, k: int = 5, w_text: float = 0.6) -> pd.DataFrame:
        i = self.diseases.index(disease)
        score = w_text * self._text_sim[i] + (1 - w_text) * self._sym_sim[i]
        score[i] = -1
        top = np.argsort(-score)[:k]
        return pd.DataFrame({
            "condition": [self.diseases[j] for j in top],
            "category": [get_info(self.diseases[j])["category"] for j in top],
            "similarity": [float(score[j]) for j in top],
        })

    # ---------------- One-call convenience ----------------
    def full_report(self, disease: str, profile: Profile) -> dict:
        info = get_info(disease)
        risk = self.predict_risk(profile, disease)
        cap, why = self.exercise_cap(disease, profile, risk)
        exercises = self.recommend_exercises(disease, cap)
        return {
            "info": info, "severity_style": SEVERITY_TEXT[info["severity"]],
            "risk": risk, "risk_level": max(risk, key=risk.get),
            "medicines": self.recommend_medicines(disease, profile),
            "cap": cap, "cap_reasons": why, "exercises": exercises,
            "plan": self.weekly_plan(exercises, cap),
            "similar": self.similar_conditions(disease),
        }
