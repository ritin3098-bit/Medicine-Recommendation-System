"""Detailed symptom -> disease model trained on the 132-symptom open dataset (41 diseases)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from .config import MODEL_DIR, RANDOM_STATE
from .external import EXT, to_kb

MODEL_PATH = MODEL_DIR / "symptom_model.joblib"


def load_training() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Return (X, y, info). Duplicates are removed: random CV on duplicated rows would leak and inflate accuracy."""
    raw = pd.read_csv(EXT / "Training.csv")
    raw = raw.loc[:, ~raw.columns.str.contains(r"^Unnamed")]
    raw["prognosis"] = raw["prognosis"].map(to_kb)
    X = raw.drop(columns="prognosis")
    n_raw = len(raw)
    dedup = raw.drop_duplicates().reset_index(drop=True)
    conflicts = int(dedup.drop(columns="prognosis").duplicated(keep=False).sum())
    X, y = dedup.drop(columns="prognosis"), dedup["prognosis"]
    return X, y, {"raw_rows": n_raw, "unique_rows": len(dedup), "conflicting_rows": conflicts, "n_symptoms": X.shape[1]}


def severity_weights(columns) -> dict[str, float]:
    sev = pd.read_csv(EXT / "Symptom-severity.csv")
    w = {" ".join(str(s).replace("_", " ").split()).lower(): float(v) for s, v in zip(sev["Symptom"], sev["weight"])}
    return {c: w.get(" ".join(c.replace("_", " ").replace(".1", "").split()).lower(), 1.0) for c in columns}


def candidates():
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, C=5.0),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Naive Bayes (Bernoulli)": BernoulliNB(alpha=0.5),
        "k-Nearest Neighbours": KNeighborsClassifier(n_neighbors=3, weights="distance", metric="cosine"),
    }


def _drop_symptoms(X: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate incomplete user input: each reported symptom is forgotten with probability p (keep at least one)."""
    out = X.copy()
    mask = (rng.random(X.shape) < p) & (X == 1)
    out[mask] = 0
    for i in np.where(out.sum(axis=1) == 0)[0]:
        out[i, rng.choice(np.where(X[i] == 1)[0])] = 1
    return out


def _topk(proba: np.ndarray, y_idx: np.ndarray, k: int) -> float:
    return float(np.mean([y in row for y, row in zip(y_idx, np.argsort(-proba, axis=1)[:, :k])]))


def evaluate(model, X: np.ndarray, y: np.ndarray, n_classes: int, drop: float = 0.3) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    res = {k: [] for k in ("top1", "top3", "top1_dropout", "top3_dropout")}
    for tr, te in RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=RANDOM_STATE).split(X, y):
        m = model.fit(X[tr], y[tr])
        for tag, Xt in (("", X[te]), ("_dropout", _drop_symptoms(X[te], drop, rng))):
            proba = np.zeros((len(te), n_classes))
            proba[:, m.classes_] = m.predict_proba(Xt)
            res["top1" + tag].append(_topk(proba, y[te], 1))
            res["top3" + tag].append(_topk(proba, y[te], 3))
    return {k: float(np.mean(v)) for k, v in res.items()}


def train() -> dict | None:
    if not (EXT / "Training.csv").exists():
        return None
    X, y, info = load_training()
    le = LabelEncoder().fit(y)
    yi = le.transform(y)
    Xa = X.values.astype(float)
    scores = {n: evaluate(m, Xa, yi, len(le.classes_)) for n, m in candidates().items()}
    best = max(scores, key=lambda n: (scores[n]["top3_dropout"], scores[n]["top1_dropout"]))
    model = candidates()[best].fit(Xa, yi)
    joblib.dump({"model": model, "encoder": le, "symptoms": list(X.columns), "weights": severity_weights(X.columns)},
                MODEL_PATH, compress=3)
    return {"scores": scores, "best": best, "info": info, "n_diseases": int(len(le.classes_)), "dropout": 0.3,
            "baseline_top1": float(y.value_counts(normalize=True).iloc[0])}
