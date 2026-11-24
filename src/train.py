"""Train, cross-validate and save the ML models.

Run:  python -m src.train

Models
------
1. Disease classifier  - symptoms + vitals + demographics  -> disease (116 classes)
2. Risk classifier     - patient features + disease category -> Low / Medium / High
Evaluation uses K-fold cross-validation because the dataset is small (349 rows) and most
diseases have only 1-5 examples, so a single hold-out split would be too noisy to trust.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import MODEL_DIR, RANDOM_STATE, RISK_ORDER
from . import symptom_model
from .data import BASE_FEATURES, RISK_FEATURES, disease_matrix, load_clean, risk_matrix


def disease_candidates():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "k-Nearest Neighbours": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance")),
    }


def risk_candidates():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def _topk(proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
    top = np.argsort(-proba, axis=1)[:, :k]
    return float(np.mean([y in row for y, row in zip(y_true, top)]))


def cv_disease(model, X: pd.DataFrame, y: np.ndarray, n_classes: int, folds: int = 5) -> dict:
    proba = np.zeros((len(y), n_classes))
    for tr, te in KFold(folds, shuffle=True, random_state=RANDOM_STATE).split(X):
        m = model.fit(X.iloc[tr], y[tr])
        classes = m.classes_ if hasattr(m, "classes_") else m[-1].classes_
        proba[np.ix_(te, classes)] = m.predict_proba(X.iloc[te])
    return {f"top{k}": _topk(proba, y, k) for k in (1, 3, 5)}


def cv_risk(model, X: pd.DataFrame, y: pd.Series, folds: int = 5):
    pred = np.empty(len(y), dtype=object)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=RANDOM_STATE).split(X, y):
        pred[te] = model.fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }, pred


def main() -> dict:
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_clean()

    # ---------------- Disease model ----------------
    X = disease_matrix(df)
    le = LabelEncoder().fit(df["disease"])
    y = le.transform(df["disease"])
    d_scores = {name: cv_disease(m, X, y, len(le.classes_)) for name, m in disease_candidates().items()}
    best_d = max(d_scores, key=lambda n: (d_scores[n]["top3"], d_scores[n]["top1"]))
    disease_model = disease_candidates()[best_d].fit(X, y)
    baseline_top1 = float(df["disease"].value_counts(normalize=True).iloc[0])

    # ---------------- Risk model ----------------
    Xr = risk_matrix(df)
    yr = df["risk_level"]
    r_scores, r_preds = {}, {}
    for name, m in risk_candidates().items():
        r_scores[name], r_preds[name] = cv_risk(m, Xr, yr)
    best_r = max(r_scores, key=lambda n: r_scores[n]["macro_f1"])
    risk_model = risk_candidates()[best_r].fit(Xr, yr)
    cm = confusion_matrix(yr, r_preds[best_r], labels=RISK_ORDER).tolist()
    majority = float(yr.value_counts(normalize=True).iloc[0])

    # Interpretable feature importances (Random Forest, fitted on all data)
    rf_d = RandomForestClassifier(300, random_state=RANDOM_STATE).fit(X, y)
    rf_r = RandomForestClassifier(300, random_state=RANDOM_STATE).fit(Xr, yr)
    imp_d = dict(zip(BASE_FEATURES, map(float, rf_d.feature_importances_)))
    imp_r = dict(sorted(zip(RISK_FEATURES, map(float, rf_r.feature_importances_)), key=lambda kv: -kv[1])[:10])

    metrics = {
        "n_rows": int(len(df)), "n_diseases": int(len(le.classes_)),
        "disease": {"scores": d_scores, "best": best_d, "baseline_top1": baseline_top1, "importance": imp_d},
        "risk": {"scores": r_scores, "best": best_r, "baseline_accuracy": majority,
                 "confusion_matrix": cm, "labels": RISK_ORDER, "importance": imp_r},
        "cv_folds": 5,
        "symptom": symptom_model.train(),
    }
    joblib.dump({"model": disease_model, "encoder": le}, MODEL_DIR / "disease_model.joblib", compress=3)
    joblib.dump({"model": risk_model, "features": RISK_FEATURES}, MODEL_DIR / "risk_model.joblib", compress=3)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Disease model: {best_d}  top-1={d_scores[best_d]['top1']:.1%}  top-3={d_scores[best_d]['top3']:.1%}"
          f"  (majority baseline {baseline_top1:.1%})")
    print(f"Risk model:    {best_r}  acc={r_scores[best_r]['accuracy']:.1%}  macro-F1={r_scores[best_r]['macro_f1']:.2f}"
          f"  (majority baseline {majority:.1%})")
    sm = metrics["symptom"]
    if sm:
        b = sm["scores"][sm["best"]]
        print(f"Symptom model: {sm['best']}  top-1={b['top1']:.1%} top-3={b['top3']:.1%}  | with 30% symptoms forgotten: "
              f"top-1={b['top1_dropout']:.1%} top-3={b['top3_dropout']:.1%}  ({sm['info']['unique_rows']} unique rows)")
    return metrics


if __name__ == "__main__":
    main()
