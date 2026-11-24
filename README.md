# 💊 Personalized Healthcare — Medicine, Exercise & Doctor Recommendation System

A Data Science / ML project: pick (or predict) a disease → get **safety-checked medicine options**, a **risk-scaled
exercise plan**, **precautions**, **similar conditions** and a **ranked list of doctors**, in a Streamlit dashboard.

> ⚠️ **Educational project, not medical advice.** It does not diagnose or prescribe, stores no numeric doses, routes
> emergencies / cancers / genetic syndromes to a specialist, and the bundled doctor directory is **synthetic placeholder data**.

## Quick start
```bash
pip install -r requirements.txt
python -m src.train                    # (re)trains all models -> models/  (pre-trained copies included)
streamlit run app.py
pytest -q                              # 56 tests: ML, doctors, auth/security, and headless UI flows
```

## Accounts, login & roles
The app now opens on a **Log in / Register / Forgot password** screen. Nothing else is reachable until you sign in.

**First run:** create an administrator once (registration only ever creates normal users):
```bash
python -m scripts.create_admin                  # interactive, password not echoed
# or promote an existing account:  python -m scripts.create_admin --promote alice
# or set MEDREC_ADMIN_USER / MEDREC_ADMIN_EMAIL / MEDREC_ADMIN_PASSWORD before starting the app
```
| Role | Can use |
|---|---|
| **user** (default) | Recommend, My History, My Profile, Knowledge Base, About |
| **analyst** | + Data Explorer, Model Performance, Datasets, Usage & Feedback (aggregate, anonymous) |
| **admin** | + Admin page: change roles, enable/disable, unlock, issue temporary passwords, delete accounts |

What a signed-in user gets: profile pre-filled from their saved health profile, **save recommendations to history**
(view / export / delete), downloadable summary, star-rating feedback, account & password management, and account deletion
(removes all their data).

**Security design**
- Passwords and recovery codes are hashed with scrypt + random salt; the database never holds plain secrets.
- Generic login errors, timing-equalised lookups, 5 failed attempts lock the account for 10 minutes.
- Password rules: 8+ characters, a letter and a number, not a common password, not containing the username.
- Sessions live server-side and expire after 30 minutes idle; the role and active flag are re-checked from the database on
  every interaction, so a demoted or disabled user loses access immediately.
- Least privilege: analysts see only aggregates; admins manage accounts but **cannot read anyone's saved health history**.
  All data queries are scoped to the signed-in user.
- No email service, so password recovery uses a **one-time recovery code** shown at registration (or an admin-issued temporary
  password that forces a change at next login).

**Known limits (be upfront about these)**
- A browser refresh logs you out (Streamlit keeps sessions in memory; there is no secure cookie support built in).
- Data lives in a local SQLite file `data/app.db` (git-ignored). Back it up and protect it - it contains health information.
- For real deployment put the app behind HTTPS (e.g. a reverse proxy) and review privacy law obligations (GDPR / India's DPDP Act).
  This is a learning project, not a certified medical or compliance product.

## Datasets
| Dataset | Rows | Used for | Caveat |
|---|---|---|---|
| `data/Cleaned_Dataset.csv` (your medicare.zip) | 349 | Risk model, quick 4-symptom model, EDA | 116 diseases but only 4 symptoms |
| `data/external/Training.csv` | 4,920 → **304 unique** | Detailed model: 132 symptoms → 41 diseases | 94% duplicates (removed before evaluation); templated data |
| `Symptom-severity.csv` | 133 | Symptom burden score | Heuristic weights |
| `description.csv`, `precautions_df.csv`, `diets.csv`, `workout_df.csv` | 41 diseases | Description, precautions, diet & lifestyle tips | Generic reference text |
| `medications.csv` | 41 diseases | **Not shown** | Unfiltered lists would bypass the safety layer |
| `data/doctors_sample.csv` | 649 | Doctor finder demo | **Synthetic** placeholder names |

The open bundle is the widely used Kaggle-origin "Disease Prediction Using ML" dataset (mirrored at GitHub `sobhan2204/HealthEdu`) —
check the original licence before redistributing. The knowledge base grew from 116 to **131 diseases** so every dataset disease maps to it.

## Architecture
| Layer | File | Role |
|---|---|---|
| Data | `src/data.py`, `src/external.py` | Cleaning, label mapping, loaders for the open datasets |
| Knowledge base | `src/knowledge_base.py` | Disease → medicines (class, Rx/OTC, risk tags), exercises, specialties |
| ML | `src/train.py`, `src/symptom_model.py` | CV training of disease / risk / detailed-symptom models → `models/metrics.json` |
| Engine | `src/recommender.py` | Predictions, medicine safety filter, exercise cap, weekly plan, similarity |
| Doctors | `src/doctors.py`, `scripts/` | Specialty mapping, ranking, sample generator, real-data importer |
| Accounts | `src/db.py`, `src/auth.py`, `src/userdata.py`, `src/account_ui.py` | SQLite storage, hashing, roles, profile/history/feedback, account pages |
| Dashboard | `app.py` | Login gate + role-based pages: Recommend · History · Profile · Data Explorer · Model Performance · Datasets · Usage · Knowledge Base · Admin · About |

## Honest results
| Task | Best model | Result | Baseline |
|---|---|---|---|
| Quick disease check (116 classes, 4 symptoms) | Random Forest | top-1 ≈ 23%, top-3 ≈ 28% (5-fold CV) | 6.6% |
| Risk level (3 classes) | Gradient Boosting | accuracy ≈ 81%, macro-F1 ≈ 0.78 | 56.7% |
| Detailed symptoms (41 classes, deduplicated) | Logistic Regression | top-1 100%; **95.9% top-1 / 99.6% top-3 with 30% of symptoms forgotten** | ~3% |

The detailed model's near-perfect score reflects the templated dataset (every disease has a fixed symptom pattern), not clinical skill.

## Doctor finder
Disease → ordered specialties (e.g. Asthma → Pulmonologist, General Physician) → ranking:
`40% specialty match + 25% location + 20% rating (shrunk for few reviews) + 10% experience + 5% availability`,
with filters for city, max fee, language and video consult. Emergency conditions show "call emergency services" first.

To use **real** doctors: build a CSV with provenance (official medical-council register export, a hospital directory you may legally use)
and run `python scripts/import_doctors.py file.csv`. It validates specialties, requires `registration_no` + `source`, drops duplicates and
writes `data/doctors.csv`, which the app then uses instead of the sample. Don't scrape sites whose terms forbid it.

## Next steps
Drug–drug interaction data (openFDA / DDInter), lab values instead of coded BP/cholesterol, free-text symptom entry (NLP),
PDF summary export, and user feedback logging (the data needed for collaborative filtering).
