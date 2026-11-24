"""Generate the SYNTHETIC demo doctor directory (data/doctors_sample.csv). Names are placeholders by design."""
import numpy as np
import pandas as pd

from src.config import ROOT
from src.doctors import CITY_COORDS
from src.knowledge_base import SPECIALTIES

rng = np.random.default_rng(7)
LANG = {"Punjab": "English;Hindi;Punjabi", "Delhi": "English;Hindi", "Mumbai": "English;Hindi;Marathi", "Pune": "English;Hindi;Marathi",
        "Bengaluru": "English;Kannada;Hindi", "Hyderabad": "English;Telugu;Hindi", "Chennai": "English;Tamil",
        "Kolkata": "English;Bengali;Hindi", "Jaipur": "English;Hindi", "Lucknow": "English;Hindi", "Chandigarh": "English;Hindi;Punjabi"}
PUNJAB = {"Ludhiana", "Amritsar", "Jalandhar", "Patiala"}
BIG = {"Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Ludhiana", "Chandigarh"}
BASE_FEE = {"General Physician": 400, "Pediatrician": 500, "Physiotherapist": 500, "Emergency Medicine": 600}

rows = []
for city in CITY_COORDS:
    for spec in SPECIALTIES:
        if spec == "Emergency Medicine" and city not in BIG:
            continue
        for _ in range(int(rng.integers(2, 4)) if city in BIG else int(rng.integers(1, 3))):
            n = len(rows) + 1
            rows.append(dict(
                doctor_id=f"SD{n:04d}", name=f"Sample Doctor {n:04d}", specialty=spec, city=city,
                hospital=f"Sample {'Hospital' if rng.random() < .6 else 'Clinic'} {city} {'ABCD'[int(rng.integers(0, 4))]}",
                experience_years=int(rng.integers(2, 36)), rating=round(float(np.clip(rng.normal(4.3, .35), 3.2, 5.0)), 1),
                reviews=int(rng.integers(3, 900)),
                fee_inr=int(BASE_FEE.get(spec, 700) * rng.uniform(.8, 1.9) * (1.25 if city in {"Delhi", "Mumbai", "Bengaluru"} else 1) // 50 * 50),
                languages=LANG["Punjab" if city in PUNJAB else city], teleconsult=bool(rng.random() < .55),
                available_days=str(rng.choice(["Mon-Sat", "Mon-Fri", "Tue-Sun", "Mon,Wed,Fri"])),
                registration_no=f"SAMPLE-{n:04d}", source="synthetic demo data", is_sample=True))
pd.DataFrame(rows).to_csv(ROOT / "data" / "doctors_sample.csv", index=False)
print(len(rows), "sample doctors written")
