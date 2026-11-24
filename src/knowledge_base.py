"""Curated, *educational* knowledge base: disease -> medicines + exercise guidance.

Design choices (deliberate, for safety):
* No numeric doses are stored. Dose / duration always come from a doctor or the label.
* Every medicine carries risk tags that the recommender checks against the patient profile
  (allergies, pregnancy, age, kidney / liver disease, ulcer, asthma, ...).
* Serious conditions (emergencies, cancers, genetic syndromes) intentionally have no
  self-directed medicines - the system routes the user to a specialist instead.
"""
from __future__ import annotations

# --------------------------------------------------------------------------------------
# Medicines: key -> (display name, drug class, used for, prescription-only?, risk tags)
# --------------------------------------------------------------------------------------
_M = {
    # pain / fever / allergy / cold
    "paracetamol": ("Paracetamol (Acetaminophen)", "Analgesic / antipyretic", "Fever and mild-to-moderate pain", False, ("liver", "allergy:paracetamol")),
    "ibuprofen": ("Ibuprofen", "NSAID", "Pain, inflammation and fever", False, ("ulcer", "kidney", "hypertension", "asthma", "pregnancy", "age_gt_65", "allergy:nsaid")),
    "topical_nsaid": ("Diclofenac gel (topical NSAID)", "Topical NSAID", "Joint pain applied to the skin", False, ("pregnancy", "allergy:nsaid")),
    "ors": ("Oral rehydration solution (ORS)", "Rehydration", "Replaces fluid and salts lost with fever, vomiting or diarrhoea", False, ()),
    "cetirizine": ("Cetirizine", "Antihistamine", "Allergy symptoms and itching", False, ("kidney",)),
    "loratadine": ("Loratadine", "Antihistamine", "Hay fever and allergic symptoms", False, ()),
    "saline_spray": ("Saline nasal spray / rinse", "Nasal saline", "Clears a blocked nose and sinuses", False, ()),
    "nasal_steroid": ("Fluticasone nasal spray", "Nasal corticosteroid", "Nasal inflammation (allergic rhinitis, sinusitis)", True, ()),
    "lozenges": ("Throat lozenges / warm honey drink", "Soothing agent", "Sore throat and cough (honey not for infants under 1 year)", False, ("age_lt_12",)),
    # anti-infectives
    "amoxicillin": ("Amoxicillin", "Penicillin antibiotic", "Bacterial infections (chest, ear, throat, sinus)", True, ("allergy:penicillin",)),
    "amox_clav": ("Amoxicillin-clavulanate", "Penicillin antibiotic", "Mixed bacterial infections (e.g. abdominal)", True, ("allergy:penicillin", "liver")),
    "azithromycin": ("Azithromycin", "Macrolide antibiotic", "Bacterial infections (typhoid, cholera, chest)", True, ("allergy:macrolide", "liver")),
    "doxycycline": ("Doxycycline", "Tetracycline antibiotic", "Acne, Lyme disease, some bacterial infections", True, ("pregnancy", "age_lt_12")),
    "ciprofloxacin": ("Ciprofloxacin", "Fluoroquinolone antibiotic", "Urinary and some gut infections", True, ("allergy:fluoroquinolone", "pregnancy", "age_lt_12", "age_gt_65")),
    "nitrofurantoin": ("Nitrofurantoin", "Urinary antibiotic", "Uncomplicated urinary tract infection", True, ("kidney", "age_lt_12")),
    "cotrimoxazole": ("Co-trimoxazole", "Sulfonamide antibiotic", "Urinary infections, Pneumocystis pneumonia", True, ("allergy:sulfa", "pregnancy", "kidney")),
    "eye_antibiotic": ("Antibiotic eye drops", "Topical antibiotic", "Bacterial conjunctivitis", True, ()),
    "oseltamivir": ("Oseltamivir", "Antiviral", "Influenza (best started early)", True, ("kidney",)),
    "acyclovir": ("Acyclovir", "Antiviral", "Chickenpox / herpes-family viruses", True, ("kidney",)),
    "tenofovir": ("Tenofovir / Entecavir", "Antiviral", "Chronic hepatitis B", True, ("kidney",)),
    "art": ("Antiretroviral therapy (ART)", "Antiretroviral", "HIV infection (combination regimen set by specialist)", True, ()),
    "antitb": ("Anti-tuberculosis regimen", "Antitubercular combination", "Tuberculosis (multi-drug, months long)", True, ("liver",)),
    "antimalarial": ("Artemisinin-based combination therapy", "Antimalarial", "Malaria", True, ("pregnancy",)),
    "vitamin_a": ("Vitamin A (as advised)", "Supplement", "Measles in children - reduces complications", False, ()),
    # respiratory
    "salbutamol": ("Salbutamol (reliever inhaler)", "Short-acting bronchodilator", "Quick relief of wheeze and breathlessness", True, ()),
    "ics": ("Inhaled corticosteroid (controller)", "Corticosteroid inhaler", "Daily asthma / COPD control", True, ()),
    "montelukast": ("Montelukast", "Leukotriene antagonist", "Asthma and allergic rhinitis control", True, ()),
    "tiotropium": ("Tiotropium inhaler", "Long-acting bronchodilator", "COPD maintenance", True, ()),
    # cardiovascular / metabolic
    "amlodipine": ("Amlodipine", "Calcium-channel blocker", "High blood pressure, angina", True, ()),
    "ramipril": ("Ramipril / Enalapril", "ACE inhibitor", "High blood pressure, heart protection", True, ("pregnancy", "kidney")),
    "losartan": ("Losartan", "ARB", "High blood pressure, heart protection", True, ("pregnancy", "kidney")),
    "hctz": ("Hydrochlorothiazide", "Thiazide diuretic", "High blood pressure, fluid retention", True, ("allergy:sulfa", "pregnancy", "kidney")),
    "metoprolol": ("Metoprolol", "Beta-blocker", "Heart rate / BP control, angina", True, ("asthma",)),
    "atorvastatin": ("Atorvastatin", "Statin", "Lowers cholesterol, protects arteries", True, ("liver", "pregnancy")),
    "aspirin_low": ("Low-dose aspirin", "Antiplatelet", "Prevents clots in established heart disease", True, ("ulcer", "asthma", "allergy:nsaid", "pregnancy", "age_lt_12")),
    "metformin": ("Metformin", "Biguanide", "Type 2 diabetes", True, ("kidney", "liver")),
    "insulin": ("Insulin (as prescribed)", "Hormone therapy", "Diabetes / severe hyperglycaemia", True, ()),
    "glucose_tabs": ("Glucose tablets / fast sugar", "Fast carbohydrate", "Treats low blood sugar immediately", False, ()),
    "levothyroxine": ("Levothyroxine", "Thyroid hormone", "Underactive thyroid", True, ()),
    "methimazole": ("Methimazole", "Antithyroid", "Overactive thyroid", True, ("pregnancy", "liver")),
    "combined_oc": ("Hormonal contraceptive pill", "Hormonal therapy", "PCOS / endometriosis symptom control", True, ("pregnancy", "hypertension")),
    # digestive
    "ppi": ("Pantoprazole / Omeprazole", "Proton-pump inhibitor", "Acid reflux, stomach protection", False, ()),
    "antacid": ("Antacid", "Acid neutraliser", "Short-term heartburn relief", False, ("kidney",)),
    "loperamide": ("Loperamide", "Anti-motility", "Short-term non-bloody diarrhoea", False, ("age_lt_12",)),
    "mesalamine": ("Mesalamine (5-ASA)", "Anti-inflammatory (bowel)", "Ulcerative colitis", True, ("kidney", "allergy:nsaid")),
    "lactulose": ("Lactulose", "Osmotic laxative", "Constipation / liver-related confusion", True, ()),
    "fiber": ("Psyllium fibre", "Bulk-forming fibre", "Softens stools, prevents straining", False, ()),
    "hemorrhoid_cream": ("Haemorrhoid cream / suppository", "Topical", "Local pain, itching and swelling", False, ()),
    # neuro / mental health
    "sumatriptan": ("Sumatriptan", "Triptan", "Acute migraine attacks", True, ("hypertension",)),
    "levetiracetam": ("Levetiracetam", "Anti-seizure", "Epilepsy", True, ("kidney", "pregnancy")),
    "levodopa": ("Levodopa-carbidopa", "Dopamine replacement", "Parkinson's disease", True, ()),
    "donepezil": ("Donepezil", "Cholinesterase inhibitor", "Alzheimer's / dementia symptoms", True, ()),
    "sertraline": ("Sertraline", "SSRI antidepressant", "Depression, anxiety, OCD", True, ("pregnancy",)),
    "lithium": ("Lithium", "Mood stabiliser", "Bipolar disorder", True, ("kidney", "pregnancy")),
    "antipsychotic": ("Antipsychotic (e.g. risperidone)", "Antipsychotic", "Schizophrenia / psychosis", True, ("pregnancy",)),
    "duloxetine": ("Duloxetine", "SNRI", "Fibromyalgia pain, depression", True, ("liver", "kidney", "pregnancy")),
    # skin / eye
    "benzoyl": ("Benzoyl peroxide", "Topical antibacterial", "Acne", False, ()),
    "adapalene": ("Adapalene", "Topical retinoid", "Acne", False, ("pregnancy",)),
    "emollient": ("Emollient / moisturiser", "Skin barrier repair", "Eczema and psoriasis dryness", False, ()),
    "hc_cream": ("Hydrocortisone cream", "Topical corticosteroid", "Eczema / itchy inflamed skin", False, ()),
    "calcipotriol": ("Calcipotriol", "Vitamin D analogue (topical)", "Plaque psoriasis", True, ("kidney",)),
    "calamine": ("Calamine lotion", "Soothing topical", "Itch from chickenpox / rashes", False, ()),
    "artificial_tears": ("Lubricating eye drops", "Ocular lubricant", "Dry or irritated eyes, mild conjunctivitis", False, ()),
    "latanoprost": ("Latanoprost eye drops", "Prostaglandin analogue", "Glaucoma (lowers eye pressure)", True, ("pregnancy",)),
    "timolol": ("Timolol eye drops", "Beta-blocker drops", "Glaucoma (lowers eye pressure)", True, ("asthma",)),
    # musculoskeletal / autoimmune / blood
    "methotrexate": ("Methotrexate", "DMARD / immunosuppressant", "Rheumatoid arthritis, severe psoriasis", True, ("pregnancy", "liver", "kidney")),
    "prednisolone": ("Prednisolone", "Oral corticosteroid", "Flares of autoimmune / inflammatory disease", True, ("ulcer", "diabetes", "hypertension")),
    "hydroxychloroquine": ("Hydroxychloroquine", "Antimalarial / DMARD", "Lupus and rheumatoid arthritis", True, ()),
    "allopurinol": ("Allopurinol", "Xanthine-oxidase inhibitor", "Prevents gout attacks", True, ("kidney",)),
    "colchicine": ("Colchicine", "Anti-gout", "Acute gout attacks", True, ("kidney", "liver")),
    "calcium_d": ("Calcium + Vitamin D", "Supplement", "Bone health", False, ("kidney",)),
    "alendronate": ("Alendronate", "Bisphosphonate", "Osteoporosis", True, ("ulcer", "kidney")),
    "iron": ("Iron supplement (ferrous sulfate)", "Supplement", "Iron-deficiency anaemia", False, ("ulcer",)),
    "folic_acid": ("Folic acid", "Vitamin", "Folate-deficiency anaemia, sickle cell support", False, ()),
    "b12": ("Vitamin B12", "Vitamin", "B12-deficiency anaemia", False, ()),
    "hydroxyurea": ("Hydroxyurea", "Antimetabolite", "Reduces sickle-cell crises", True, ("pregnancy", "kidney", "liver")),
    "factor": ("Clotting-factor replacement", "Biologic", "Hemophilia bleeding prevention (specialist centre)", True, ()),    "antifungal": ("Clotrimazole / terbinafine cream", "Topical antifungal", "Fungal skin infections (ringworm, athlete's foot)", False, ()),
    "mupirocin": ("Mupirocin ointment", "Topical antibiotic", "Impetigo (localised skin infection)", True, ()),
    "daa": ("Direct-acting antivirals (e.g. sofosbuvir-based)", "Antiviral", "Hepatitis C cure (specialist-prescribed)", True, ("pregnancy", "kidney")),
    "h_pylori": ("H. pylori eradication regimen", "Antibiotic combination + PPI", "Ulcers caused by H. pylori (test first)", True, ("allergy:penicillin", "allergy:macrolide", "pregnancy")),
}

# --------------------------------------------------------------------------------------
# Exercises: key -> (name, intensity 0-3, minutes, benefit)
# intensity: 0 rest, 1 light, 2 moderate, 3 vigorous
# --------------------------------------------------------------------------------------
_EX = {
    "rest": ("Rest, fluids and sleep", 0, 0, "Lets the body recover - exercise can wait until you feel better"),
    "diaphragmatic": ("Diaphragmatic (belly) breathing", 1, 10, "Improves lung efficiency and calms the nervous system"),
    "pursed_lip": ("Pursed-lip breathing", 1, 10, "Keeps airways open and eases breathlessness"),
    "gentle_stretch": ("Gentle full-body stretching", 1, 15, "Maintains flexibility and reduces stiffness"),
    "easy_walk": ("Easy walking", 1, 20, "Low-impact circulation boost, safe starting point"),
    "brisk_walk": ("Brisk walking", 2, 30, "Heart, mood and metabolic health"),
    "post_meal_walk": ("Short walk after meals", 1, 15, "Helps lower post-meal blood sugar"),
    "cycling": ("Steady cycling / stationary bike", 2, 30, "Joint-friendly cardio"),
    "swim": ("Swimming / water aerobics", 2, 30, "Full-body, low-impact and warm moist air"),
    "yoga": ("Gentle yoga", 1, 20, "Flexibility, stress relief and breath control"),
    "tai_chi": ("Tai chi", 1, 20, "Balance, blood-pressure and stress benefits"),
    "strength_light": ("Light strength training (bodyweight / light weights)", 2, 25, "Muscle, bone and insulin sensitivity"),
    "strength_band": ("Resistance-band exercises", 2, 20, "Joint-friendly muscle strengthening"),
    "jog": ("Jogging / interval cardio", 3, 25, "Vigorous fitness, weight and metabolic control"),
    "balance": ("Balance training", 1, 15, "Reduces fall risk"),
    "kegel": ("Pelvic-floor / gentle core work", 1, 10, "Pelvic and trunk stability"),
    "rom": ("Range-of-motion joint exercises", 1, 10, "Keeps joints moving without strain"),
    "meditation": ("Mindfulness / meditation", 1, 10, "Lowers stress and anxiety"),
    "chair": ("Seated chair exercises", 1, 15, "Safe movement when standing is difficult"),    "neck_mobility": ("Neck & shoulder mobility", 1, 10, "Eases neck stiffness and shoulder tension"),
}

# --------------------------------------------------------------------------------------
# Categories: shared exercise guidance
# --------------------------------------------------------------------------------------
_CAT = {
    "Respiratory": dict(ex=["diaphragmatic", "pursed_lip", "easy_walk", "yoga", "brisk_walk", "swim"], max_int=2,
                        avoid="Avoid cold, dusty or polluted air; stop if wheezy or breathless."),
    "Cardiovascular": dict(ex=["easy_walk", "brisk_walk", "cycling", "tai_chi", "strength_light", "gentle_stretch"], max_int=2,
                           avoid="Avoid heavy lifting, breath-holding and extreme heat; stop for chest pain or dizziness."),
    "Metabolic & Endocrine": dict(ex=["post_meal_walk", "brisk_walk", "strength_light", "cycling", "swim", "yoga"], max_int=3,
                                  avoid="Check glucose before/after exercise if on insulin; carry fast sugar."),
    "Digestive & Liver": dict(ex=["easy_walk", "yoga", "diaphragmatic", "kegel", "gentle_stretch"], max_int=2,
                              avoid="Avoid intense abdominal workouts and dehydration, especially during flares."),
    "Kidney & Urinary": dict(ex=["easy_walk", "gentle_stretch", "tai_chi", "kegel", "chair"], max_int=2,
                             avoid="Avoid dehydration and very strenuous workouts."),
    "Neurological": dict(ex=["balance", "tai_chi", "easy_walk", "gentle_stretch", "chair", "rom"], max_int=2,
                         avoid="Avoid exercising alone, heights and overheating; use support for balance."),
    "Mental Health": dict(ex=["brisk_walk", "yoga", "meditation", "tai_chi", "swim", "strength_light"], max_int=3,
                          avoid="Keep a steady routine; avoid over-exercising as a way to cope."),
    "Musculoskeletal & Autoimmune": dict(ex=["swim", "rom", "gentle_stretch", "strength_band", "cycling", "easy_walk", "balance"], max_int=2,
                                         avoid="Avoid high-impact moves and exercising through joint pain; rest during flares."),
    "Skin": dict(ex=["brisk_walk", "yoga", "cycling", "strength_light", "easy_walk"], max_int=3,
                 avoid="Shower and change after sweating; avoid chafing clothes and chlorinated pools if skin is irritated."),
    "Eye & Ear": dict(ex=["easy_walk", "gentle_stretch", "yoga", "tai_chi"], max_int=2,
                      avoid="Avoid heavy straining and head-down poses (glaucoma); keep ears dry during infection."),
    "Infectious Disease": dict(ex=["rest", "diaphragmatic", "gentle_stretch", "easy_walk"], max_int=1,
                               avoid="Rest until fever-free for 24 h; return to activity gradually; avoid gyms while contagious."),
    "Cancer": dict(ex=["easy_walk", "gentle_stretch", "yoga", "balance", "strength_band"], max_int=1,
                   avoid="Only with your oncology team's clearance; stop if exhausted, dizzy or in pain."),
    "Blood & Genetic": dict(ex=["easy_walk", "gentle_stretch", "swim", "rom", "balance"], max_int=2,
                            avoid="Avoid contact sports and extreme exertion unless your specialist clears you."),
    "Reproductive": dict(ex=["easy_walk", "yoga", "kegel", "brisk_walk", "strength_light"], max_int=2,
                         avoid="Listen to your body; reduce intensity during painful days."),
}
CATEGORY_LIST = list(_CAT)
SEVERITY_RANK = {"self_care": 0, "doctor": 1, "emergency": 2}

SPECIALIST_NOTE = ("No self-directed medicine is suggested for this condition. Treatment is planned and "
                   "monitored by a specialist - please consult a doctor.")

_CANCER = ("Cancer", "doctor", [], {"note": "Cancer treatment (surgery, chemotherapy, radiotherapy, targeted or immune therapy) is planned by an oncology team."})
_EMERG = lambda cat, note="": (cat, "emergency", [], {"max_int": 0, "ex": ["rest"], "note": note or "This can be life-threatening. Seek emergency medical care immediately - do not self-medicate."})

# --------------------------------------------------------------------------------------
# Diseases: name -> (category, severity, [medicine keys], overrides)
# overrides: ex (list of exercise keys), max_int, avoid (text), note (text)
# --------------------------------------------------------------------------------------
_D = {
    # ---- Respiratory
    "Asthma": ("Respiratory", "doctor", ["salbutamol", "ics", "montelukast"], {"ex": ["diaphragmatic", "pursed_lip", "swim", "easy_walk", "yoga"], "avoid": "Warm up slowly, keep your reliever inhaler with you, avoid cold/dry air and known triggers."}),
    "Bronchitis": ("Respiratory", "self_care", ["paracetamol", "lozenges", "saline_spray"], {}),
    "Chronic Obstructive Pulmonary Disease (COPD)": ("Respiratory", "doctor", ["tiotropium", "salbutamol", "ics"], {"ex": ["pursed_lip", "diaphragmatic", "easy_walk", "chair", "tai_chi"], "max_int": 1}),
    "Pneumonia": ("Respiratory", "doctor", ["amoxicillin", "azithromycin", "paracetamol", "ors"], {"max_int": 0, "ex": ["rest", "diaphragmatic"]}),
    "Pneumothorax": _EMERG("Respiratory", "A collapsed lung needs urgent medical assessment - go to an emergency department."),
    "Tuberculosis": ("Respiratory", "doctor", ["antitb"], {"max_int": 1}),
    "Cystic Fibrosis": ("Respiratory", "doctor", [], {"ex": ["diaphragmatic", "easy_walk", "swim", "gentle_stretch"], "note": "Managed by a specialist CF centre (airway clearance, enzymes, modulators)."}),
    "Sleep Apnea": ("Respiratory", "doctor", [], {"ex": ["brisk_walk", "swim", "cycling", "strength_light"], "max_int": 3, "note": "First-line treatment is CPAP (a device), plus weight management; see a sleep specialist."}),
    "Sinusitis": ("Respiratory", "self_care", ["saline_spray", "nasal_steroid", "paracetamol", "amoxicillin"], {}),
    "Allergic Rhinitis": ("Respiratory", "self_care", ["cetirizine", "loratadine", "nasal_steroid", "saline_spray"], {"max_int": 3, "ex": ["brisk_walk", "swim", "yoga", "diaphragmatic"], "avoid": "Exercise indoors on high-pollen days."}),
    "Common Cold": ("Infectious Disease", "self_care", ["paracetamol", "saline_spray", "lozenges", "cetirizine"], {}),
    "Influenza": ("Infectious Disease", "self_care", ["oseltamivir", "paracetamol", "ors"], {}),
    "Tonsillitis": ("Respiratory", "self_care", ["paracetamol", "ibuprofen", "lozenges", "amoxicillin"], {"max_int": 1}),
    "Pneumocystis Pneumonia (PCP)": ("Respiratory", "doctor", ["cotrimoxazole"], {"max_int": 0, "ex": ["rest", "diaphragmatic"], "note": "Usually occurs with weakened immunity; needs hospital-level assessment."}),
    # ---- Cardiovascular
    "Hypertension": ("Cardiovascular", "doctor", ["amlodipine", "ramipril", "losartan", "hctz"], {"avoid": "Avoid heavy isometric lifting and breath-holding; check BP regularly."}),
    "Hypertensive Heart Disease": ("Cardiovascular", "doctor", ["ramipril", "hctz", "metoprolol"], {"max_int": 1}),
    "Coronary Artery Disease": ("Cardiovascular", "doctor", ["atorvastatin", "aspirin_low", "metoprolol"], {"max_int": 1, "avoid": "Exercise only within the limits set by your cardiologist; stop for chest pain."}),
    "Atherosclerosis": ("Cardiovascular", "doctor", ["atorvastatin", "aspirin_low"], {}),
    "Myocardial Infarction (Heart Attack)": _EMERG("Cardiovascular", "Chest pain, pressure, sweating or breathlessness may be a heart attack - call emergency services now."),
    "Stroke": _EMERG("Cardiovascular", "Face drooping, arm weakness or speech trouble: call emergency services immediately (time is brain)."),
    # ---- Metabolic & endocrine
    "Diabetes": ("Metabolic & Endocrine", "doctor", ["metformin", "insulin", "glucose_tabs"], {}),
    "Hyperglycemia": ("Metabolic & Endocrine", "doctor", ["insulin", "metformin"], {"max_int": 1, "avoid": "Do not exercise vigorously if glucose is very high or ketones are present."}),
    "Hypoglycemia": ("Metabolic & Endocrine", "doctor", ["glucose_tabs"], {"max_int": 1, "ex": ["easy_walk", "gentle_stretch"], "avoid": "Do not exercise during a low; treat first, recheck, and always carry fast sugar."}),
    "Hypothyroidism": ("Metabolic & Endocrine", "doctor", ["levothyroxine"], {}),
    "Hyperthyroidism": ("Metabolic & Endocrine", "doctor", ["methimazole", "metoprolol"], {"max_int": 1, "avoid": "Avoid strenuous exercise until thyroid levels are controlled (fast heart rate)."}),
    "Polycystic Ovary Syndrome (PCOS)": ("Reproductive", "doctor", ["metformin", "combined_oc"], {"max_int": 3, "ex": ["brisk_walk", "strength_light", "cycling", "yoga", "jog"]}),
    "Endometriosis": ("Reproductive", "doctor", ["ibuprofen", "combined_oc"], {}),
    # ---- Digestive & liver
    "Gastroenteritis": ("Digestive & Liver", "self_care", ["ors", "loperamide", "paracetamol"], {"max_int": 1}),
    "Crohn's Disease": ("Digestive & Liver", "doctor", ["prednisolone"], {"note": "Maintenance therapy (immunomodulators / biologics) is chosen by a gastroenterologist."}),
    "Ulcerative Colitis": ("Digestive & Liver", "doctor", ["mesalamine", "prednisolone"], {}),
    "Diverticulitis": ("Digestive & Liver", "doctor", ["amox_clav", "paracetamol"], {"max_int": 0, "ex": ["rest"]}),
    "Hemorrhoids": ("Digestive & Liver", "self_care", ["fiber", "hemorrhoid_cream", "paracetamol"], {}),
    "Appendicitis": _EMERG("Digestive & Liver", "Sudden worsening abdominal pain (especially lower right) needs emergency surgical assessment."),
    "Cholecystitis": _EMERG("Digestive & Liver", "Severe upper-right abdominal pain with fever needs urgent hospital care."),
    "Pancreatitis": _EMERG("Digestive & Liver", "Severe upper abdominal pain radiating to the back needs urgent hospital care."),
    "Cirrhosis": ("Digestive & Liver", "doctor", ["lactulose"], {"max_int": 1, "note": "Avoid alcohol and NSAIDs; management is by a hepatologist."}),
    "Hepatitis": ("Digestive & Liver", "doctor", [], {"max_int": 1, "note": "Mostly supportive care (rest, fluids, no alcohol). Avoid unnecessary medicines - the liver is inflamed."}),
    "Hepatitis B": ("Digestive & Liver", "doctor", ["tenofovir"], {"max_int": 1}),
    "Liver Disease": ("Digestive & Liver", "doctor", [], {"max_int": 1, "note": "Avoid alcohol and unnecessary medicines (including NSAIDs); see a hepatologist."}),
    "Cholera": ("Digestive & Liver", "doctor", ["ors", "azithromycin", "doxycycline"], {"max_int": 0, "ex": ["rest"], "note": "Rapid rehydration is life-saving - seek care urgently."}),
    "Typhoid Fever": ("Infectious Disease", "doctor", ["azithromycin", "paracetamol", "ors"], {}),
    # ---- Kidney & urinary
    "Chronic Kidney Disease": ("Kidney & Urinary", "doctor", [], {"note": "Avoid NSAIDs. Medicines are dose-adjusted to kidney function by a nephrologist."}),
    "Kidney Disease": ("Kidney & Urinary", "doctor", [], {"note": "Avoid NSAIDs. Medicines are dose-adjusted to kidney function by a nephrologist."}),
    "Urinary Tract Infection": ("Kidney & Urinary", "doctor", ["nitrofurantoin", "cotrimoxazole", "ciprofloxacin", "paracetamol"], {"max_int": 1}),
    # ---- Neurological
    "Migraine": ("Neurological", "self_care", ["paracetamol", "ibuprofen", "sumatriptan"], {"ex": ["easy_walk", "yoga", "tai_chi", "meditation", "gentle_stretch"], "avoid": "Avoid exercise during an attack; dehydration and skipped meals can trigger migraines."}),
    "Epilepsy": ("Neurological", "doctor", ["levetiracetam"], {"avoid": "Never swim or cycle alone; avoid heights; follow your neurologist's activity advice."}),
    "Parkinson's Disease": ("Neurological", "doctor", ["levodopa"], {"ex": ["balance", "tai_chi", "easy_walk", "rom", "chair"]}),
    "Alzheimer's Disease": ("Neurological", "doctor", ["donepezil"], {"ex": ["easy_walk", "tai_chi", "chair", "balance"], "avoid": "Exercise with a companion for safety."}),
    "Dementia": ("Neurological", "doctor", ["donepezil"], {"ex": ["easy_walk", "tai_chi", "chair", "balance"], "avoid": "Exercise with a companion for safety."}),
    "Multiple Sclerosis": ("Neurological", "doctor", [], {"avoid": "Avoid overheating and fatigue; rest breaks are essential.", "note": "Disease-modifying therapy is chosen by a neurologist."}),
    "Cerebral Palsy": ("Blood & Genetic", "doctor", [], {"ex": ["rom", "gentle_stretch", "swim", "chair", "balance"], "note": "Care is multidisciplinary (physiotherapy, orthopaedics, neurology)."}),
    "Spina Bifida": ("Blood & Genetic", "doctor", [], {"ex": ["rom", "swim", "chair", "gentle_stretch"], "note": "Care is multidisciplinary (neurosurgery, urology, physiotherapy)."}),
    "Tourette Syndrome": ("Neurological", "doctor", [], {"max_int": 3, "note": "Behavioural therapy (CBIT) is first-line; medicines only via a specialist."}),
    # ---- Mental health
    "Depression": ("Mental Health", "doctor", ["sertraline"], {}),
    "Anxiety Disorders": ("Mental Health", "doctor", ["sertraline"], {}),
    "Bipolar Disorder": ("Mental Health", "doctor", ["lithium"], {}),
    "Schizophrenia": ("Mental Health", "doctor", ["antipsychotic"], {"max_int": 2}),
    "Obsessive-Compulsive Disorder (OCD)": ("Mental Health", "doctor", ["sertraline"], {}),
    "Eating Disorders (Anorexia/Bulimia)": ("Mental Health", "doctor", [], {"max_int": 0, "ex": ["rest", "meditation"], "note": "Treatment is therapy-led with medical and nutritional monitoring. Exercise should only restart when your care team says so."}),
    "Autism Spectrum Disorder (ASD)": ("Mental Health", "doctor", [], {"max_int": 3, "note": "Support is therapy and education based; any medicine is for specific symptoms and specialist-led."}),
    # ---- Musculoskeletal & autoimmune
    "Osteoarthritis": ("Musculoskeletal & Autoimmune", "doctor", ["paracetamol", "topical_nsaid", "ibuprofen"], {}),
    "Rheumatoid Arthritis": ("Musculoskeletal & Autoimmune", "doctor", ["methotrexate", "ibuprofen", "prednisolone"], {}),
    "Osteoporosis": ("Musculoskeletal & Autoimmune", "doctor", ["calcium_d", "alendronate"], {"ex": ["easy_walk", "strength_band", "balance", "tai_chi"], "avoid": "Avoid high-impact moves, deep forward spine bends and twisting under load."}),
    "Gout": ("Musculoskeletal & Autoimmune", "doctor", ["colchicine", "ibuprofen", "allopurinol"], {"avoid": "Rest the joint during an attack; resume gently afterwards."}),
    "Fibromyalgia": ("Musculoskeletal & Autoimmune", "doctor", ["paracetamol", "duloxetine"], {"ex": ["easy_walk", "swim", "yoga", "tai_chi", "gentle_stretch"], "avoid": "Start very low and progress slowly; avoid boom-and-bust activity."}),
    "Scoliosis": ("Musculoskeletal & Autoimmune", "doctor", [], {"ex": ["gentle_stretch", "swim", "kegel", "yoga", "rom"], "note": "Management is observation, bracing, physiotherapy or surgery depending on curve severity."}),
    "Osteomyelitis": ("Musculoskeletal & Autoimmune", "doctor", [], {"max_int": 0, "ex": ["rest"], "note": "Bone infection needs prolonged, usually IV, antibiotics under specialist care."}),
    "Muscular Dystrophy": ("Musculoskeletal & Autoimmune", "doctor", ["prednisolone"], {"max_int": 1, "ex": ["rom", "swim", "gentle_stretch", "chair"], "avoid": "Avoid eccentric / high-resistance exercise that damages muscle."}),
    "Systemic Lupus Erythematosus (SLE)": ("Musculoskeletal & Autoimmune", "doctor", ["hydroxychloroquine", "prednisolone"], {"avoid": "Avoid midday sun exposure; rest during flares."}),
    # ---- Skin
    "Acne": ("Skin", "self_care", ["benzoyl", "adapalene", "doxycycline"], {}),
    "Eczema": ("Skin", "self_care", ["emollient", "hc_cream", "cetirizine"], {}),
    "Psoriasis": ("Skin", "doctor", ["emollient", "calcipotriol", "methotrexate"], {}),
    "Melanoma": _CANCER,
    # ---- Eye & ear
    "Cataracts": ("Eye & Ear", "doctor", [], {"note": "Treatment for visually significant cataracts is surgery."}),
    "Glaucoma": ("Eye & Ear", "doctor", ["latanoprost", "timolol"], {}),
    "Conjunctivitis (Pink Eye)": ("Eye & Ear", "self_care", ["artificial_tears", "eye_antibiotic"], {"max_int": 1}),
    "Otitis Media (Ear Infection)": ("Eye & Ear", "doctor", ["paracetamol", "ibuprofen", "amoxicillin"], {"max_int": 1}),
    # ---- Infectious disease
    "Chickenpox": ("Infectious Disease", "doctor", ["paracetamol", "calamine", "cetirizine", "acyclovir"], {"note": "Avoid ibuprofen and aspirin in chickenpox (skin and Reye-syndrome risks)."}),
    "Measles": ("Infectious Disease", "doctor", ["paracetamol", "vitamin_a", "ors"], {}),
    "Mumps": ("Infectious Disease", "doctor", ["paracetamol", "ibuprofen"], {}),
    "Rubella": ("Infectious Disease", "doctor", ["paracetamol"], {"note": "Dangerous in pregnancy - pregnant contacts should contact a doctor."}),
    "Polio": _EMERG("Infectious Disease", "Polio has no cure; it needs urgent hospital care. Vaccination is the prevention."),
    "Dengue Fever": ("Infectious Disease", "doctor", ["paracetamol", "ors"], {"max_int": 0, "ex": ["rest"], "note": "Use paracetamol only - avoid ibuprofen and aspirin (bleeding risk). Watch for warning signs and seek care."}),
    "Zika Virus": ("Infectious Disease", "doctor", ["paracetamol", "ors"], {"note": "Avoid NSAIDs until dengue is ruled out. Pregnant women need specialist advice."}),
    "Ebola Virus": _EMERG("Infectious Disease"),
    "Rabies": _EMERG("Infectious Disease", "After any animal bite, wash the wound and get post-exposure prophylaxis immediately; rabies is fatal once symptoms start."),
    "HIV/AIDS": ("Infectious Disease", "doctor", ["art"], {"max_int": 2}),
    "Malaria": ("Infectious Disease", "doctor", ["antimalarial", "paracetamol"], {"max_int": 0, "ex": ["rest"], "note": "Needs a blood test and prompt treatment - severe malaria is an emergency."}),
    "Lyme Disease": ("Infectious Disease", "doctor", ["doxycycline"], {}),
    "Tetanus": _EMERG("Infectious Disease"),
    "Sepsis": _EMERG("Infectious Disease", "Fever with confusion, fast breathing or very low BP is an emergency - call for help now."),
    # ---- Cancer
    "Bladder Cancer": _CANCER, "Brain Tumor": _CANCER, "Breast Cancer": _CANCER, "Colorectal Cancer": _CANCER,
    "Esophageal Cancer": _CANCER, "Kidney Cancer": _CANCER, "Liver Cancer": _CANCER, "Lung Cancer": _CANCER,
    "Lymphoma": _CANCER, "Ovarian Cancer": _CANCER, "Pancreatic Cancer": _CANCER, "Prostate Cancer": _CANCER,
    "Testicular Cancer": _CANCER, "Thyroid Cancer": _CANCER,
    # ---- Blood & genetic
    "Anemia": ("Blood & Genetic", "doctor", ["iron", "folic_acid", "b12"], {"max_int": 1, "note": "The right supplement depends on the type of anaemia - get a blood test first."}),
    "Hemophilia": ("Blood & Genetic", "doctor", ["factor"], {"ex": ["swim", "easy_walk", "gentle_stretch", "rom"], "max_int": 2, "avoid": "No contact sports; avoid aspirin/NSAIDs unless prescribed."}),
    "Sickle Cell Anemia": ("Blood & Genetic", "doctor", ["hydroxyurea", "folic_acid"], {"avoid": "Avoid extreme heat, cold, altitude and dehydration."}),
    "Down Syndrome": ("Blood & Genetic", "doctor", [], {"max_int": 3, "note": "Care is supportive and multidisciplinary (cardiac, thyroid, hearing and vision checks)."}),
    "Klinefelter Syndrome": ("Blood & Genetic", "doctor", [], {"max_int": 3, "note": "Hormone therapy (testosterone) is specialist-led."}),
    "Marfan Syndrome": ("Blood & Genetic", "doctor", [], {"max_int": 1, "avoid": "Avoid contact sports, heavy lifting and isometric straining (aortic risk).", "note": "Aortic monitoring and any protective medicines are specialist-led."}),
    "Prader-Willi Syndrome": ("Blood & Genetic", "doctor", [], {"max_int": 3, "note": "Care focuses on nutrition control, growth-hormone therapy and supervised activity."}),
    "Turner Syndrome": ("Blood & Genetic", "doctor", [], {"max_int": 3, "note": "Growth and sex-hormone therapy are specialist-led."}),
    "Williams Syndrome": ("Blood & Genetic", "doctor", [], {"max_int": 2, "note": "Needs cardiac and calcium monitoring by a specialist."}),
    # ---- Added for the 132-symptom open dataset (data/external/Training.csv)
    "Allergy": ("Skin", "self_care", ["cetirizine", "loratadine", "hc_cream"], {"max_int": 3, "note": "Swelling of lips/face, wheeze or faintness after exposure is an emergency - seek urgent care."}),
    "Drug Reaction": ("Skin", "doctor", [], {"max_int": 0, "ex": ["rest"], "note": "Do not stop or restart prescribed medicines on your own - contact a doctor promptly. Swelling, breathing trouble, blistering skin or fever needs emergency care."}),
    "Fungal Infection": ("Skin", "self_care", ["antifungal"], {}),
    "Impetigo": ("Skin", "doctor", ["mupirocin"], {"note": "Contagious - keep sores covered and do not share towels."}),
    "Cervical Spondylosis": ("Musculoskeletal & Autoimmune", "doctor", ["paracetamol", "topical_nsaid", "ibuprofen"], {"ex": ["neck_mobility", "gentle_stretch", "easy_walk", "swim"], "avoid": "Avoid heavy overhead lifting and long periods of neck bending; seek care for arm weakness or numbness."}),
    "Arthritis": ("Musculoskeletal & Autoimmune", "doctor", ["paracetamol", "topical_nsaid", "ibuprofen"], {"note": "'Arthritis' covers many types - a doctor should identify which one (e.g. osteo- vs rheumatoid)."}),
    "GERD": ("Digestive & Liver", "self_care", ["antacid", "ppi"], {"ex": ["easy_walk", "yoga", "diaphragmatic"], "avoid": "Avoid intense exercise within 2-3 hours of eating and bending/inverted poses."}),
    "Peptic Ulcer Disease": ("Digestive & Liver", "doctor", ["ppi", "h_pylori"], {"note": "Avoid NSAIDs (ibuprofen, aspirin) and alcohol. Black stools or vomiting blood is an emergency."}),
    "Jaundice": ("Digestive & Liver", "doctor", [], {"max_int": 1, "note": "Jaundice is a sign of an underlying liver, bile or blood problem and needs tests. Avoid alcohol and unnecessary medicines."}),
    "Chronic Cholestasis": ("Digestive & Liver", "doctor", [], {"max_int": 1, "note": "Specialist (hepatology) care; avoid alcohol."}),
    "Alcoholic Hepatitis": ("Digestive & Liver", "doctor", [], {"max_int": 1, "note": "Stopping alcohol is essential; treatment is specialist / hospital-led."}),
    "Hepatitis A": ("Digestive & Liver", "doctor", [], {"max_int": 1, "note": "Supportive care (rest, fluids, no alcohol). Vaccination prevents it."}),
    "Hepatitis C": ("Digestive & Liver", "doctor", ["daa"], {"max_int": 2}),
    "Hepatitis D": ("Digestive & Liver", "doctor", [], {"max_int": 1, "note": "Occurs together with hepatitis B - specialist-led."}),
    "Hepatitis E": ("Digestive & Liver", "doctor", [], {"max_int": 1, "note": "Usually supportive care; pregnant women need urgent medical advice."}),
    "Varicose Veins": ("Cardiovascular", "doctor", [], {"ex": ["easy_walk", "cycling", "swim", "gentle_stretch"], "note": "Compression stockings and leg elevation are typical first steps; see a vascular specialist if painful or ulcerated."}),
    "Vertigo (BPPV)": ("Neurological", "doctor", [], {"max_int": 1, "ex": ["balance", "gentle_stretch", "easy_walk"], "avoid": "Avoid sudden head movements, ladders and driving during episodes.", "note": "BPPV is usually treated with repositioning manoeuvres (e.g. Epley) performed by a clinician."}),
}

SEVERITY_TEXT = {
    "self_care": ("info", "Often manageable at home - see a doctor if symptoms are severe, last more than a few days, or worsen."),
    "doctor": ("warning", "This condition needs a doctor's diagnosis and a prescription-based plan."),
    "emergency": ("error", "Potential medical emergency - seek urgent medical care."),
}


def all_diseases() -> list[str]:
    return sorted(_D)


def get_info(disease: str) -> dict:
    """Return the structured knowledge-base entry for a disease."""
    if disease not in _D:
        raise KeyError(f"'{disease}' is not in the knowledge base")
    cat, sev, med_keys, ov = _D[disease]
    meta = _CAT[cat]
    ex_keys = ov.get("ex", meta["ex"])
    return {
        "name": disease,
        "category": cat,
        "severity": sev,
        "medicines": [
            dict(key=k, name=_M[k][0], drug_class=_M[k][1], use=_M[k][2], rx=_M[k][3], avoid=_M[k][4])
            for k in med_keys
        ],
        "note": ov.get("note", SPECIALIST_NOTE if not med_keys else ""),
        "exercises": [
            dict(key=k, name=_EX[k][0], intensity=_EX[k][1], minutes=_EX[k][2], benefit=_EX[k][3])
            for k in ex_keys
        ],
        "max_intensity": ov.get("max_int", meta["max_int"]),
        "avoid_exercise": ov.get("avoid", meta["avoid"]),
    }


# --------------------------------------------------------------------------------------
# Doctor specialties (used by the doctor finder)
# --------------------------------------------------------------------------------------
SPECIALTIES = [
    "General Physician", "Pulmonologist", "Cardiologist", "Endocrinologist", "Gastroenterologist", "Hepatologist",
    "Nephrologist", "Urologist", "Neurologist", "Psychiatrist", "Rheumatologist", "Orthopedist", "Dermatologist",
    "Ophthalmologist", "ENT Specialist", "Infectious Disease Specialist", "Oncologist", "Hematologist",
    "Gynecologist", "Pediatrician", "Vascular Surgeon", "Emergency Medicine", "Physiotherapist",
]
_SPEC_CAT = {
    "Respiratory": "Pulmonologist", "Cardiovascular": "Cardiologist", "Metabolic & Endocrine": "Endocrinologist",
    "Digestive & Liver": "Gastroenterologist", "Kidney & Urinary": "Nephrologist", "Neurological": "Neurologist",
    "Mental Health": "Psychiatrist", "Musculoskeletal & Autoimmune": "Rheumatologist", "Skin": "Dermatologist",
    "Eye & Ear": "Ophthalmologist", "Infectious Disease": "Infectious Disease Specialist", "Cancer": "Oncologist",
    "Blood & Genetic": "Hematologist", "Reproductive": "Gynecologist",
}
_HEPATIC = ["Hepatitis", "Hepatitis A", "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E", "Cirrhosis",
            "Liver Disease", "Jaundice", "Chronic Cholestasis", "Alcoholic Hepatitis"]
_SPEC_OV = {
    **{d: ["Hepatologist", "Gastroenterologist"] for d in _HEPATIC},
    "Common Cold": ["General Physician"], "Influenza": ["General Physician"],
    **{d: ["ENT Specialist"] for d in ["Tonsillitis", "Sinusitis", "Allergic Rhinitis", "Otitis Media (Ear Infection)"]},
    "Vertigo (BPPV)": ["ENT Specialist", "Neurologist"],
    "Urinary Tract Infection": ["Urologist"],
    **{d: ["Orthopedist"] for d in ["Osteoarthritis", "Osteoporosis", "Cervical Spondylosis", "Scoliosis", "Osteomyelitis"]},
    "Varicose Veins": ["Vascular Surgeon"],
    "Hypertension": ["Cardiologist"], "Allergy": ["Dermatologist"],
    **{d: ["Pediatrician"] for d in ["Down Syndrome", "Klinefelter Syndrome", "Turner Syndrome", "Williams Syndrome",
                                     "Prader-Willi Syndrome", "Marfan Syndrome", "Autism Spectrum Disorder (ASD)"]},
    "Cerebral Palsy": ["Neurologist", "Pediatrician"], "Spina Bifida": ["Neurologist", "Pediatrician"],
    "Cystic Fibrosis": ["Pulmonologist", "Pediatrician"], "Sleep Apnea": ["Pulmonologist", "ENT Specialist"],
    "Lyme Disease": ["Infectious Disease Specialist"],
}
_REHAB = {"Chronic Obstructive Pulmonary Disease (COPD)", "Cerebral Palsy", "Spina Bifida", "Muscular Dystrophy",
          "Scoliosis", "Cystic Fibrosis"}


def specialties_for(disease: str) -> list[str]:
    """Ordered doctor specialties to consult (most relevant first)."""
    cat, sev, _, _ = _D[disease]
    if sev == "emergency":
        return ["Emergency Medicine"]
    main = list(_SPEC_OV.get(disease, [_SPEC_CAT[cat]]))
    if "General Physician" not in main:
        main = ["General Physician"] + main if sev == "self_care" else main + ["General Physician"]
    if cat in ("Musculoskeletal & Autoimmune", "Neurological") or disease in _REHAB:
        main.append("Physiotherapist")
    return main
