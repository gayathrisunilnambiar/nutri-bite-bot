import joblib
import pandas as pd

# Load models
sodium_model = joblib.load('artifacts/models/sodium_sensitivity.joblib')
potassium_model = joblib.load('artifacts/models/potassium_sensitivity.joblib')
protein_model = joblib.load('artifacts/models/protein_restriction.joblib')
carb_model = joblib.load('artifacts/models/carb_sensitivity.joblib')
imputer = joblib.load('artifacts/models/imputer.joblib')
scaler = joblib.load('artifacts/models/scaler.joblib')
features = joblib.load('artifacts/models/feature_names.joblib')

# Test patient with diabetes risk factors
patient = {
    'age': 55,
    'sex_male': 1,
    'has_htn': 1,
    'has_dm': 1,        # Has diabetes
    'has_ckd': 1,
    'serum_sodium': 144,
    'serum_potassium': 5.6,
    'creatinine': 2.3,
    'egfr': 28,
    'hba1c': 8.4,       # High HbA1c (diabetic)
    'fbs': 170,         # High fasting blood sugar
    'sbp': 152,
    'dbp': 94,
    'bmi': 32           # Obese
}

# Prepare features
X = pd.DataFrame([patient])[features]
X = scaler.transform(imputer.transform(X))

# Get predictions
label_map = {0: 'low', 1: 'moderate', 2: 'high'}

models = {
    'sodium_sensitivity': sodium_model,
    'potassium_sensitivity': potassium_model,
    'protein_restriction': protein_model,
    'carb_sensitivity': carb_model
}

print("=" * 50)
print("CLINICAL RISK STRATIFICATION RESULTS")
print("=" * 50)
print(f"\nPatient: Age {patient['age']}, Diabetes: Yes, CKD: Yes")
print(f"HbA1c: {patient['hba1c']}%, FBS: {patient['fbs']} mg/dL, BMI: {patient['bmi']}")
print("\nRisk Levels:")
print("-" * 50)

for target, model in models.items():
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    print(f"  {target:25s}: {label_map[pred]:10s} ({proba[pred]*100:.1f}% confidence)")

print("=" * 50)