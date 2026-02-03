import joblib
import pandas as pd
# Load models
model = joblib.load('artifacts/models/sodium_sensitivity.joblib')
imputer = joblib.load('artifacts/models/imputer.joblib')
scaler = joblib.load('artifacts/models/scaler.joblib')
features = joblib.load('artifacts/models/feature_names.joblib')
# Predict for a patient
patient = {
    'age': 55,
    'sex_male': 1,
    'has_htn': 1,
    'has_dm': 1,
    'has_ckd': 1,
    'serum_sodium': 164,
    'serum_potassium': 5.6,
    'creatinine': 2.3,
    'egfr': 28,
    'hba1c': 8.4,
    'fbs': 170,
    'sbp': 152,
    'dbp': 94,
    'bmi': 29
}
X = pd.DataFrame([patient])[features]
X = scaler.transform(imputer.transform(X))
prediction = model.predict(X)  # 0=low, 1=moderate, 2=high

# Print result
label_map = {0: 'low', 1: 'moderate', 2: 'high'}
print(f"Sodium sensitivity risk: {label_map[prediction[0]]}")