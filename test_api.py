"""Test script for Nutri-Bite Bot API"""
import requests
import json

data = {
    'lab_values': {'egfr': 45, 'creatinine': 2.1, 'potassium': 5.2, 'sodium': 142, 'glucose': 180, 'hba1c': 7.5},
    'conditions': {'diabetes_t1': True, 'hypertension': True, 'ckd': True},
    'ingredients_text': 'potato, apple, chicken breast, spinach, banana'
}

resp = requests.post('http://localhost:5000/analyze', json=data)
result = resp.json()

print('=== CBC Analysis ===')
print(f"CKD Stage: {result['cbc_analysis']['ckd_stage']}")
print(f"Alerts: {len(result['cbc_analysis']['alerts'])}")
for a in result['cbc_analysis']['alerts'][:3]:
    print(f"  [{a['level']}] {a['message']}")

print()
print('=== Daily Limits ===')
print(f"Potassium: {result['daily_limits']['potassium_mg']} mg/day")
print(f"Sodium: {result['daily_limits']['sodium_mg']} mg/day")

print()
print('=== Recommendations ===')
for r in result['recommendations']:
    status = r['status'].upper()
    name = r['name']
    max_g = r['max_allowed_g']
    print(f"  {status:10} {name:15} max {max_g:.0f}g")
