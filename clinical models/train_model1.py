"""
Clinical Model #1 Training Script
Train risk stratification model on MIMIC-IV EventLog and ActivityAttributes data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
import joblib
import argparse

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# ===============================
# Configuration
# ===============================

@dataclass
class ClinicalModelConfig:
    random_state: int = 42
    test_size: float = 0.2
    
    feature_cols: Tuple[str, ...] = (
        "age",
        "sex_male",
        "has_htn",
        "has_dm",
        "has_ckd",
        "serum_sodium",
        "serum_potassium",
        "creatinine",
        "egfr",
        "hba1c",
        "fbs",
        "sbp",
        "dbp",
        "bmi",
    )
    
    targets: Tuple[str, ...] = (
        "sodium_sensitivity",
        "potassium_sensitivity",
        "protein_restriction",
    )
    
    lgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": 0.05,
        "n_estimators": 600,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
    })
    
    model_dir: Path = Path("../artifacts/models")
    data_dir: Path = Path("../mimic IV")


# ===============================
# Data Loader
# ===============================

class MIMICDataLoader:
    """Load MIMIC-IV EventLog and ActivityAttributes with chunked reading"""
    
    def __init__(self, config: ClinicalModelConfig, sample_rows: int = 100000):
        self.config = config
        self.sample_rows = sample_rows
        self.eventlog = None
        self.activity_attrs = None
        
    def load_data(self) -> bool:
        """Load data files with sampling for initial testing"""
        print("\n" + "="*80)
        print("LOADING MIMIC-IV DATA")
        print("="*80)
        
        # Load EventLog
        eventlog_path = self.config.data_dir / "B_EventLog.csv"
        print(f"\nLoading EventLog (sample={self.sample_rows})...")
        self.eventlog = pd.read_csv(eventlog_path, nrows=self.sample_rows)
        print(f"  ✓ EventLog shape: {self.eventlog.shape}")
        print(f"  ✓ Activities: {self.eventlog['Activity'].nunique()} unique")
        
        # Load ActivityAttributes
        attrs_path = self.config.data_dir / "E_ActivityAttributes.csv"
        print(f"\nLoading ActivityAttributes (sample={self.sample_rows})...")
        self.activity_attrs = pd.read_csv(attrs_path, nrows=self.sample_rows)
        print(f"  ✓ ActivityAttributes shape: {self.activity_attrs.shape}")
        print(f"  ✓ Attributes: {self.activity_attrs['Activity_Attribute'].nunique()} unique")
        
        return True


# ===============================
# Feature Extractor
# ===============================

class ClinicalFeatureExtractor:
    """Extract clinical features from event log and activity attributes"""
    
    def __init__(self, config: ClinicalModelConfig):
        self.config = config
        
    def extract_features(self, eventlog: pd.DataFrame, activity_attrs: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from the data"""
        print("\n" + "="*80)
        print("EXTRACTING CLINICAL FEATURES")
        print("="*80)
        
        # Get unique patient-encounter pairs
        patient_cols = ['temp_patient_id', 'temp_encounter_id']
        
        # Merge eventlog with activity attributes
        merged = eventlog.merge(
            activity_attrs,
            on='Activity_Attributes_ID',
            how='left',
            suffixes=('', '_attr')
        )
        
        # Pivot vital signs
        vitals = merged[merged['Activity_y'] == 'vitalsign'].copy() if 'Activity_y' in merged.columns else \
                 merged[merged['Activity_attr'] == 'vitalsign'].copy()
        
        if len(vitals) == 0:
            # Try with Activity column
            vitals = merged[merged['Activity'].str.contains('vital|BP', case=False, na=False)].copy()
        
        print(f"  ✓ Found {len(vitals)} vital sign records")
        
        # Extract vital features per patient-encounter
        features_df = self._aggregate_vital_features(merged, patient_cols)
        
        # Add synthetic demographics and disease flags (for demo purposes)
        features_df = self._add_synthetic_demographics(features_df)
        
        print(f"\n  ✓ Final feature matrix: {features_df.shape}")
        
        return features_df
    
    def _aggregate_vital_features(self, df: pd.DataFrame, patient_cols: List[str]) -> pd.DataFrame:
        """Aggregate vital sign measurements per patient-encounter"""
        
        # Create pivot for vital signs
        vital_attrs = ['sbp', 'dbp', 'mbp', 'heart_rate', 'glucose', 'temperature', 'resp_rate', 'spo2']
        
        records = []
        attr_col = 'Activity_Attribute' if 'Activity_Attribute' in df.columns else 'Activity_Attribute_x'
        value_col = 'Activity_Attribute_Value' if 'Activity_Attribute_Value' in df.columns else 'Activity_Attribute_Value_x'
        
        # Group by patient-encounter
        for (patient_id, encounter_id), group in df.groupby(patient_cols):
            record = {
                'temp_patient_id': patient_id,
                'temp_encounter_id': encounter_id,
            }
            
            # Get vital signs
            for attr in vital_attrs:
                attr_values = group[group[attr_col] == attr][value_col]
                if len(attr_values) > 0:
                    try:
                        record[f'{attr}_mean'] = pd.to_numeric(attr_values, errors='coerce').mean()
                    except:
                        record[f'{attr}_mean'] = np.nan
            
            records.append(record)
        
        features_df = pd.DataFrame(records)
        print(f"  ✓ Extracted features for {len(features_df)} patient-encounters")
        
        return features_df
    
    def _add_synthetic_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add synthetic demographics for demo (replace with real data in production)"""
        np.random.seed(42)
        n = len(df)
        
        # Demographics
        df['age'] = np.random.normal(65, 15, n).clip(18, 100).astype(int)
        df['sex_male'] = np.random.binomial(1, 0.52, n)
        
        # Disease flags (simulated based on clinical patterns)
        df['has_htn'] = np.random.binomial(1, 0.4, n)
        df['has_dm'] = np.random.binomial(1, 0.25, n)
        df['has_ckd'] = np.random.binomial(1, 0.15, n)
        
        # Lab values (synthetic, correlated with disease status)
        df['serum_sodium'] = np.where(
            df['has_htn'] == 1,
            np.random.normal(142, 5, n),
            np.random.normal(140, 3, n)
        ).clip(130, 155)
        
        df['serum_potassium'] = np.where(
            df['has_ckd'] == 1,
            np.random.normal(5.0, 0.8, n),
            np.random.normal(4.2, 0.5, n)
        ).clip(3.0, 7.0)
        
        df['creatinine'] = np.where(
            df['has_ckd'] == 1,
            np.random.normal(2.5, 1.0, n),
            np.random.normal(1.0, 0.3, n)
        ).clip(0.5, 8.0)
        
        # Calculate eGFR using CKD-EPI formula approximation
        df['egfr'] = 142 * (df['creatinine'] / 0.9) ** (-1.2) * (0.9938 ** df['age'])
        df['egfr'] = df['egfr'].clip(5, 120)
        
        df['hba1c'] = np.where(
            df['has_dm'] == 1,
            np.random.normal(8.5, 1.5, n),
            np.random.normal(5.5, 0.5, n)
        ).clip(4.0, 14.0)
        
        df['fbs'] = np.where(
            df['has_dm'] == 1,
            np.random.normal(180, 50, n),
            np.random.normal(95, 15, n)
        ).clip(60, 400)
        
        # Use actual sbp/dbp if available, else synthetic
        if 'sbp_mean' in df.columns:
            df['sbp'] = df['sbp_mean'].fillna(np.random.normal(130, 20, n).clip(80, 200))
        else:
            df['sbp'] = np.random.normal(130, 20, n).clip(80, 200)
            
        if 'dbp_mean' in df.columns:
            df['dbp'] = df['dbp_mean'].fillna(np.random.normal(80, 12, n).clip(50, 120))
        else:
            df['dbp'] = np.random.normal(80, 12, n).clip(50, 120)
        
        df['bmi'] = np.random.normal(28, 6, n).clip(16, 50)
        
        print(f"  ✓ Added demographics for {n} patients")
        
        return df


# ===============================
# Label Generator
# ===============================

class LabelGenerator:
    """Generate nutrition risk labels from clinical features"""
    
    def __init__(self, config: ClinicalModelConfig):
        self.config = config
        
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate multi-label targets based on clinical thresholds"""
        print("\n" + "="*80)
        print("GENERATING CLINICAL RISK LABELS")
        print("="*80)
        
        df = df.copy()
        
        # 1. Sodium sensitivity
        print("\n1. Sodium Sensitivity:")
        df['sodium_sensitivity'] = 0
        
        # Moderate: high sodium OR hypertension
        mask_moderate = (df['serum_sodium'] > 145) | (df['has_htn'] == 1)
        df.loc[mask_moderate, 'sodium_sensitivity'] = 1
        
        # High: very high sodium OR (hypertension + high BP)
        mask_high = (df['serum_sodium'] > 150) | ((df['has_htn'] == 1) & (df['sbp'] > 160))
        df.loc[mask_high, 'sodium_sensitivity'] = 2
        
        counts = df['sodium_sensitivity'].value_counts().sort_index()
        print(f"   Low: {counts.get(0, 0)}, Moderate: {counts.get(1, 0)}, High: {counts.get(2, 0)}")
        
        # 2. Potassium sensitivity
        print("\n2. Potassium Sensitivity:")
        df['potassium_sensitivity'] = 0
        
        # Moderate: high K OR reduced eGFR
        mask_moderate = (df['serum_potassium'] > 4.5) | (df['egfr'] < 60)
        df.loc[mask_moderate, 'potassium_sensitivity'] = 1
        
        # High: very high K OR (CKD + severely reduced eGFR)
        mask_high = (df['serum_potassium'] > 5.0) | ((df['has_ckd'] == 1) & (df['egfr'] < 30))
        df.loc[mask_high, 'potassium_sensitivity'] = 2
        
        counts = df['potassium_sensitivity'].value_counts().sort_index()
        print(f"   Low: {counts.get(0, 0)}, Moderate: {counts.get(1, 0)}, High: {counts.get(2, 0)}")
        
        # 3. Protein restriction
        print("\n3. Protein Restriction:")
        df['protein_restriction'] = 0
        
        # Moderate: reduced eGFR
        mask_moderate = df['egfr'] < 60
        df.loc[mask_moderate, 'protein_restriction'] = 1
        
        # High: severely reduced eGFR OR CKD OR high creatinine
        mask_high = (df['egfr'] < 30) | (df['has_ckd'] == 1) | (df['creatinine'] > 2.0)
        df.loc[mask_high, 'protein_restriction'] = 2
        
        counts = df['protein_restriction'].value_counts().sort_index()
        print(f"   Low: {counts.get(0, 0)}, Moderate: {counts.get(1, 0)}, High: {counts.get(2, 0)}")
        
        return df


# ===============================
# Monotonic Constraints
# ===============================

def build_monotonic_constraints(features: List[str]) -> List[int]:
    """
    +1 : higher value => higher dietary restriction risk
    -1 : higher value => lower risk
     0 : no constraint
    """
    constraints = []
    for f in features:
        f_lower = f.lower()
        if f_lower in {"creatinine", "serum_potassium", "hba1c", "fbs", "sbp", "dbp", "bmi"}:
            constraints.append(+1)
        elif f_lower in {"egfr"}:
            constraints.append(-1)
        elif f_lower in {"serum_sodium"}:
            constraints.append(+1)
        else:
            constraints.append(0)
    return constraints


# ===============================
# Clinical Risk Stratifier
# ===============================

class ClinicalRiskStratifier:
    """Train and evaluate clinical risk models"""
    
    def __init__(self, cfg: ClinicalModelConfig):
        self.cfg = cfg
        self.models: Dict[str, lgb.LGBMClassifier] = {}
        self.imputer = None
        self.scaler = None
        
    def fit(self, df: pd.DataFrame):
        """Train models on labeled data"""
        print("\n" + "="*80)
        print("TRAINING MODELS")
        print("="*80)
        
        # Prepare features
        feature_cols = list(self.cfg.feature_cols)
        available_cols = [c for c in feature_cols if c in df.columns]
        
        print(f"\nUsing {len(available_cols)} features: {available_cols}")
        
        X = df[available_cols].copy()
        Y = df[list(self.cfg.targets)].copy()
        
        # Impute missing values
        self.imputer = SimpleImputer(strategy="median")
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Build monotonic constraints
        mono = build_monotonic_constraints(available_cols)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, Y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=Y[self.cfg.targets[0]],
        )
        
        print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Train model for each target
        for target in self.cfg.targets:
            print(f"\n--- Training: {target} ---")
            
            model = lgb.LGBMClassifier(**self.cfg.lgbm_params)
            model.set_params(monotone_constraints=mono)
            model.fit(X_train, y_train[target])
            
            self.models[target] = model
            
            # Evaluate
            preds = model.predict(X_test)
            print(classification_report(
                y_test[target], preds,
                target_names=['Low', 'Moderate', 'High'],
                zero_division=0
            ))
        
        # Store feature names for prediction
        self.feature_names = available_cols
        
    def predict(self, clinical_input: Dict[str, Any]) -> Dict[str, str]:
        """Get risk predictions for a patient"""
        X = pd.DataFrame([clinical_input])[self.feature_names]
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        
        label_map = {0: "low", 1: "moderate", 2: "high"}
        
        return {
            target: label_map[int(model.predict(X)[0])]
            for target, model in self.models.items()
        }
    
    def save(self):
        """Save trained models"""
        self.cfg.model_dir.mkdir(parents=True, exist_ok=True)
        
        for target, model in self.models.items():
            path = self.cfg.model_dir / f"{target}.joblib"
            joblib.dump(model, path)
            print(f"  ✓ Saved: {path}")
        
        joblib.dump(self.imputer, self.cfg.model_dir / "imputer.joblib")
        joblib.dump(self.scaler, self.cfg.model_dir / "scaler.joblib")
        joblib.dump(self.feature_names, self.cfg.model_dir / "feature_names.joblib")
        
        print(f"\n  ✓ All models saved to: {self.cfg.model_dir}")


# ===============================
# Main Training Pipeline
# ===============================

def main(sample_rows: int = 100000):
    """Execute training pipeline"""
    
    print("\n" + "="*80)
    print("CLINICAL RISK STRATIFICATION - MODEL #1")
    print(f"Training on {sample_rows:,} sample rows")
    print("="*80)
    
    # Initialize
    config = ClinicalModelConfig()
    
    # 1. Load data
    loader = MIMICDataLoader(config, sample_rows=sample_rows)
    loader.load_data()
    
    # 2. Extract features
    extractor = ClinicalFeatureExtractor(config)
    features = extractor.extract_features(loader.eventlog, loader.activity_attrs)
    
    if len(features) < 10:
        print("✗ Not enough patient records to train. Exiting.")
        return
    
    # 3. Generate labels
    labeler = LabelGenerator(config)
    labeled_data = labeler.generate_labels(features)
    
    # 4. Train models
    model = ClinicalRiskStratifier(config)
    model.fit(labeled_data)
    
    # 5. Save models
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    model.save()
    
    # 6. Test prediction
    print("\n" + "="*80)
    print("TEST PREDICTION")
    print("="*80)
    
    test_patient = {
        "age": 55,
        "sex_male": 1,
        "has_htn": 1,
        "has_dm": 1,
        "has_ckd": 1,
        "serum_sodium": 144,
        "serum_potassium": 5.6,
        "creatinine": 2.3,
        "egfr": 28,
        "hba1c": 8.4,
        "fbs": 170,
        "sbp": 152,
        "dbp": 94,
        "bmi": 29,
    }
    
    predictions = model.predict(test_patient)
    print(f"\nTest patient predictions:")
    for target, risk in predictions.items():
        print(f"  {target}: {risk}")
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Clinical Risk Model #1")
    parser.add_argument("--sample", type=int, default=100000,
                        help="Number of rows to sample (default: 100000)")
    args = parser.parse_args()
    
    main(sample_rows=args.sample)
