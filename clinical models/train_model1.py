"""
Clinical Model #1 Training Script
Train risk stratification model on MIMIC-IV EventLog and ActivityAttributes data.
Includes condition-specific nutrient thresholds and accuracy analysis reports.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
import joblib
import argparse
import json

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, cohen_kappa_score,
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


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
        "carb_sensitivity",
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
    reports_dir: Path = Path("../artifacts/models/reports")
    data_dir: Path = Path("../mimic IV")


# ===============================
# Nutrient Threshold Engine
# ===============================

class NutrientThresholdEngine:
    """
    Determine condition-specific permissible daily nutrient amounts
    based on KDIGO 2024, ADA Standards of Care 2024, and AHA/ACC guidelines.
    """

    # Thresholds keyed by (has_htn, has_dm, ckd_stage)
    # ckd_stage: 0=none, 3=stage3, 4=stage4, 5=stage5/dialysis
    THRESHOLDS = {
        # --- No CKD ---
        (0, 0, 0): {  # Healthy
            "sodium_mg":      {"max": 2300, "unit": "mg/day", "rationale": "General healthy limit (AHA)"},
            "potassium_mg":   {"min": 2600, "max": 3400, "unit": "mg/day", "rationale": "Adequate intake range"},
            "protein_g_per_kg":{"min": 0.8,  "max": 1.0,  "unit": "g/kg/day", "rationale": "RDA for healthy adults"},
            "carbs_g":        {"min": 225,  "max": 325,  "unit": "g/day", "rationale": "45-65% of 2000 kcal diet"},
            "phosphorus_mg":  {"max": 1250, "unit": "mg/day", "rationale": "RDA upper range"},
            "fluid_ml":       {"min": 2000, "max": 2500, "unit": "mL/day", "rationale": "Standard hydration"},
        },
        (1, 0, 0): {  # HTN only
            "sodium_mg":      {"max": 1500, "unit": "mg/day", "rationale": "Strict sodium limit for HTN (AHA/ACC)"},
            "potassium_mg":   {"min": 3500, "max": 4700, "unit": "mg/day", "rationale": "DASH diet target — higher K helps lower BP"},
            "protein_g_per_kg":{"min": 0.8,  "max": 1.0,  "unit": "g/kg/day", "rationale": "Normal protein intake"},
            "carbs_g":        {"min": 225,  "max": 325,  "unit": "g/day", "rationale": "Normal carb intake"},
            "phosphorus_mg":  {"max": 1250, "unit": "mg/day", "rationale": "RDA upper range"},
            "fluid_ml":       {"min": 2000, "max": 2500, "unit": "mL/day", "rationale": "Standard hydration"},
        },
        (0, 1, 0): {  # DM only
            "sodium_mg":      {"max": 2300, "unit": "mg/day", "rationale": "Standard limit (ADA)"},
            "potassium_mg":   {"min": 2600, "max": 3400, "unit": "mg/day", "rationale": "Adequate intake"},
            "protein_g_per_kg":{"min": 0.8,  "max": 1.0,  "unit": "g/kg/day", "rationale": "Normal protein (ADA)"},
            "carbs_g":        {"min": 130,  "max": 200,  "unit": "g/day", "rationale": "Reduced carbs, prefer low GI (ADA 2024)"},
            "phosphorus_mg":  {"max": 1250, "unit": "mg/day", "rationale": "RDA upper range"},
            "fluid_ml":       {"min": 2000, "max": 2500, "unit": "mL/day", "rationale": "Standard hydration"},
        },
        (1, 1, 0): {  # HTN + DM
            "sodium_mg":      {"max": 1500, "unit": "mg/day", "rationale": "Strict limit for HTN+DM (AHA/ADA)"},
            "potassium_mg":   {"min": 3500, "max": 4700, "unit": "mg/day", "rationale": "DASH diet target"},
            "protein_g_per_kg":{"min": 0.8,  "max": 0.8,  "unit": "g/kg/day", "rationale": "Conservative protein"},
            "carbs_g":        {"min": 130,  "max": 200,  "unit": "g/day", "rationale": "Controlled carbs (ADA)"},
            "phosphorus_mg":  {"max": 1250, "unit": "mg/day", "rationale": "RDA"},
            "fluid_ml":       {"min": 2000, "max": 2500, "unit": "mL/day", "rationale": "Standard hydration"},
        },
        # --- CKD Stage 3 (eGFR 30-59) ---
        (0, 0, 3): {  # CKD3 only
            "sodium_mg":      {"max": 2000, "unit": "mg/day", "rationale": "KDIGO CKD stage 3 guideline"},
            "potassium_mg":   {"max": 2000, "unit": "mg/day", "rationale": "Restricted — reduced renal clearance (KDIGO)"},
            "protein_g_per_kg":{"min": 0.6,  "max": 0.8,  "unit": "g/kg/day", "rationale": "Low-protein diet to slow progression (KDIGO)"},
            "carbs_g":        {"min": 225,  "max": 325,  "unit": "g/day", "rationale": "Normal carbs"},
            "phosphorus_mg":  {"max": 800,  "unit": "mg/day", "rationale": "Phosphorus restricted in CKD (KDIGO)"},
            "fluid_ml":       {"min": 1500, "max": 2000, "unit": "mL/day", "rationale": "Per physician guidance"},
        },
        (1, 0, 3): {  # HTN + CKD3
            "sodium_mg":      {"max": 1500, "unit": "mg/day", "rationale": "Strict for HTN+CKD (KDIGO/AHA)"},
            "potassium_mg":   {"max": 2000, "unit": "mg/day", "rationale": "Restricted for CKD3"},
            "protein_g_per_kg":{"min": 0.6,  "max": 0.8,  "unit": "g/kg/day", "rationale": "Low protein (KDIGO)"},
            "carbs_g":        {"min": 225,  "max": 325,  "unit": "g/day", "rationale": "Normal"},
            "phosphorus_mg":  {"max": 800,  "unit": "mg/day", "rationale": "Phosphorus restricted"},
            "fluid_ml":       {"min": 1000, "max": 1500, "unit": "mL/day", "rationale": "Monitor fluid balance"},
        },
        (0, 1, 3): {  # DM + CKD3
            "sodium_mg":      {"max": 2000, "unit": "mg/day", "rationale": "KDIGO guideline for DKD"},
            "potassium_mg":   {"max": 2000, "unit": "mg/day", "rationale": "Restricted for CKD3"},
            "protein_g_per_kg":{"min": 0.6,  "max": 0.8,  "unit": "g/kg/day", "rationale": "Low protein for DKD (KDIGO/ADA)"},
            "carbs_g":        {"min": 130,  "max": 200,  "unit": "g/day", "rationale": "Controlled carbs (ADA)"},
            "phosphorus_mg":  {"max": 800,  "unit": "mg/day", "rationale": "Restricted"},
            "fluid_ml":       {"min": 1000, "max": 1500, "unit": "mL/day", "rationale": "Monitor fluid"},
        },
        (1, 1, 3): {  # HTN + DM + CKD3
            "sodium_mg":      {"max": 1500, "unit": "mg/day", "rationale": "Most restrictive sodium (KDIGO/AHA/ADA)"},
            "potassium_mg":   {"max": 2000, "unit": "mg/day", "rationale": "CKD restricted"},
            "protein_g_per_kg":{"min": 0.6,  "max": 0.8,  "unit": "g/kg/day", "rationale": "Low protein"},
            "carbs_g":        {"min": 130,  "max": 200,  "unit": "g/day", "rationale": "Controlled carbs"},
            "phosphorus_mg":  {"max": 800,  "unit": "mg/day", "rationale": "Restricted"},
            "fluid_ml":       {"min": 1000, "max": 1500, "unit": "mL/day", "rationale": "Fluid restricted"},
        },
        # --- CKD Stage 4 (eGFR 15-29) ---
        (0, 0, 4): {  # CKD4 only
            "sodium_mg":      {"max": 1500, "unit": "mg/day", "rationale": "KDIGO CKD stage 4"},
            "potassium_mg":   {"max": 1500, "unit": "mg/day", "rationale": "Severely restricted (KDIGO)"},
            "protein_g_per_kg":{"min": 0.6,  "max": 0.6,  "unit": "g/kg/day", "rationale": "Very low protein (KDIGO)"},
            "carbs_g":        {"min": 225,  "max": 325,  "unit": "g/day", "rationale": "Normal carbs"},
            "phosphorus_mg":  {"max": 800,  "unit": "mg/day", "rationale": "Restricted"},
            "fluid_ml":       {"min": 1000, "max": 1500, "unit": "mL/day", "rationale": "Fluid restricted"},
        },
        (1, 0, 4): {  # HTN + CKD4
            "sodium_mg":      {"max": 1500, "unit": "mg/day", "rationale": "Strict (KDIGO/AHA)"},
            "potassium_mg":   {"max": 1500, "unit": "mg/day", "rationale": "Severely restricted"},
            "protein_g_per_kg":{"min": 0.6,  "max": 0.6,  "unit": "g/kg/day", "rationale": "Very low protein"},
            "carbs_g":        {"min": 225,  "max": 325,  "unit": "g/day", "rationale": "Normal"},
            "phosphorus_mg":  {"max": 800,  "unit": "mg/day", "rationale": "Restricted"},
            "fluid_ml":       {"min": 1000, "max": 1500, "unit": "mL/day", "rationale": "Restricted"},
        },
        (0, 1, 4): {  # DM + CKD4
            "sodium_mg":      {"max": 1500, "unit": "mg/day", "rationale": "DKD stage 4"},
            "potassium_mg":   {"max": 1500, "unit": "mg/day", "rationale": "Severely restricted"},
            "protein_g_per_kg":{"min": 0.6,  "max": 0.6,  "unit": "g/kg/day", "rationale": "Very low protein"},
            "carbs_g":        {"min": 130,  "max": 200,  "unit": "g/day", "rationale": "Controlled carbs"},
            "phosphorus_mg":  {"max": 800,  "unit": "mg/day", "rationale": "Restricted"},
            "fluid_ml":       {"min": 1000, "max": 1500, "unit": "mL/day", "rationale": "Restricted"},
        },
        (1, 1, 4): {  # HTN + DM + CKD4
            "sodium_mg":      {"max": 1500, "unit": "mg/day", "rationale": "Most restrictive (KDIGO/AHA/ADA)"},
            "potassium_mg":   {"max": 1500, "unit": "mg/day", "rationale": "Severely restricted"},
            "protein_g_per_kg":{"min": 0.6,  "max": 0.6,  "unit": "g/kg/day", "rationale": "Very low protein"},
            "carbs_g":        {"min": 130,  "max": 180,  "unit": "g/day", "rationale": "Strictly controlled carbs"},
            "phosphorus_mg":  {"max": 800,  "unit": "mg/day", "rationale": "Restricted"},
            "fluid_ml":       {"min": 1000, "max": 1500, "unit": "mL/day", "rationale": "Restricted"},
        },
        # --- CKD Stage 5 / Dialysis (eGFR <15) ---
        (0, 0, 5): {  # CKD5/dialysis
            "sodium_mg":      {"max": 1500, "unit": "mg/day", "rationale": "KDIGO dialysis guideline"},
            "potassium_mg":   {"max": 1500, "unit": "mg/day", "rationale": "Severely restricted (KDIGO)"},
            "protein_g_per_kg":{"min": 1.0,  "max": 1.2,  "unit": "g/kg/day", "rationale": "Higher protein on dialysis (KDIGO)"},
            "carbs_g":        {"min": 225,  "max": 325,  "unit": "g/day", "rationale": "Adequate energy intake"},
            "phosphorus_mg":  {"max": 800,  "unit": "mg/day", "rationale": "Restricted"},
            "fluid_ml":       {"min": 500,  "max": 1000, "unit": "mL/day", "rationale": "Very restricted on dialysis"},
        },
        (1, 0, 5): {  # HTN + CKD5
            "sodium_mg":      {"max": 1500, "unit": "mg/day", "rationale": "Dialysis guideline"},
            "potassium_mg":   {"max": 1500, "unit": "mg/day", "rationale": "Severely restricted"},
            "protein_g_per_kg":{"min": 1.0,  "max": 1.2,  "unit": "g/kg/day", "rationale": "Higher on dialysis"},
            "carbs_g":        {"min": 225,  "max": 325,  "unit": "g/day", "rationale": "Normal"},
            "phosphorus_mg":  {"max": 800,  "unit": "mg/day", "rationale": "Restricted"},
            "fluid_ml":       {"min": 500,  "max": 1000, "unit": "mL/day", "rationale": "Very restricted"},
        },
        (0, 1, 5): {  # DM + CKD5
            "sodium_mg":      {"max": 1500, "unit": "mg/day", "rationale": "DKD dialysis"},
            "potassium_mg":   {"max": 1500, "unit": "mg/day", "rationale": "Severely restricted"},
            "protein_g_per_kg":{"min": 1.0,  "max": 1.2,  "unit": "g/kg/day", "rationale": "Higher on dialysis"},
            "carbs_g":        {"min": 130,  "max": 200,  "unit": "g/day", "rationale": "Controlled carbs"},
            "phosphorus_mg":  {"max": 800,  "unit": "mg/day", "rationale": "Restricted"},
            "fluid_ml":       {"min": 500,  "max": 1000, "unit": "mL/day", "rationale": "Very restricted"},
        },
        (1, 1, 5): {  # HTN + DM + CKD5
            "sodium_mg":      {"max": 1500, "unit": "mg/day", "rationale": "Most restrictive (KDIGO/AHA/ADA)"},
            "potassium_mg":   {"max": 1500, "unit": "mg/day", "rationale": "Severely restricted"},
            "protein_g_per_kg":{"min": 1.0,  "max": 1.2,  "unit": "g/kg/day", "rationale": "Higher on dialysis"},
            "carbs_g":        {"min": 130,  "max": 180,  "unit": "g/day", "rationale": "Strictly controlled"},
            "phosphorus_mg":  {"max": 800,  "unit": "mg/day", "rationale": "Restricted"},
            "fluid_ml":       {"min": 500,  "max": 1000, "unit": "mL/day", "rationale": "Very restricted"},
        },
    }

    @staticmethod
    def get_ckd_stage(egfr: float, has_ckd: int) -> int:
        """Determine CKD stage from eGFR value."""
        if has_ckd == 0 and egfr >= 60:
            return 0
        if egfr < 15:
            return 5
        if egfr < 30:
            return 4
        if egfr < 60:
            return 3
        return 0

    def get_permissible_amounts(self, clinical_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return condition-specific permissible daily nutrient amounts.

        Parameters
        ----------
        clinical_input : dict
            Must contain: has_htn, has_dm, has_ckd, egfr

        Returns
        -------
        dict with keys: condition_profile, ckd_stage, nutrients (each with min/max/unit/rationale)
        """
        has_htn = int(clinical_input.get("has_htn", 0))
        has_dm  = int(clinical_input.get("has_dm", 0))
        has_ckd = int(clinical_input.get("has_ckd", 0))
        egfr    = float(clinical_input.get("egfr", 90))

        ckd_stage = self.get_ckd_stage(egfr, has_ckd)

        # Build condition label
        conditions = []
        if has_htn: conditions.append("HTN")
        if has_dm:  conditions.append("DM")
        if ckd_stage > 0: conditions.append(f"CKD Stage {ckd_stage}")
        condition_label = " + ".join(conditions) if conditions else "Healthy"

        # Lookup key — use ckd_stage or 0
        key = (has_htn, has_dm, ckd_stage)
        thresholds = self.THRESHOLDS.get(key)

        if thresholds is None:
            # Fallback: try with just CKD if combo not found
            key = (0, 0, ckd_stage)
            thresholds = self.THRESHOLDS.get(key, self.THRESHOLDS[(0, 0, 0)])

        return {
            "condition_profile": condition_label,
            "ckd_stage": ckd_stage,
            "nutrients": thresholds,
        }

    def save_reference(self, path: Path):
        """Save thresholds as a JSON reference file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for key, val in self.THRESHOLDS.items():
            str_key = f"htn={key[0]}_dm={key[1]}_ckd={key[2]}"
            serializable[str_key] = val
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"  ✓ Saved nutrient thresholds reference: {path}")


# ===============================
# Model Evaluator
# ===============================

class ModelEvaluator:
    """Generate comprehensive accuracy analysis reports."""

    CLASS_NAMES = ['Low', 'Moderate', 'High']

    def __init__(self, reports_dir: Path):
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_all(
        self,
        targets: List[str],
        y_true: pd.DataFrame,
        y_pred: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
        """
        Run full evaluation suite for every target.
        Returns a dict of target -> metric_name -> value.
        """
        print("\n" + "="*80)
        print("MODEL ACCURACY ANALYSIS")
        print("="*80)

        all_metrics = {}
        rows_for_csv = []

        for target in targets:
            yt = y_true[target].values
            yp = y_pred[target]

            metrics = self._compute_metrics(yt, yp)
            all_metrics[target] = metrics

            # Confusion matrix heatmap
            self._save_confusion_matrix(yt, yp, target)

            # Collect rows for CSV
            rows_for_csv.append({
                "target": target,
                **metrics,
            })

            # Print per-target summary
            print(f"\n--- {target} ---")
            print(f"  Accuracy:        {metrics['accuracy']:.4f}")
            print(f"  Precision (wt):  {metrics['precision_weighted']:.4f}")
            print(f"  Recall (wt):     {metrics['recall_weighted']:.4f}")
            print(f"  F1-score (wt):   {metrics['f1_weighted']:.4f}")
            print(f"  Cohen's Kappa:   {metrics['cohen_kappa']:.4f}")

            # Also print the sklearn classification report
            print(classification_report(
                yt, yp,
                target_names=self.CLASS_NAMES,
                zero_division=0,
            ))

        # Save CSV
        csv_path = self.reports_dir / "classification_reports.csv"
        pd.DataFrame(rows_for_csv).to_csv(csv_path, index=False)
        print(f"\n  ✓ Saved classification metrics CSV: {csv_path}")

        # Save text summary
        self._save_accuracy_summary(all_metrics)

        return all_metrics

    # ---- private helpers ----

    def _compute_metrics(self, y_true, y_pred) -> Dict[str, float]:
        return {
            "accuracy":           accuracy_score(y_true, y_pred),
            "precision_macro":    precision_score(y_true, y_pred, average='macro', zero_division=0),
            "precision_weighted":  precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall_macro":       recall_score(y_true, y_pred, average='macro', zero_division=0),
            "recall_weighted":    recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1_macro":           f1_score(y_true, y_pred, average='macro', zero_division=0),
            "f1_weighted":        f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "cohen_kappa":        cohen_kappa_score(y_true, y_pred),
        }

    def _save_confusion_matrix(self, y_true, y_pred, target: str):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.CLASS_NAMES,
            yticklabels=self.CLASS_NAMES,
            ax=ax,
        )
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'Confusion Matrix — {target}', fontsize=14)
        fig.tight_layout()
        path = self.reports_dir / f"confusion_matrix_{target}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved confusion matrix: {path}")

    def _save_accuracy_summary(self, all_metrics: Dict[str, Dict[str, float]]):
        path = self.reports_dir / "accuracy_summary.txt"
        lines = [
            "=" * 70,
            "CLINICAL MODEL #1 — ACCURACY ANALYSIS SUMMARY",
            "=" * 70,
            "",
        ]
        for target, metrics in all_metrics.items():
            lines.append(f"Target: {target}")
            lines.append("-" * 40)
            for k, v in metrics.items():
                lines.append(f"  {k:25s} : {v:.4f}")
            lines.append("")

        # Overall average
        lines.append("=" * 70)
        lines.append("OVERALL (macro-averaged across targets)")
        lines.append("=" * 70)
        avg_acc = np.mean([m['accuracy'] for m in all_metrics.values()])
        avg_f1  = np.mean([m['f1_weighted'] for m in all_metrics.values()])
        avg_kappa = np.mean([m['cohen_kappa'] for m in all_metrics.values()])
        lines.append(f"  Mean Accuracy:     {avg_acc:.4f}")
        lines.append(f"  Mean F1 (wt):      {avg_f1:.4f}")
        lines.append(f"  Mean Cohen Kappa:  {avg_kappa:.4f}")
        lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"  ✓ Saved accuracy summary: {path}")


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
        
        # 4. Carb sensitivity (diabetes-focused)
        print("\n4. Carb Sensitivity (Diabetes Risk):")
        df['carb_sensitivity'] = 0
        
        # Moderate: pre-diabetes OR diabetes OR high BMI OR elevated FBS
        mask_moderate = (
            (df['hba1c'] >= 5.7) |  # Pre-diabetes threshold
            (df['fbs'] >= 100) |    # Impaired fasting glucose
            (df['bmi'] >= 25) |     # Overweight
            (df['has_dm'] == 1)
        )
        df.loc[mask_moderate, 'carb_sensitivity'] = 1
        
        # High: uncontrolled diabetes OR very high HbA1c OR very high FBS OR obesity
        mask_high = (
            (df['hba1c'] >= 7.0) |   # Diabetic range
            (df['fbs'] >= 126) |     # Diabetic FBS threshold
            (df['bmi'] >= 30) |      # Obesity
            ((df['has_dm'] == 1) & (df['hba1c'] >= 6.5))
        )
        df.loc[mask_high, 'carb_sensitivity'] = 2
        
        counts = df['carb_sensitivity'].value_counts().sort_index()
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
        
        # Train model for each target and collect predictions
        y_predictions = {}
        for target in self.cfg.targets:
            print(f"\n--- Training: {target} ---")
            
            model = lgb.LGBMClassifier(**self.cfg.lgbm_params)
            model.set_params(monotone_constraints=mono)
            model.fit(X_train, y_train[target])
            
            self.models[target] = model
            y_predictions[target] = model.predict(X_test)
        
        # Store feature names for prediction
        self.feature_names = available_cols
        
        # Run accuracy analysis reports
        evaluator = ModelEvaluator(self.cfg.reports_dir)
        self.eval_metrics = evaluator.evaluate_all(
            list(self.cfg.targets), y_test, y_predictions,
        )
        
        # Save nutrient thresholds reference
        threshold_engine = NutrientThresholdEngine()
        threshold_engine.save_reference(
            self.cfg.reports_dir / "nutrient_thresholds_reference.json"
        )
        
    def predict(self, clinical_input: Dict[str, Any]) -> Dict[str, Any]:
        """Get risk predictions AND permissible nutrient amounts for a patient."""
        X = pd.DataFrame([clinical_input])[self.feature_names]
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        
        label_map = {0: "low", 1: "moderate", 2: "high"}
        
        risk_levels = {
            target: label_map[int(model.predict(X)[0])]
            for target, model in self.models.items()
        }
        
        # Get condition-specific permissible nutrient amounts
        threshold_engine = NutrientThresholdEngine()
        permissible = threshold_engine.get_permissible_amounts(clinical_input)
        
        return {
            "risk_levels": risk_levels,
            "permissible_amounts": permissible,
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
    
    print(f"\nTest patient: HTN + DM + CKD (eGFR=28)")
    print(f"\n  Risk Levels:")
    for target, risk in predictions["risk_levels"].items():
        print(f"    {target}: {risk}")
    
    pa = predictions["permissible_amounts"]
    print(f"\n  Condition Profile: {pa['condition_profile']}")
    print(f"  CKD Stage: {pa['ckd_stage']}")
    print(f"\n  Permissible Daily Nutrient Amounts:")
    print(f"  {'Nutrient':<22} {'Limit':<20} {'Rationale'}")
    print(f"  {'-'*70}")
    for nutrient, info in pa["nutrients"].items():
        lo = info.get('min', '')
        hi = info.get('max', '')
        unit = info['unit']
        if lo and hi:
            limit_str = f"{lo}–{hi} {unit}"
        elif hi:
            limit_str = f"≤{hi} {unit}"
        else:
            limit_str = f"≥{lo} {unit}"
        print(f"    {nutrient:<20} {limit_str:<20} {info['rationale']}")
    
    # 7. Final summary
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE")
    print("="*80)
    print(f"\n  Reports saved to: {config.reports_dir.resolve()}")
    print(f"  Models saved to:  {config.model_dir.resolve()}")
    
    if hasattr(model, 'eval_metrics'):
        print(f"\n  {'Target':<30} {'Accuracy':>10} {'F1 (wt)':>10} {'Kappa':>10}")
        print(f"  {'-'*62}")
        for t, m in model.eval_metrics.items():
            print(f"  {t:<30} {m['accuracy']:>10.4f} {m['f1_weighted']:>10.4f} {m['cohen_kappa']:>10.4f}")
    
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Clinical Risk Model #1")
    parser.add_argument("--sample", type=int, default=100000,
                        help="Number of rows to sample (default: 100000)")
    args = parser.parse_args()
    
    main(sample_rows=args.sample)
