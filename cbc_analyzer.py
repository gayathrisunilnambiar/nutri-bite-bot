"""
CBC Report Analyzer
====================
Parses and validates complete blood count (CBC) and metabolic panel lab values
for clinical nutrition recommendations.

Focuses on key markers for:
- Diabetes Type 1: Blood glucose, HbA1c
- Hypertension: Sodium levels
- Chronic Kidney Disease (CKD): eGFR, Creatinine, Potassium

Author: Nutri-Bite Bot Development Team
Version: 2.0.0
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CKDStage(Enum):
    """CKD staging based on eGFR (KDIGO guidelines)."""
    NORMAL = "Normal (≥90)"
    STAGE_1 = "Stage 1 (≥90 with kidney damage)"
    STAGE_2 = "Stage 2 (60-89)"
    STAGE_3A = "Stage 3a (45-59)"
    STAGE_3B = "Stage 3b (30-44)"
    STAGE_4 = "Stage 4 (15-29)"
    STAGE_5 = "Stage 5 (<15)"


class RiskLevel(Enum):
    """Risk level for clinical alerts."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LabValue:
    """Single lab value with reference range."""
    value: float
    unit: str
    low_normal: float
    high_normal: float
    
    @property
    def is_low(self) -> bool:
        return self.value < self.low_normal
    
    @property
    def is_high(self) -> bool:
        return self.value > self.high_normal
    
    @property
    def is_normal(self) -> bool:
        return self.low_normal <= self.value <= self.high_normal
    
    @property
    def status(self) -> str:
        if self.is_low:
            return "LOW"
        elif self.is_high:
            return "HIGH"
        return "NORMAL"


@dataclass
class CBCReport:
    """Complete blood count and metabolic panel results."""
    # Kidney function
    egfr: Optional[LabValue] = None          # mL/min/1.73m²
    creatinine: Optional[LabValue] = None    # mg/dL
    
    # Electrolytes
    potassium: Optional[LabValue] = None     # mEq/L
    sodium: Optional[LabValue] = None        # mEq/L
    
    # Diabetes markers
    glucose: Optional[LabValue] = None       # mg/dL (fasting)
    hba1c: Optional[LabValue] = None         # %
    
    # Patient conditions (user-provided)
    has_diabetes_t1: bool = False
    has_hypertension: bool = False
    has_ckd: bool = False


class CBCAnalyzer:
    """
    Analyzes CBC/metabolic panel reports and generates clinical alerts.
    """
    
    # Reference ranges (adult values)
    REFERENCE_RANGES = {
        'egfr': {'low': 90, 'high': 999, 'unit': 'mL/min/1.73m²'},
        'creatinine': {'low': 0.7, 'high': 1.3, 'unit': 'mg/dL'},
        'potassium': {'low': 3.5, 'high': 5.0, 'unit': 'mEq/L'},
        'sodium': {'low': 136, 'high': 145, 'unit': 'mEq/L'},
        'glucose': {'low': 70, 'high': 100, 'unit': 'mg/dL'},
        'hba1c': {'low': 4.0, 'high': 5.6, 'unit': '%'},
    }
    
    def __init__(self):
        """Initialize the CBC analyzer."""
        self.report: Optional[CBCReport] = None
        self.alerts: List[Dict] = []
    
    def parse_lab_values(
        self,
        egfr: Optional[float] = None,
        creatinine: Optional[float] = None,
        potassium: Optional[float] = None,
        sodium: Optional[float] = None,
        glucose: Optional[float] = None,
        hba1c: Optional[float] = None,
        has_diabetes_t1: bool = False,
        has_hypertension: bool = False,
        has_ckd: bool = False
    ) -> CBCReport:
        """
        Parse raw lab values into a structured CBCReport.
        
        Args:
            egfr: Estimated GFR (mL/min/1.73m²)
            creatinine: Serum creatinine (mg/dL)
            potassium: Serum potassium (mEq/L)
            sodium: Serum sodium (mEq/L)
            glucose: Fasting blood glucose (mg/dL)
            hba1c: Hemoglobin A1c (%)
            has_diabetes_t1: User has Type 1 Diabetes
            has_hypertension: User has Hypertension
            has_ckd: User has Chronic Kidney Disease
            
        Returns:
            CBCReport with validated lab values
        """
        report = CBCReport(
            has_diabetes_t1=has_diabetes_t1,
            has_hypertension=has_hypertension,
            has_ckd=has_ckd
        )
        
        # Parse each lab value
        if egfr is not None:
            ref = self.REFERENCE_RANGES['egfr']
            report.egfr = LabValue(egfr, ref['unit'], ref['low'], ref['high'])
        
        if creatinine is not None:
            ref = self.REFERENCE_RANGES['creatinine']
            report.creatinine = LabValue(creatinine, ref['unit'], ref['low'], ref['high'])
        
        if potassium is not None:
            ref = self.REFERENCE_RANGES['potassium']
            report.potassium = LabValue(potassium, ref['unit'], ref['low'], ref['high'])
        
        if sodium is not None:
            ref = self.REFERENCE_RANGES['sodium']
            report.sodium = LabValue(sodium, ref['unit'], ref['low'], ref['high'])
        
        if glucose is not None:
            ref = self.REFERENCE_RANGES['glucose']
            report.glucose = LabValue(glucose, ref['unit'], ref['low'], ref['high'])
        
        if hba1c is not None:
            ref = self.REFERENCE_RANGES['hba1c']
            report.hba1c = LabValue(hba1c, ref['unit'], ref['low'], ref['high'])
        
        self.report = report
        self._generate_alerts()
        
        logger.info(f"Parsed CBC report with {len(self.alerts)} alerts")
        return report
    
    def classify_ckd_stage(self) -> Tuple[CKDStage, str]:
        """
        Classify CKD stage based on eGFR.
        
        Returns:
            Tuple of (CKDStage, description)
        """
        if self.report is None or self.report.egfr is None:
            return CKDStage.NORMAL, "eGFR not available"
        
        egfr = self.report.egfr.value
        
        if egfr >= 90:
            if self.report.has_ckd:
                return CKDStage.STAGE_1, f"eGFR {egfr:.0f} with kidney damage markers"
            return CKDStage.NORMAL, f"eGFR {egfr:.0f} - Normal kidney function"
        elif egfr >= 60:
            return CKDStage.STAGE_2, f"eGFR {egfr:.0f} - Mild reduction"
        elif egfr >= 45:
            return CKDStage.STAGE_3A, f"eGFR {egfr:.0f} - Moderate reduction"
        elif egfr >= 30:
            return CKDStage.STAGE_3B, f"eGFR {egfr:.0f} - Moderate to severe reduction"
        elif egfr >= 15:
            return CKDStage.STAGE_4, f"eGFR {egfr:.0f} - Severe reduction"
        else:
            return CKDStage.STAGE_5, f"eGFR {egfr:.0f} - Kidney failure"
    
    def _generate_alerts(self):
        """Generate clinical alerts based on lab values."""
        self.alerts = []
        
        if self.report is None:
            return
        
        # Potassium alerts (critical for CKD)
        if self.report.potassium:
            k = self.report.potassium.value
            if k > 6.0:
                self.alerts.append({
                    'level': RiskLevel.CRITICAL,
                    'type': 'potassium',
                    'message': f'CRITICAL: Potassium {k} mEq/L - Risk of cardiac arrhythmia',
                    'action': 'Immediate potassium restriction required'
                })
            elif k > 5.5:
                self.alerts.append({
                    'level': RiskLevel.HIGH,
                    'type': 'potassium',
                    'message': f'HIGH: Potassium {k} mEq/L - Hyperkalemia',
                    'action': 'Strict dietary potassium restriction'
                })
            elif k > 5.0:
                self.alerts.append({
                    'level': RiskLevel.MODERATE,
                    'type': 'potassium',
                    'message': f'Potassium {k} mEq/L - Upper limit',
                    'action': 'Monitor potassium intake'
                })
        
        # eGFR alerts (kidney function)
        if self.report.egfr:
            egfr = self.report.egfr.value
            if egfr < 15:
                self.alerts.append({
                    'level': RiskLevel.CRITICAL,
                    'type': 'kidney',
                    'message': f'CRITICAL: eGFR {egfr} - Kidney failure (Stage 5)',
                    'action': 'Strict dietary management essential'
                })
            elif egfr < 30:
                self.alerts.append({
                    'level': RiskLevel.HIGH,
                    'type': 'kidney',
                    'message': f'eGFR {egfr} - Severe CKD (Stage 4)',
                    'action': 'Potassium, phosphorus, protein restriction'
                })
            elif egfr < 60:
                self.alerts.append({
                    'level': RiskLevel.MODERATE,
                    'type': 'kidney',
                    'message': f'eGFR {egfr} - Moderate CKD (Stage 3)',
                    'action': 'Consider potassium restriction'
                })
        
        # Glucose/HbA1c alerts
        if self.report.glucose:
            glu = self.report.glucose.value
            if glu > 200:
                self.alerts.append({
                    'level': RiskLevel.HIGH,
                    'type': 'glucose',
                    'message': f'Glucose {glu} mg/dL - Poor glycemic control',
                    'action': 'Carbohydrate awareness critical'
                })
            elif glu > 126:
                self.alerts.append({
                    'level': RiskLevel.MODERATE,
                    'type': 'glucose',
                    'message': f'Glucose {glu} mg/dL - Elevated',
                    'action': 'Monitor carbohydrate intake'
                })
        
        if self.report.hba1c:
            a1c = self.report.hba1c.value
            if a1c > 9.0:
                self.alerts.append({
                    'level': RiskLevel.HIGH,
                    'type': 'hba1c',
                    'message': f'HbA1c {a1c}% - Poor long-term control',
                    'action': 'Strict carbohydrate management'
                })
            elif a1c > 7.0:
                self.alerts.append({
                    'level': RiskLevel.MODERATE,
                    'type': 'hba1c',
                    'message': f'HbA1c {a1c}% - Above target',
                    'action': 'Improve glycemic control'
                })
        
        # Sodium alerts (for hypertension)
        if self.report.sodium and self.report.has_hypertension:
            na = self.report.sodium.value
            if na > 145:
                self.alerts.append({
                    'level': RiskLevel.MODERATE,
                    'type': 'sodium',
                    'message': f'Sodium {na} mEq/L with HTN',
                    'action': 'Strict sodium restriction (<1500mg/day)'
                })
    
    def get_summary(self) -> Dict:
        """
        Get a summary of the CBC analysis.
        
        Returns:
            Dict with lab values, CKD stage, and alerts
        """
        if self.report is None:
            return {'error': 'No report loaded'}
        
        ckd_stage, ckd_desc = self.classify_ckd_stage()
        
        summary = {
            'conditions': {
                'diabetes_t1': self.report.has_diabetes_t1,
                'hypertension': self.report.has_hypertension,
                'ckd': self.report.has_ckd
            },
            'lab_values': {},
            'ckd_stage': ckd_stage.value,
            'ckd_description': ckd_desc,
            'alerts': [
                {
                    'level': a['level'].value,
                    'type': a['type'],
                    'message': a['message'],
                    'action': a['action']
                }
                for a in self.alerts
            ],
            'critical_count': sum(1 for a in self.alerts if a['level'] == RiskLevel.CRITICAL),
            'high_count': sum(1 for a in self.alerts if a['level'] == RiskLevel.HIGH)
        }
        
        # Add lab values
        if self.report.egfr:
            summary['lab_values']['egfr'] = {
                'value': self.report.egfr.value,
                'unit': self.report.egfr.unit,
                'status': self.report.egfr.status
            }
        
        if self.report.creatinine:
            summary['lab_values']['creatinine'] = {
                'value': self.report.creatinine.value,
                'unit': self.report.creatinine.unit,
                'status': self.report.creatinine.status
            }
        
        if self.report.potassium:
            summary['lab_values']['potassium'] = {
                'value': self.report.potassium.value,
                'unit': self.report.potassium.unit,
                'status': self.report.potassium.status
            }
        
        if self.report.sodium:
            summary['lab_values']['sodium'] = {
                'value': self.report.sodium.value,
                'unit': self.report.sodium.unit,
                'status': self.report.sodium.status
            }
        
        if self.report.glucose:
            summary['lab_values']['glucose'] = {
                'value': self.report.glucose.value,
                'unit': self.report.glucose.unit,
                'status': self.report.glucose.status
            }
        
        if self.report.hba1c:
            summary['lab_values']['hba1c'] = {
                'value': self.report.hba1c.value,
                'unit': self.report.hba1c.unit,
                'status': self.report.hba1c.status
            }
        
        return summary


# Demo usage
if __name__ == "__main__":
    analyzer = CBCAnalyzer()
    
    # Sample CKD patient with hypertension
    report = analyzer.parse_lab_values(
        egfr=45,
        creatinine=2.1,
        potassium=5.2,
        sodium=142,
        glucose=180,
        hba1c=7.5,
        has_diabetes_t1=True,
        has_hypertension=True,
        has_ckd=True
    )
    
    summary = analyzer.get_summary()
    
    print("=" * 60)
    print("CBC ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"CKD Stage: {summary['ckd_stage']}")
    print(f"Description: {summary['ckd_description']}")
    print(f"\nConditions: {summary['conditions']}")
    print(f"\nLab Values:")
    for name, data in summary['lab_values'].items():
        print(f"  {name}: {data['value']} {data['unit']} ({data['status']})")
    
    print(f"\nAlerts ({len(summary['alerts'])}):")
    for alert in summary['alerts']:
        print(f"  [{alert['level'].upper()}] {alert['message']}")
        print(f"    → {alert['action']}")
