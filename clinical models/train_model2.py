"""
Clinical Model #2: Portion Control System
==========================================
Recommends safe ingredient portions based on:
- Model1 clinical outputs (sodium/potassium/protein/carb sensitivity levels)
- Disease flags (has_ckd, has_htn, has_dm)
- IFCT nutritional data (Indian Food Composition Tables 2017)

Usage:
    python train_model2.py
    
Dependencies:
    - Model1 trained models in ../artifacts/models/
    - IFCT database in ./ifct_database.csv
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from difflib import get_close_matches
import warnings

warnings.filterwarnings('ignore')


# ===============================
# Configuration
# ===============================
@dataclass
class Model2Config:
    """Configuration for Model2 Portion Control System"""
    
    # Model1 artifacts path
    model1_dir: Path = Path("../artifacts/models")
    
    # IFCT database path
    ifct_path: Path = Path("ifct_database.csv")
    
    # Daily nutrient budgets by condition (mg or g per day)
    # Based on clinical guidelines
    daily_budgets: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "healthy": {
            "sodium_mg": 2300,      # AHA recommendation
            "potassium_mg": 4700,   # DASH diet
            "protein_g": 56,        # 0.8g/kg for 70kg adult
            "carbs_g": 275,         # ~55% of 2000 kcal
            "phosphorus_mg": 1250,  # RDA for adults
        },
        "ckd": {
            "sodium_mg": 2000,      # CKD Stage 3-4
            "potassium_mg": 2000,   # CKD restriction
            "protein_g": 42,        # 0.6g/kg for CKD
            "carbs_g": 275,
            "phosphorus_mg": 800,   # CKD limit (kidney.org.uk guideline)
        },
        "htn": {
            "sodium_mg": 1500,      # HTN strict limit
            "potassium_mg": 4700,   # DASH encourages K+
            "protein_g": 56,
            "carbs_g": 275,
            "phosphorus_mg": 1250,
        },
        "dm": {
            "sodium_mg": 2300,
            "potassium_mg": 4700,
            "protein_g": 56,
            "carbs_g": 180,         # Diabetes carb limit
            "phosphorus_mg": 1250,
        },
        "ckd_htn": {
            "sodium_mg": 1500,      # HTN strict limit (priority)
            "potassium_mg": 2000,   # CKD restriction (priority)
            "protein_g": 42,
            "carbs_g": 275,
            "phosphorus_mg": 800,   # CKD phosphorus restriction
        },
        "ckd_dm": {
            "sodium_mg": 2000,
            "potassium_mg": 2000,
            "protein_g": 42,        # 0.6g/kg for CKD+DM
            "carbs_g": 150,         # More strict with DM
            "phosphorus_mg": 800,   # CKD phosphorus restriction
        },
        "ckd_htn_dm": {
            "sodium_mg": 1500,      # Most restrictive
            "potassium_mg": 2000,
            "protein_g": 42,
            "carbs_g": 150,
            "phosphorus_mg": 800,   # CKD phosphorus restriction
        },
    })
    
    # Risk level fractions - what % of daily budget can ONE ingredient use
    risk_fractions: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.40,      # 40% of remaining budget
        "moderate": 0.25, # 25% of remaining budget
        "high": 0.10,     # 10% of remaining budget (very restricted)
    })
    
    # Portion thresholds
    default_cap_g: float = 300.0     # Max grams per ingredient
    half_portion_g: float = 75.0     # Below this = "Half portion"
    avoid_threshold_g: float = 5.0   # Below this = "Avoid"


# ===============================
# IFCT Database Interface
# ===============================
class IFCTDatabase:
    """
    Read-only interface to IFCT 2017 nutritional data.
    All values are per 100g edible portion.
    """
    
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        
        # Required columns for portion control
        required = {
            "ingredient",
            "sodium_mg_per_100g",
            "potassium_mg_per_100g",
            "protein_g_per_100g",
            "carbs_g_per_100g",
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"IFCT CSV missing columns: {sorted(missing)}")
        
        # Create normalized index for searching
        self.df["ingredient_norm"] = self.df["ingredient"].astype(str).str.lower().str.strip()
        self.idx = self.df.set_index("ingredient_norm")
        
        # Create list of all ingredients for fuzzy matching
        self.all_ingredients = self.df["ingredient"].tolist()
        self.all_ingredients_lower = [i.lower() for i in self.all_ingredients]
        
        print(f"✓ Loaded IFCT database with {len(self.df)} ingredients")
    
    def get_nutrients_per_100g(self, ingredient: str) -> Dict[str, float]:
        """Get nutrient values per 100g for an ingredient"""
        key = ingredient.lower().strip()
        
        if key not in self.idx.index:
            raise KeyError(f"Ingredient not found: {ingredient}")
        
        row = self.idx.loc[key]
        
        return {
            "sodium_mg": float(row.get("sodium_mg_per_100g", 0)),
            "potassium_mg": float(row.get("potassium_mg_per_100g", 0)),
            "protein_g": float(row.get("protein_g_per_100g", 0)),
            "carbs_g": float(row.get("carbs_g_per_100g", 0)),
            "phosphorus_mg": float(row.get("phosphorus_mg_per_100g", 0)),
            "fat_g": float(row.get("fat_g_per_100g", 0)),
            "fiber_g": float(row.get("fiber_g_per_100g", 0)),
            "calories": float(row.get("calories_per_100g", 0)),
        }
    
    def search_ingredient(self, query: str, n_matches: int = 5) -> List[str]:
        """Fuzzy search for ingredients matching query"""
        query_lower = query.lower().strip()
        
        # Exact match
        if query_lower in self.all_ingredients_lower:
            idx = self.all_ingredients_lower.index(query_lower)
            return [self.all_ingredients[idx]]
        
        # Partial matches
        matches = []
        for i, ing_lower in enumerate(self.all_ingredients_lower):
            if query_lower in ing_lower or ing_lower in query_lower:
                matches.append(self.all_ingredients[i])
        
        if matches:
            return matches[:n_matches]
        
        # Fuzzy match
        close = get_close_matches(query_lower, self.all_ingredients_lower, n=n_matches, cutoff=0.5)
        return [self.all_ingredients[self.all_ingredients_lower.index(m)] for m in close]
    
    def list_by_category(self, category: str) -> List[str]:
        """List all ingredients in a category"""
        mask = self.df["category"].str.lower() == category.lower()
        return self.df.loc[mask, "ingredient"].tolist()
    
    def get_all_categories(self) -> List[str]:
        """Get list of all categories"""
        return self.df["category"].unique().tolist()


# ===============================
# Data Classes
# ===============================
@dataclass
class DailyBudget:
    """Remaining nutrient budget for the day"""
    sodium_mg_remaining: float
    potassium_mg_remaining: float
    protein_g_remaining: float
    carbs_g_remaining: float
    phosphorus_mg_remaining: float = 700.0


@dataclass
class PortionDecision:
    """Portion recommendation for a single ingredient"""
    ingredient: str
    max_grams: float
    label: str                      # "Allowed" | "Half portion" | "Avoid"
    explanation: str
    nutrient_load_at_max: Dict[str, float]
    binding_constraint: str         # Which nutrient constrained the portion


# ===============================
# Model1 Integration
# ===============================
class Model1Integration:
    """
    Load and use Model1's trained clinical risk models.
    Predicts: sodium_sensitivity, potassium_sensitivity, 
              protein_restriction, carb_sensitivity
    """
    
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.imputer = None
        self.scaler = None
        self.feature_names = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all Model1 components"""
        try:
            self.models = {
                "sodium_sensitivity": joblib.load(self.model_dir / "sodium_sensitivity.joblib"),
                "potassium_sensitivity": joblib.load(self.model_dir / "potassium_sensitivity.joblib"),
                "protein_restriction": joblib.load(self.model_dir / "protein_restriction.joblib"),
                "carb_sensitivity": joblib.load(self.model_dir / "carb_sensitivity.joblib"),
            }
            self.imputer = joblib.load(self.model_dir / "imputer.joblib")
            self.scaler = joblib.load(self.model_dir / "scaler.joblib")
            self.feature_names = joblib.load(self.model_dir / "feature_names.joblib")
            
            print(f"✓ Loaded Model1 components from {self.model_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Model1: {e}")
    
    def predict_risk_levels(self, patient_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Predict clinical risk levels from patient data.
        
        Args:
            patient_data: Dict with keys like age, sex_male, has_htn, has_dm, 
                         has_ckd, serum_sodium, serum_potassium, etc.
        
        Returns:
            Dict with risk levels: {
                "sodium_sensitivity": "low/moderate/high",
                "potassium_sensitivity": "low/moderate/high",
                "protein_restriction": "low/moderate/high",
                "carb_sensitivity": "low/moderate/high",
                "phosphorus_sensitivity": "low/moderate/high"
            }
        """
        # Prepare features
        X = pd.DataFrame([patient_data])[self.feature_names]
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        
        label_map = {0: "low", 1: "moderate", 2: "high"}
        
        results = {
            target: label_map[int(model.predict(X)[0])]
            for target, model in self.models.items()
        }
        
        # Add phosphorus sensitivity based on CKD status (kidney.org.uk guideline)
        # Phosphate additives are harmful for CKD patients - restrict phosphorus
        has_ckd = patient_data.get("has_ckd", 0) == 1
        egfr = patient_data.get("egfr", 90)
        
        if has_ckd or egfr < 30:
            results["phosphorus_sensitivity"] = "high"
        elif egfr < 60:
            results["phosphorus_sensitivity"] = "moderate"
        else:
            results["phosphorus_sensitivity"] = "low"
        
        return results


# ===============================
# Portion Recommender
# ===============================
class PortionRecommender:
    """
    Deterministic portion recommendation engine.
    Uses IFCT nutritional data + Model1 risk levels to recommend safe portions.
    """
    
    def __init__(
        self,
        ifct: IFCTDatabase,
        config: Model2Config
    ):
        self.ifct = ifct
        self.config = config
    
    @staticmethod
    def _grams_from_budget(
        nutrient_per_100g: float, 
        remaining_budget: float, 
        risk_fraction: float
    ) -> float:
        """
        Compute maximum grams such that:
        nutrient_load <= remaining_budget * risk_fraction
        
        grams = (allowed_amount / nutrient_per_100g) * 100
        """
        if nutrient_per_100g <= 0:
            return float("inf")  # Not constrained by this nutrient
        
        allowed = max(0.0, remaining_budget) * risk_fraction
        grams = (allowed / nutrient_per_100g) * 100.0
        return max(0.0, grams)
    
    def recommend_one(
        self,
        ingredient: str,
        risk_levels: Dict[str, str],
        budget: DailyBudget
    ) -> PortionDecision:
        """
        Recommend portion for a single ingredient.
        
        Args:
            ingredient: Name of ingredient (must be in IFCT)
            risk_levels: Output from Model1 (sodium_sensitivity, etc.)
            budget: Remaining daily nutrient budget
        
        Returns:
            PortionDecision with max_grams, label, and explanation
        """
        # Get nutrient values per 100g
        try:
            n = self.ifct.get_nutrients_per_100g(ingredient)
        except KeyError:
            # Ingredient not in database
            suggestions = self.ifct.search_ingredient(ingredient)
            suggestion_text = f" Did you mean: {', '.join(suggestions)}" if suggestions else ""
            return PortionDecision(
                ingredient=ingredient,
                max_grams=0.0,
                label="Avoid",
                explanation=f"Avoid: Ingredient '{ingredient}' not found in IFCT database.{suggestion_text}",
                nutrient_load_at_max={k: 0.0 for k in ["sodium_mg", "potassium_mg", "protein_g", "carbs_g"]},
                binding_constraint="unknown"
            )
        
        # Get risk levels with defaults
        def norm_risk(r: str) -> str:
            r = (r or "moderate").lower().strip()
            return r if r in self.config.risk_fractions else "moderate"
        
        sod_risk = norm_risk(risk_levels.get("sodium_sensitivity", "moderate"))
        pot_risk = norm_risk(risk_levels.get("potassium_sensitivity", "moderate"))
        pro_risk = norm_risk(risk_levels.get("protein_restriction", "moderate"))
        carb_risk = norm_risk(risk_levels.get("carb_sensitivity", "moderate"))
        # Phosphorus risk is high for CKD patients (kidney.org.uk guideline)
        phos_risk = norm_risk(risk_levels.get("phosphorus_sensitivity", "low"))
        
        # Get risk fractions
        rf = self.config.risk_fractions
        
        # Compute grams allowed by each constraint
        g_sod = self._grams_from_budget(n["sodium_mg"], budget.sodium_mg_remaining, rf[sod_risk])
        g_pot = self._grams_from_budget(n["potassium_mg"], budget.potassium_mg_remaining, rf[pot_risk])
        g_pro = self._grams_from_budget(n["protein_g"], budget.protein_g_remaining, rf[pro_risk])
        g_carb = self._grams_from_budget(n["carbs_g"], budget.carbs_g_remaining, rf[carb_risk])
        g_phos = self._grams_from_budget(n["phosphorus_mg"], budget.phosphorus_mg_remaining, rf[phos_risk])
        
        constraints = {
            "sodium": g_sod,
            "potassium": g_pot,
            "protein": g_pro,
            "carbs": g_carb,
            "phosphorus": g_phos,
        }
        
        # Find most restrictive constraint
        binding = min(constraints, key=constraints.get)
        max_g = min(constraints.values())
        max_g = min(max_g, self.config.default_cap_g)  # Apply practical cap
        
        # Determine label
        if max_g <= self.config.avoid_threshold_g:
            label = "Avoid"
        elif max_g <= self.config.half_portion_g:
            label = "Half portion"
        else:
            label = "Allowed"
        
        # Calculate nutrient load at max portion
        factor = max_g / 100.0
        load = {
            "sodium_mg": round(n["sodium_mg"] * factor, 1),
            "potassium_mg": round(n["potassium_mg"] * factor, 1),
            "protein_g": round(n["protein_g"] * factor, 1),
            "carbs_g": round(n["carbs_g"] * factor, 1),
        }
        
        # Build explanation
        risk_explanations = {
            "sodium": f"sodium sensitivity ({sod_risk})",
            "potassium": f"potassium sensitivity ({pot_risk})",
            "protein": f"protein restriction ({pro_risk})",
            "carbs": f"carb sensitivity ({carb_risk})",
            "phosphorus": f"phosphorus restriction ({phos_risk}) - CKD guideline",
        }
        
        explanation = (
            f"{label}: max ~{max_g:.0f}g. "
            f"Limited by {risk_explanations[binding]}. "
            f"At this portion: {load['sodium_mg']:.0f}mg Na, "
            f"{load['potassium_mg']:.0f}mg K, "
            f"{load['protein_g']:.1f}g protein, "
            f"{load['carbs_g']:.1f}g carbs."
        )
        
        return PortionDecision(
            ingredient=ingredient,
            max_grams=round(max_g, 1),
            label=label,
            explanation=explanation,
            nutrient_load_at_max=load,
            binding_constraint=binding
        )
    
    def recommend_batch(
        self,
        ingredients: List[str],
        risk_levels: Dict[str, str],
        budget: DailyBudget
    ) -> List[PortionDecision]:
        """Recommend portions for multiple ingredients"""
        return [
            self.recommend_one(ing, risk_levels, budget)
            for ing in ingredients
        ]


# ===============================
# Budget Calculator
# ===============================
class BudgetCalculator:
    """
    Calculate daily nutrient budgets based on disease conditions.
    """
    
    def __init__(self, config: Model2Config):
        self.config = config
    
    def get_daily_budget(
        self,
        has_ckd: bool = False,
        has_htn: bool = False,
        has_dm: bool = False,
        consumed_today: Optional[Dict[str, float]] = None
    ) -> DailyBudget:
        """
        Get remaining daily budget based on conditions.
        
        Args:
            has_ckd: Has chronic kidney disease
            has_htn: Has hypertension
            has_dm: Has diabetes
            consumed_today: Nutrients already consumed (optional)
        
        Returns:
            DailyBudget with remaining nutrient allowances
        """
        # Determine condition key
        conditions = []
        if has_ckd:
            conditions.append("ckd")
        if has_htn:
            conditions.append("htn")
        if has_dm:
            conditions.append("dm")
        
        if not conditions:
            key = "healthy"
        else:
            key = "_".join(sorted(conditions))
        
        # Get base budget (fall back to most restrictive if combo not defined)
        budgets = self.config.daily_budgets
        if key in budgets:
            base = budgets[key]
        elif "ckd_htn_dm" in budgets and has_ckd:
            base = budgets["ckd_htn_dm"]  # Most restrictive
        elif "ckd" in budgets and has_ckd:
            base = budgets["ckd"]
        else:
            base = budgets["healthy"]
        
        # Subtract already consumed
        consumed = consumed_today or {}
        
        return DailyBudget(
            sodium_mg_remaining=max(0, base["sodium_mg"] - consumed.get("sodium_mg", 0)),
            potassium_mg_remaining=max(0, base["potassium_mg"] - consumed.get("potassium_mg", 0)),
            protein_g_remaining=max(0, base["protein_g"] - consumed.get("protein_g", 0)),
            carbs_g_remaining=max(0, base["carbs_g"] - consumed.get("carbs_g", 0)),
            phosphorus_mg_remaining=max(0, base.get("phosphorus_mg", 700) - consumed.get("phosphorus_mg", 0)),
        )


# ===============================
# Ingredient Input Handler
# ===============================
class IngredientInputHandler:
    """Handle user input for available ingredients"""
    
    def __init__(self, ifct: IFCTDatabase):
        self.ifct = ifct
    
    def parse_input(self, input_str: str) -> Tuple[List[str], List[str]]:
        """
        Parse comma-separated ingredient list.
        
        Args:
            input_str: "tomato, spinach, banana, rice"
        
        Returns:
            (found_ingredients, not_found_with_suggestions)
        """
        raw_ingredients = [i.strip() for i in input_str.split(",") if i.strip()]
        
        found = []
        not_found = []
        
        for ing in raw_ingredients:
            # Check if exists (case-insensitive)
            if ing.lower() in self.ifct.all_ingredients_lower:
                idx = self.ifct.all_ingredients_lower.index(ing.lower())
                found.append(self.ifct.all_ingredients[idx])
            else:
                # Try to find suggestions
                suggestions = self.ifct.search_ingredient(ing, n_matches=3)
                if suggestions:
                    not_found.append(f"'{ing}' → Did you mean: {', '.join(suggestions)}?")
                else:
                    not_found.append(f"'{ing}' → No matches found")
        
        return found, not_found


# ===============================
# Main Model2 Class
# ===============================
class PortionControlModel:
    """
    Complete Model2 Portion Control System.
    Integrates Model1 risk predictions with IFCT-based portion recommendations.
    """
    
    def __init__(self, config: Model2Config = None):
        self.config = config or Model2Config()
        
        # Initialize components
        print("\n" + "="*60)
        print("INITIALIZING MODEL2: PORTION CONTROL SYSTEM")
        print("="*60)
        
        # Load IFCT database
        ifct_path = Path(__file__).parent / self.config.ifct_path
        self.ifct = IFCTDatabase(str(ifct_path))
        
        # Load Model1
        model1_path = Path(__file__).parent / self.config.model1_dir
        self.model1 = Model1Integration(model1_path)
        
        # Initialize helpers
        self.recommender = PortionRecommender(self.ifct, self.config)
        self.budget_calc = BudgetCalculator(self.config)
        self.input_handler = IngredientInputHandler(self.ifct)
        
        print("✓ Model2 initialization complete")
        print("="*60 + "\n")
    
    def get_recommendations(
        self,
        patient_data: Dict[str, Any],
        ingredients: List[str],
        consumed_today: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Get portion recommendations for ingredients given patient data.
        
        Args:
            patient_data: Clinical data dict with keys:
                - age, sex_male, has_htn, has_dm, has_ckd
                - serum_sodium, serum_potassium, creatinine, egfr
                - hba1c, fbs, sbp, dbp, bmi
            ingredients: List of ingredient names
            consumed_today: Already consumed nutrients (optional)
        
        Returns:
            Dict with risk_levels, budget, recommendations, and summary
        """
        # Step 1: Get risk levels from Model1
        risk_levels = self.model1.predict_risk_levels(patient_data)
        
        # Step 2: Calculate remaining budget
        budget = self.budget_calc.get_daily_budget(
            has_ckd=patient_data.get("has_ckd", 0) == 1,
            has_htn=patient_data.get("has_htn", 0) == 1,
            has_dm=patient_data.get("has_dm", 0) == 1,
            consumed_today=consumed_today
        )
        
        # Step 3: Get portion recommendations
        recommendations = self.recommender.recommend_batch(ingredients, risk_levels, budget)
        
        # Step 4: Build summary
        allowed = [r.ingredient for r in recommendations if r.label == "Allowed"]
        half_portion = [r.ingredient for r in recommendations if r.label == "Half portion"]
        avoid = [r.ingredient for r in recommendations if r.label == "Avoid"]
        
        return {
            "patient_conditions": {
                "has_ckd": patient_data.get("has_ckd", 0) == 1,
                "has_htn": patient_data.get("has_htn", 0) == 1,
                "has_dm": patient_data.get("has_dm", 0) == 1,
            },
            "risk_levels": risk_levels,
            "daily_budget": {
                "sodium_mg": budget.sodium_mg_remaining,
                "potassium_mg": budget.potassium_mg_remaining,
                "protein_g": budget.protein_g_remaining,
                "carbs_g": budget.carbs_g_remaining,
                "phosphorus_mg": budget.phosphorus_mg_remaining,
            },
            "recommendations": [
                {
                    "ingredient": r.ingredient,
                    "max_grams": r.max_grams,
                    "label": r.label,
                    "explanation": r.explanation,
                    "binding_constraint": r.binding_constraint,
                    "nutrient_load": r.nutrient_load_at_max,
                }
                for r in recommendations
            ],
            "summary": {
                "allowed": allowed,
                "half_portion": half_portion,
                "avoid": avoid,
            }
        }
    
    def interactive_mode(self):
        """Run interactive ingredient input mode"""
        print("\n" + "="*60)
        print("MODEL2 INTERACTIVE MODE")
        print("="*60)
        
        # Get patient data
        print("\nEnter patient clinical data:")
        patient = {
            "age": int(input("  Age: ") or "55"),
            "sex_male": int(input("  Sex (0=female, 1=male): ") or "1"),
            "has_htn": int(input("  Has HTN? (0/1): ") or "1"),
            "has_dm": int(input("  Has DM? (0/1): ") or "1"),
            "has_ckd": int(input("  Has CKD? (0/1): ") or "1"),
            "serum_sodium": float(input("  Serum sodium (mEq/L): ") or "140"),
            "serum_potassium": float(input("  Serum potassium (mEq/L): ") or "4.5"),
            "creatinine": float(input("  Creatinine (mg/dL): ") or "1.5"),
            "egfr": float(input("  eGFR (mL/min): ") or "45"),
            "hba1c": float(input("  HbA1c (%): ") or "7.5"),
            "fbs": float(input("  FBS (mg/dL): ") or "130"),
            "sbp": int(input("  SBP (mmHg): ") or "145"),
            "dbp": int(input("  DBP (mmHg): ") or "90"),
            "bmi": float(input("  BMI: ") or "28"),
        }
        
        # Get ingredients
        print("\nEnter available ingredients (comma-separated):")
        ing_input = input("  → ")
        
        found, not_found = self.input_handler.parse_input(ing_input)
        
        if not_found:
            print("\n⚠ Some ingredients not found:")
            for nf in not_found:
                print(f"    {nf}")
        
        if not found:
            print("\nNo valid ingredients to analyze.")
            return
        
        print(f"\n✓ Found {len(found)} ingredients: {', '.join(found)}")
        
        # Get recommendations
        results = self.get_recommendations(patient, found)
        
        # Display results
        print("\n" + "="*60)
        print("PORTION RECOMMENDATIONS")
        print("="*60)
        
        print(f"\nPatient Risk Levels (from Model1):")
        for key, val in results["risk_levels"].items():
            print(f"  • {key}: {val}")
        
        print(f"\nDaily Budget Remaining:")
        for key, val in results["daily_budget"].items():
            print(f"  • {key}: {val}")
        
        print("\nRecommendations:")
        print("-"*60)
        for rec in results["recommendations"]:
            print(f"\n  {rec['ingredient']}")
            print(f"    Label: {rec['label']}")
            print(f"    Max portion: {rec['max_grams']}g")
            print(f"    Reason: {rec['explanation']}")
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"  ✓ Allowed: {', '.join(results['summary']['allowed']) or 'None'}")
        print(f"  ⚠ Half portion: {', '.join(results['summary']['half_portion']) or 'None'}")
        print(f"  ✗ Avoid: {', '.join(results['summary']['avoid']) or 'None'}")


# ===============================
# Main Entry Point
# ===============================
def main():
    """Test Model2 with sample data"""
    
    # Initialize model
    model = PortionControlModel()
    
    # Test patient with CKD + HTN + DM
    patient = {
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
        "bmi": 32
    }
    
    # Test ingredients
    ingredients = [
        "Spinach (Palak)",
        "Banana, ripe",
        "Potato (Aloo)",
        "Rice, milled (white)",
        "Tomato, ripe",
        "Curd (Dahi/Yogurt)",
        "Chicken, breast",
        "Apple",
    ]
    
    # Get recommendations
    print("\n" + "="*60)
    print("TEST: CKD + HTN + DM PATIENT")
    print("="*60)
    
    results = model.get_recommendations(patient, ingredients)
    
    # Display results
    print("\n📋 Risk Levels (from Model1):")
    for key, val in results["risk_levels"].items():
        print(f"   {key}: {val}")
    
    print("\n📊 Daily Budget:")
    for key, val in results["daily_budget"].items():
        print(f"   {key}: {val}")
    
    print("\n🍽️ Portion Recommendations:")
    print("-"*60)
    
    # Create results table
    df = pd.DataFrame([
        {
            "Ingredient": r["ingredient"],
            "Label": r["label"],
            "Max (g)": r["max_grams"],
            "Limited by": r["binding_constraint"],
        }
        for r in results["recommendations"]
    ])
    print(df.to_string(index=False))
    
    print(f"\n✓ ALLOWED: {', '.join(results['summary']['allowed']) or 'None'}")
    print(f"⚠ HALF PORTION: {', '.join(results['summary']['half_portion']) or 'None'}")
    print(f"✗ AVOID: {', '.join(results['summary']['avoid']) or 'None'}")
    
    return model


if __name__ == "__main__":
    model = main()
