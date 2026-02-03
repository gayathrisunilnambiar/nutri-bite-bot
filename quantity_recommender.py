"""
Ingredient Quantity Recommender
================================
Calculates safe ingredient quantities based on clinical constraints and USDA nutrient data.

For patients with:
- Diabetes Type 1: Carbohydrate awareness
- Hypertension: Sodium restriction (<1500mg/day)
- CKD: Potassium (<2000mg/day), Phosphorus, Protein limits

Author: Nutri-Bite Bot Development Team
Version: 2.0.0
"""

import json
import logging
import requests
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationStatus(Enum):
    """Status of ingredient recommendation."""
    SAFE = "safe"
    LIMITED = "limited"
    PROHIBITED = "prohibited"


@dataclass
class NutrientInfo:
    """Nutrient content per 100g from USDA."""
    potassium_mg: float = 0
    sodium_mg: float = 0
    phosphorus_mg: float = 0
    carbohydrates_g: float = 0
    protein_g: float = 0
    calories: float = 0


@dataclass
class IngredientRecommendation:
    """Safe quantity recommendation for an ingredient."""
    name: str
    detected_quantity_g: float
    max_allowed_g: float
    status: RecommendationStatus
    limiting_nutrient: Optional[str] = None
    warning: Optional[str] = None
    nutrients_per_100g: Optional[NutrientInfo] = None


@dataclass
class DailyLimits:
    """Daily nutrient limits based on conditions."""
    potassium_mg: float = 4700  # General population
    sodium_mg: float = 2300     # General population
    phosphorus_mg: float = 1250  # General population
    carbohydrates_g: float = 300  # General (varies)
    protein_g: float = 56        # Average adult


class USDANutrientLookup:
    """
    Looks up nutrient data from USDA FoodData Central API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize USDA client.
        
        Args:
            api_key: USDA FDC API key. Uses DEMO_KEY if not provided.
        """
        self.api_key = api_key or "DEMO_KEY"
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        self.cache: Dict[str, NutrientInfo] = {}
        
        # Built-in nutrient data for common ingredients (fallback)
        self.builtin_data = {
            'potato': NutrientInfo(potassium_mg=425, sodium_mg=6, phosphorus_mg=57, carbohydrates_g=17, protein_g=2, calories=77),
            'sweet potato': NutrientInfo(potassium_mg=475, sodium_mg=55, phosphorus_mg=47, carbohydrates_g=20, protein_g=1.6, calories=86),
            'banana': NutrientInfo(potassium_mg=358, sodium_mg=1, phosphorus_mg=22, carbohydrates_g=23, protein_g=1.1, calories=89),
            'apple': NutrientInfo(potassium_mg=107, sodium_mg=1, phosphorus_mg=11, carbohydrates_g=14, protein_g=0.3, calories=52),
            'orange': NutrientInfo(potassium_mg=181, sodium_mg=0, phosphorus_mg=14, carbohydrates_g=12, protein_g=0.9, calories=47),
            'tomato': NutrientInfo(potassium_mg=237, sodium_mg=5, phosphorus_mg=24, carbohydrates_g=3.9, protein_g=0.9, calories=18),
            'spinach': NutrientInfo(potassium_mg=558, sodium_mg=79, phosphorus_mg=49, carbohydrates_g=3.6, protein_g=2.9, calories=23),
            'cabbage': NutrientInfo(potassium_mg=170, sodium_mg=18, phosphorus_mg=26, carbohydrates_g=5.8, protein_g=1.3, calories=25),
            'carrot': NutrientInfo(potassium_mg=320, sodium_mg=69, phosphorus_mg=35, carbohydrates_g=10, protein_g=0.9, calories=41),
            'broccoli': NutrientInfo(potassium_mg=316, sodium_mg=33, phosphorus_mg=66, carbohydrates_g=7, protein_g=2.8, calories=34),
            'cauliflower': NutrientInfo(potassium_mg=299, sodium_mg=30, phosphorus_mg=44, carbohydrates_g=5, protein_g=1.9, calories=25),
            'lettuce': NutrientInfo(potassium_mg=194, sodium_mg=28, phosphorus_mg=29, carbohydrates_g=2.9, protein_g=1.4, calories=15),
            'cucumber': NutrientInfo(potassium_mg=147, sodium_mg=2, phosphorus_mg=24, carbohydrates_g=3.6, protein_g=0.7, calories=15),
            'chicken breast': NutrientInfo(potassium_mg=256, sodium_mg=74, phosphorus_mg=196, carbohydrates_g=0, protein_g=31, calories=165),
            'chicken': NutrientInfo(potassium_mg=256, sodium_mg=74, phosphorus_mg=196, carbohydrates_g=0, protein_g=31, calories=165),
            'fish': NutrientInfo(potassium_mg=363, sodium_mg=59, phosphorus_mg=252, carbohydrates_g=0, protein_g=20, calories=206),
            'salmon': NutrientInfo(potassium_mg=363, sodium_mg=59, phosphorus_mg=252, carbohydrates_g=0, protein_g=20, calories=206),
            'egg': NutrientInfo(potassium_mg=138, sodium_mg=142, phosphorus_mg=198, carbohydrates_g=0.7, protein_g=13, calories=155),
            'rice': NutrientInfo(potassium_mg=35, sodium_mg=1, phosphorus_mg=43, carbohydrates_g=28, protein_g=2.7, calories=130),
            'bread': NutrientInfo(potassium_mg=115, sodium_mg=450, phosphorus_mg=102, carbohydrates_g=49, protein_g=9, calories=265),
            'oats': NutrientInfo(potassium_mg=362, sodium_mg=2, phosphorus_mg=410, carbohydrates_g=66, protein_g=17, calories=389),
            'milk': NutrientInfo(potassium_mg=132, sodium_mg=43, phosphorus_mg=84, carbohydrates_g=5, protein_g=3.4, calories=42),
            'cheese': NutrientInfo(potassium_mg=98, sodium_mg=621, phosphorus_mg=512, carbohydrates_g=1.3, protein_g=25, calories=403),
            'beans': NutrientInfo(potassium_mg=406, sodium_mg=2, phosphorus_mg=138, carbohydrates_g=22, protein_g=9, calories=127),
            'lentils': NutrientInfo(potassium_mg=369, sodium_mg=2, phosphorus_mg=180, carbohydrates_g=20, protein_g=9, calories=116),
        }
    
    def get_nutrients(self, ingredient_name: str) -> NutrientInfo:
        """
        Get nutrient info for an ingredient.
        
        Args:
            ingredient_name: Name of the ingredient
            
        Returns:
            NutrientInfo with values per 100g
        """
        name_lower = ingredient_name.lower().strip()
        
        # Check cache
        if name_lower in self.cache:
            return self.cache[name_lower]
        
        # Check built-in data first
        for key, data in self.builtin_data.items():
            if key in name_lower or name_lower in key:
                self.cache[name_lower] = data
                return data
        
        # Try USDA API
        nutrients = self._fetch_from_usda(name_lower)
        if nutrients:
            self.cache[name_lower] = nutrients
            return nutrients
        
        # Return default if not found
        logger.warning(f"No nutrient data for '{ingredient_name}', using default")
        return NutrientInfo()
    
    def _fetch_from_usda(self, query: str) -> Optional[NutrientInfo]:
        """Fetch nutrient data from USDA API."""
        try:
            # Search for food
            search_url = f"{self.base_url}/foods/search"
            params = {
                'api_key': self.api_key,
                'query': query,
                'pageSize': 1,
                'dataType': ['Foundation', 'SR Legacy']
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            foods = data.get('foods', [])
            
            if not foods:
                return None
            
            fdc_id = foods[0].get('fdcId')
            
            # Get detailed nutrients
            detail_url = f"{self.base_url}/food/{fdc_id}"
            response = requests.get(detail_url, params={'api_key': self.api_key}, timeout=10)
            response.raise_for_status()
            
            food_data = response.json()
            
            # Extract nutrients
            nutrients = NutrientInfo()
            
            for nutrient in food_data.get('foodNutrients', []):
                name = nutrient.get('nutrient', {}).get('name', '').lower()
                value = nutrient.get('amount', 0) or 0
                
                if 'potassium' in name:
                    nutrients.potassium_mg = value
                elif 'sodium' in name:
                    nutrients.sodium_mg = value
                elif 'phosphorus' in name:
                    nutrients.phosphorus_mg = value
                elif 'carbohydrate' in name:
                    nutrients.carbohydrates_g = value
                elif 'protein' in name:
                    nutrients.protein_g = value
                elif 'energy' in name and 'kcal' in name.lower():
                    nutrients.calories = value
            
            # Rate limiting for DEMO_KEY
            if self.api_key == "DEMO_KEY":
                time.sleep(0.5)
            
            return nutrients
            
        except Exception as e:
            logger.error(f"USDA API error for '{query}': {e}")
            return None


class QuantityRecommender:
    """
    Recommends safe ingredient quantities based on clinical constraints.
    """
    
    # High-potassium foods that are restricted for CKD
    HIGH_K_FOODS = {'potato', 'sweet potato', 'banana', 'orange', 'tomato', 'spinach'}
    
    # High-sodium foods restricted for hypertension
    HIGH_NA_FOODS = {'cheese', 'bread', 'processed meat', 'soy sauce', 'pickle'}
    
    # High-phosphorus foods (problematic for advanced CKD)
    HIGH_P_FOODS = {'cheese', 'milk', 'oats', 'beans', 'lentils', 'fish'}
    
    def __init__(self, usda_api_key: Optional[str] = None):
        """Initialize recommender with USDA lookup."""
        self.usda = USDANutrientLookup(api_key=usda_api_key)
        self.daily_limits = DailyLimits()
    
    def set_limits_for_conditions(
        self,
        has_diabetes: bool = False,
        has_hypertension: bool = False,
        has_ckd: bool = False,
        egfr: Optional[float] = None,
        current_potassium: Optional[float] = None
    ):
        """
        Set daily limits based on patient conditions.
        
        Args:
            has_diabetes: Type 1 Diabetes
            has_hypertension: Hypertension
            has_ckd: Chronic Kidney Disease
            egfr: eGFR value (for CKD staging)
            current_potassium: Current serum K+ (mEq/L)
        """
        # Start with general limits
        limits = DailyLimits()
        
        # CKD restrictions (most restrictive)
        if has_ckd:
            if egfr is not None and egfr < 60:
                limits.potassium_mg = 2000  # KDOQI guideline
                limits.phosphorus_mg = 1000
                
                if egfr < 30:  # Stage 4-5
                    limits.potassium_mg = 1500
                    limits.phosphorus_mg = 800
            
            # Extra restriction if K+ is elevated
            if current_potassium is not None and current_potassium > 5.0:
                limits.potassium_mg = min(limits.potassium_mg, 1500)
        
        # Hypertension - sodium restriction
        if has_hypertension:
            limits.sodium_mg = 1500  # AHA recommendation
        
        # Diabetes - carb awareness (not strict limit, but flag)
        if has_diabetes:
            limits.carbohydrates_g = 200  # Conservative daily target
        
        self.daily_limits = limits
        logger.info(f"Set limits: K+={limits.potassium_mg}mg, Na+={limits.sodium_mg}mg")
    
    def recommend_quantities(
        self,
        ingredients: List[Dict],  # [{name, detected_quantity_g}]
        meals_per_day: int = 3
    ) -> List[IngredientRecommendation]:
        """
        Calculate safe quantities for each ingredient.
        
        Args:
            ingredients: List of detected ingredients with quantities
            meals_per_day: Number of meals to distribute nutrients across
            
        Returns:
            List of recommendations with max allowed quantities
        """
        recommendations = []
        
        # Per-meal limits
        per_meal_k = self.daily_limits.potassium_mg / meals_per_day
        per_meal_na = self.daily_limits.sodium_mg / meals_per_day
        per_meal_p = self.daily_limits.phosphorus_mg / meals_per_day
        
        for item in ingredients:
            name = item.get('name', '').lower().strip()
            detected_g = item.get('detected_quantity_g', 0)
            
            # Get nutrient info
            nutrients = self.usda.get_nutrients(name)
            
            # Calculate max allowed based on each limiting nutrient
            max_by_k = self._calc_max_by_nutrient(
                nutrients.potassium_mg, per_meal_k, name, self.HIGH_K_FOODS
            )
            max_by_na = self._calc_max_by_nutrient(
                nutrients.sodium_mg, per_meal_na, name, self.HIGH_NA_FOODS
            )
            max_by_p = self._calc_max_by_nutrient(
                nutrients.phosphorus_mg, per_meal_p, name, self.HIGH_P_FOODS
            )
            
            # Most restrictive limit wins
            limits = [
                (max_by_k, 'potassium'),
                (max_by_na, 'sodium'),
                (max_by_p, 'phosphorus')
            ]
            
            max_allowed, limiting_nutrient = min(limits, key=lambda x: x[0])
            
            # Determine status
            if max_allowed == 0:
                status = RecommendationStatus.PROHIBITED
                warning = f"Prohibited: High {limiting_nutrient} content"
            elif max_allowed < detected_g:
                status = RecommendationStatus.LIMITED
                warning = f"Limit to {max_allowed:.0f}g (high {limiting_nutrient})"
            else:
                status = RecommendationStatus.SAFE
                warning = None
                limiting_nutrient = None
            
            recommendations.append(IngredientRecommendation(
                name=name,
                detected_quantity_g=detected_g,
                max_allowed_g=max_allowed,
                status=status,
                limiting_nutrient=limiting_nutrient,
                warning=warning,
                nutrients_per_100g=nutrients
            ))
        
        # Sort by status (prohibited first, then limited, then safe)
        status_order = {
            RecommendationStatus.PROHIBITED: 0,
            RecommendationStatus.LIMITED: 1,
            RecommendationStatus.SAFE: 2
        }
        recommendations.sort(key=lambda x: status_order[x.status])
        
        return recommendations
    
    def _calc_max_by_nutrient(
        self,
        nutrient_per_100g: float,
        per_meal_limit: float,
        ingredient_name: str,
        high_risk_set: set
    ) -> float:
        """Calculate max grams allowed based on a nutrient limit."""
        
        # Check if in high-risk category
        is_high_risk = any(hr in ingredient_name for hr in high_risk_set)
        
        if nutrient_per_100g <= 0:
            return 1000  # No limit if nutrient not present
        
        # Calculate max based on getting 30% of per-meal limit from one ingredient
        # (leave room for other ingredients)
        target_nutrient = per_meal_limit * 0.3
        
        # For high-risk foods in CKD, be more restrictive
        if is_high_risk:
            target_nutrient = per_meal_limit * 0.15
        
        max_g = (target_nutrient / nutrient_per_100g) * 100
        
        # Minimum 0, max 500g
        return max(0, min(500, max_g))
    
    def format_recommendations(self, recommendations: List[IngredientRecommendation]) -> str:
        """Format recommendations for display."""
        lines = [
            "=" * 60,
            "INGREDIENT QUANTITY RECOMMENDATIONS",
            "=" * 60,
            f"Daily Limits: K+={self.daily_limits.potassium_mg}mg, Na+={self.daily_limits.sodium_mg}mg",
            ""
        ]
        
        # Group by status
        prohibited = [r for r in recommendations if r.status == RecommendationStatus.PROHIBITED]
        limited = [r for r in recommendations if r.status == RecommendationStatus.LIMITED]
        safe = [r for r in recommendations if r.status == RecommendationStatus.SAFE]
        
        if prohibited:
            lines.append("⛔ PROHIBITED:")
            for r in prohibited:
                lines.append(f"   {r.name}: 0g ({r.warning})")
            lines.append("")
        
        if limited:
            lines.append("⚠️  LIMITED:")
            for r in limited:
                lines.append(f"   {r.name}: max {r.max_allowed_g:.0f}g (detected: {r.detected_quantity_g:.0f}g)")
                if r.nutrients_per_100g:
                    n = r.nutrients_per_100g
                    lines.append(f"      └─ per 100g: K+={n.potassium_mg:.0f}mg, Na+={n.sodium_mg:.0f}mg")
            lines.append("")
        
        if safe:
            lines.append("✅ SAFE:")
            for r in safe:
                lines.append(f"   {r.name}: up to {r.max_allowed_g:.0f}g")
        
        return "\n".join(lines)
    
    def to_json(self, recommendations: List[IngredientRecommendation]) -> List[Dict]:
        """Convert recommendations to JSON-serializable format."""
        return [
            {
                'name': r.name,
                'detected_quantity_g': r.detected_quantity_g,
                'max_allowed_g': r.max_allowed_g,
                'status': r.status.value,
                'limiting_nutrient': r.limiting_nutrient,
                'warning': r.warning,
                'nutrients_per_100g': asdict(r.nutrients_per_100g) if r.nutrients_per_100g else None
            }
            for r in recommendations
        ]


# Demo usage
if __name__ == "__main__":
    # Initialize recommender
    recommender = QuantityRecommender()
    
    # Set limits for CKD + Hypertension patient
    recommender.set_limits_for_conditions(
        has_diabetes=True,
        has_hypertension=True,
        has_ckd=True,
        egfr=45,
        current_potassium=5.2
    )
    
    # Sample detected ingredients
    ingredients = [
        {'name': 'potato', 'detected_quantity_g': 500},
        {'name': 'banana', 'detected_quantity_g': 300},
        {'name': 'apple', 'detected_quantity_g': 300},
        {'name': 'cabbage', 'detected_quantity_g': 400},
        {'name': 'chicken breast', 'detected_quantity_g': 600},
        {'name': 'tomato', 'detected_quantity_g': 200},
        {'name': 'spinach', 'detected_quantity_g': 100},
    ]
    
    # Get recommendations
    recommendations = recommender.recommend_quantities(ingredients)
    
    # Display
    print(recommender.format_recommendations(recommendations))
