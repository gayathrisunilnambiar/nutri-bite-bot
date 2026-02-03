"""
Mistral Vision Integration
===========================
Analyzes pantry/ingredient images using Mistral's vision API (pixtral).
Detects ingredients and estimates quantities from photos.

Author: Nutri-Bite Bot Development Team
Version: 2.0.0
"""

import base64
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectedIngredient:
    """Single detected ingredient from image."""
    name: str
    estimated_quantity_g: float
    confidence: float  # 0.0 to 1.0
    notes: Optional[str] = None


class MistralVision:
    """
    Analyzes pantry images using Mistral Vision API.
    Falls back to mock mode if no API key is provided.
    """
    
    PROMPT = """Analyze this image of food ingredients/pantry items.

For each ingredient you can identify:
1. Name of the ingredient (use common names like "potato", "apple", "chicken breast")
2. Estimated quantity in grams
3. Your confidence level (low, medium, high)

Return your response as a JSON array with this exact format:
[
    {"name": "ingredient_name", "quantity_g": 100, "confidence": "high"},
    {"name": "another_ingredient", "quantity_g": 50, "confidence": "medium"}
]

Only return the JSON array, no other text. If you cannot identify any ingredients, return an empty array: []
"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Mistral Vision client.
        
        Args:
            api_key: Mistral API key. If None, uses MISTRAL_API_KEY env var.
                    If neither available, falls back to mock mode.
        """
        self.api_key = api_key or os.environ.get('MISTRAL_API_KEY')
        self.mock_mode = self.api_key is None
        
        if self.mock_mode:
            logger.warning("No Mistral API key found. Using mock mode.")
        else:
            logger.info("Mistral Vision initialized with API key")
    
    def analyze_image(self, image_path: str) -> List[DetectedIngredient]:
        """
        Analyze an image file for ingredients.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detected ingredients
        """
        if self.mock_mode:
            return self._mock_analysis(image_path)
        
        return self._real_analysis(image_path)
    
    def analyze_base64(self, image_base64: str, mime_type: str = "image/jpeg") -> List[DetectedIngredient]:
        """
        Analyze a base64-encoded image.
        
        Args:
            image_base64: Base64-encoded image data
            mime_type: MIME type of the image
            
        Returns:
            List of detected ingredients
        """
        if self.mock_mode:
            return self._mock_analysis(None)
        
        return self._real_analysis_base64(image_base64, mime_type)
    
    def _real_analysis(self, image_path: str) -> List[DetectedIngredient]:
        """Analyze image using Mistral API."""
        try:
            from mistralai import Mistral
            
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Determine MIME type
            suffix = Path(image_path).suffix.lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.webp': 'image/webp',
                '.gif': 'image/gif'
            }
            mime_type = mime_types.get(suffix, 'image/jpeg')
            
            return self._real_analysis_base64(image_data, mime_type)
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return []
    
    def _real_analysis_base64(self, image_base64: str, mime_type: str) -> List[DetectedIngredient]:
        """Analyze base64 image using Mistral API."""
        try:
            from mistralai import Mistral
            
            client = Mistral(api_key=self.api_key)
            
            # Use pixtral-large for vision
            response = client.chat.complete(
                model="pixtral-large-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": f"data:{mime_type};base64,{image_base64}"
                            },
                            {
                                "type": "text",
                                "text": self.PROMPT
                            }
                        ]
                    }
                ]
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            try:
                # Try to find JSON array in response
                if response_text.startswith('['):
                    items = json.loads(response_text)
                else:
                    # Try to extract JSON from markdown code block
                    import re
                    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if json_match:
                        items = json.loads(json_match.group())
                    else:
                        logger.warning(f"Could not parse response: {response_text[:200]}")
                        return []
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
                return []
            
            # Convert to DetectedIngredient objects
            confidence_map = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
            
            ingredients = []
            for item in items:
                conf_str = item.get('confidence', 'medium').lower()
                conf_val = confidence_map.get(conf_str, 0.5)
                
                ingredients.append(DetectedIngredient(
                    name=item.get('name', 'unknown').lower().strip(),
                    estimated_quantity_g=float(item.get('quantity_g', 100)),
                    confidence=conf_val
                ))
            
            logger.info(f"Detected {len(ingredients)} ingredients from image")
            return ingredients
            
        except ImportError:
            logger.error("mistralai package not installed. Run: pip install mistralai")
            return self._mock_analysis(None)
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            return []
    
    def _mock_analysis(self, image_path: Optional[str]) -> List[DetectedIngredient]:
        """Return mock ingredients for testing without API key."""
        logger.info("Using mock ingredient detection")
        
        # Simulated pantry items
        mock_items = [
            DetectedIngredient(
                name="potato",
                estimated_quantity_g=500,
                confidence=0.9,
                notes="Mock: russet potatoes"
            ),
            DetectedIngredient(
                name="apple",
                estimated_quantity_g=300,
                confidence=0.85,
                notes="Mock: red apples"
            ),
            DetectedIngredient(
                name="cabbage",
                estimated_quantity_g=400,
                confidence=0.8,
                notes="Mock: green cabbage"
            ),
            DetectedIngredient(
                name="chicken breast",
                estimated_quantity_g=600,
                confidence=0.9,
                notes="Mock: raw chicken"
            ),
            DetectedIngredient(
                name="carrot",
                estimated_quantity_g=250,
                confidence=0.85,
                notes="Mock: orange carrots"
            ),
            DetectedIngredient(
                name="tomato",
                estimated_quantity_g=200,
                confidence=0.8,
                notes="Mock: red tomatoes"
            ),
            DetectedIngredient(
                name="banana",
                estimated_quantity_g=300,
                confidence=0.9,
                notes="Mock: ripe bananas"
            ),
            DetectedIngredient(
                name="spinach",
                estimated_quantity_g=100,
                confidence=0.75,
                notes="Mock: fresh spinach"
            )
        ]
        
        return mock_items
    
    def format_results(self, ingredients: List[DetectedIngredient]) -> str:
        """Format detection results for display."""
        if not ingredients:
            return "No ingredients detected."
        
        lines = ["Detected Ingredients:", "-" * 40]
        
        for ing in ingredients:
            conf_pct = int(ing.confidence * 100)
            lines.append(f"  • {ing.name}: ~{ing.estimated_quantity_g:.0f}g ({conf_pct}% confidence)")
        
        return "\n".join(lines)


# Demo usage
if __name__ == "__main__":
    # Initialize (will use mock mode without API key)
    vision = MistralVision()
    
    # Test mock detection
    ingredients = vision.analyze_image("test_pantry.jpg")
    
    print("=" * 60)
    print("MISTRAL VISION - PANTRY ANALYSIS")
    print("=" * 60)
    print(f"Mode: {'Mock' if vision.mock_mode else 'API'}")
    print()
    print(vision.format_results(ingredients))
