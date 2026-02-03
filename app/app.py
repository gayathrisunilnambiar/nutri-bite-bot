"""
Nutri-Bite Bot Web Application
===============================
Flask-based web interface for CBC-based ingredient quantity recommendations.

Author: Nutri-Bite Bot Development Team
Version: 2.0.0
"""

import os
import sys
import json
import base64
from pathlib import Path

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from cbc_analyzer import CBCAnalyzer
from mistral_vision import MistralVision
from quantity_recommender import QuantityRecommender

app = Flask(__name__)
CORS(app)

# Initialize components
cbc_analyzer = CBCAnalyzer()
mistral_vision = MistralVision()
quantity_recommender = QuantityRecommender()


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze CBC report and pantry image, return ingredient recommendations.
    
    Expected JSON body:
    {
        "lab_values": {
            "egfr": 45,
            "creatinine": 2.1,
            "potassium": 5.2,
            "sodium": 142,
            "glucose": 180,
            "hba1c": 7.5
        },
        "conditions": {
            "diabetes_t1": true,
            "hypertension": true,
            "ckd": true
        },
        "pantry_image": "base64_encoded_image_data",  // Optional
        "ingredients_text": "potato, apple, chicken"   // Optional alternative
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract lab values
        lab_values = data.get('lab_values', {})
        conditions = data.get('conditions', {})
        
        # 1. Analyze CBC report
        cbc_report = cbc_analyzer.parse_lab_values(
            egfr=lab_values.get('egfr'),
            creatinine=lab_values.get('creatinine'),
            potassium=lab_values.get('potassium'),
            sodium=lab_values.get('sodium'),
            glucose=lab_values.get('glucose'),
            hba1c=lab_values.get('hba1c'),
            has_diabetes_t1=conditions.get('diabetes_t1', False),
            has_hypertension=conditions.get('hypertension', False),
            has_ckd=conditions.get('ckd', False)
        )
        
        cbc_summary = cbc_analyzer.get_summary()
        
        # 2. Detect ingredients from image or text
        ingredients = []
        
        pantry_image = data.get('pantry_image')
        ingredients_text = data.get('ingredients_text', '')
        
        if pantry_image:
            # Analyze image with Mistral Vision
            detected = mistral_vision.analyze_base64(pantry_image)
            ingredients = [
                {'name': ing.name, 'detected_quantity_g': ing.estimated_quantity_g}
                for ing in detected
            ]
        elif ingredients_text:
            # Parse text input
            items = [item.strip() for item in ingredients_text.split(',') if item.strip()]
            ingredients = [
                {'name': item, 'detected_quantity_g': 200}  # Default 200g
                for item in items
            ]
        
        # 3. Calculate quantity recommendations
        quantity_recommender.set_limits_for_conditions(
            has_diabetes=conditions.get('diabetes_t1', False),
            has_hypertension=conditions.get('hypertension', False),
            has_ckd=conditions.get('ckd', False),
            egfr=lab_values.get('egfr'),
            current_potassium=lab_values.get('potassium')
        )
        
        recommendations = []
        if ingredients:
            recs = quantity_recommender.recommend_quantities(ingredients)
            recommendations = quantity_recommender.to_json(recs)
        
        # 4. Build response
        response = {
            'success': True,
            'cbc_analysis': cbc_summary,
            'detected_ingredients': len(ingredients),
            'recommendations': recommendations,
            'daily_limits': {
                'potassium_mg': quantity_recommender.daily_limits.potassium_mg,
                'sodium_mg': quantity_recommender.daily_limits.sodium_mg,
                'phosphorus_mg': quantity_recommender.daily_limits.phosphorus_mg
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'mistral_mode': 'mock' if mistral_vision.mock_mode else 'api'
    })


if __name__ == '__main__':
    print("=" * 60)
    print("NUTRI-BITE BOT - Ingredient Quantity Recommender")
    print("=" * 60)
    print(f"Mistral Vision: {'Mock Mode' if mistral_vision.mock_mode else 'API Mode'}")
    print("Starting server at http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, port=5000)
