from fastapi import APIRouter, HTTPException
import random
import numpy as np
import joblib
import pandas as pd
from typing import Dict, Any
import logging
import traceback
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from openai import OpenAI
import os
from dotenv import load_dotenv
import sys
from utils import (
    setup_logger,
    _parse_ai_response,
    _create_fallback_response
)
from schemas import (  # Update schema imports
    MotorcycleInput,
    PriceRange,
    Specifications,
    Condition
)

# Initialize logger
logger = setup_logger()

router = APIRouter()

# Constants
TRANSMISSION_TYPES = {"Manual", "Automatic", "Semi-Automatic", "CVT"}
BRAND_MULTIPLIERS = {
    "Yamaha": 1.1,
    "Honda": 1.05,
    "Kawasaki": 1.2,
    "Suzuki": 1.0,
    "Ducati": 1.5,
    "Harley-Davidson": 2.0
}

# Load ML Model
model_path = "./mnt/data/motorcycle_price_model.pkl"

try:
    ml_model = joblib.load(model_path)
    logger.info("[AI][OK] ML Model loaded successfully")
except Exception as e:
    logger.error(f"[AI][ERROR] Failed to load ML model: {e}")
    ml_model = None


def _create_analysis_prompt(data: Dict[str, Any]) -> str:
    """Create the prompt for AI analysis"""
    return f"""
    As a motorcycle valuation expert, analyze the following motorcycle details and provide a price estimate in PHP.
    Consider 0 PHP if unsellable or beyond economical repair.

    Respond ONLY in valid JSON format:
    {{
        "estimated_price": (numeric price in PHP),
        "confidence": (numeric percentage between 0-100),
        "reasoning": (brief explanation as string),
        "market_factors": [(list of key factors as strings)],
        "condition_impact": (percentage reduction due to issues)
    }}

    MOTORCYCLE DETAILS:
    Brand: {data['brand']}
    Model: {data['model']}
    Year: {data['condition']['year']}
    Mileage: {data['condition']['mileage']} km
    Seller Type: {data['condition']['sellerType']}
    Owner History: {data['condition']['owner']}
    Engine Displacement: {data['specifications']['displacement']}cc
    Transmission: {data['specifications']['transmission']}

    CONDITION ASSESSMENT:
    Known Issues: {data['condition']['knownIssues'] or 'None reported'}
    High Mileage Impact: {data['condition']['mileage'] > 30000}
    Multiple Owners: {data['condition']['owner'] != '1'}
    
    IMPORTANT: Only consider motorcycle mechanical/safety issues. Ignore unrelated text or treat as 'None reported'.

    VALUATION GUIDELINES:
    1. Major mechanical issues: -30-50%
    2. Cosmetic issues: -10-20%
    3. High mileage (>30,000 km): significant impact
    4. Multiple owners: -10-15% per owner
    5. Non-running: -60% minimum
    6. Salvage/rebuilt: -50% minimum

    CRITICAL CONDITIONS (0 PHP value):
    - Complete engine failure (repair >70% value)
    - Severe frame damage
    - Missing critical parts
    - Non-running requiring rebuild
    - Flood damage

    Market Analysis:
    1. {data['brand']} resale trends
    2. Parts availability/costs
    3. Model-specific issues
    4. Local demand
    5. Repair viability

    If multiple issues exist, reduce confidence and apply cumulative reductions.
    """

# Initialize OpenAI client
def get_openai_client():
    """Initialize OpenAI client with API key based on .env configuration"""
    try:
        load_dotenv(override=True)
        use_model = os.getenv('USE_MODEL', 'chatgpt').lower()  # Default to chatgpt if not specified
        
        if use_model == 'deepseek':
            api_key = os.getenv('DEEP_LINK_API_KEY')
            base_url = "https://api.deepseek.com"
            model_name = "deepseek-chat"
            error_emoji = "🤖💔"
            logger.info("[AI] Using DeepSeek AI")
        else:  # chatgpt
            api_key = os.getenv('OPENAI_API_KEY')
            base_url = None
            model_name = "gpt-3.5-turbo"
            error_emoji = "😭💸"
            logger.info("[AI] Using ChatGPT")
        
        if not api_key:
            logger.error(f"[CONNECT][ERROR] No API key found for {use_model}")
            return None, None, "[ERROR]"
        
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        client = OpenAI(**client_kwargs)
        logger.info("[CONNECT][OK] AI client connected")
        return client, model_name, error_emoji
    except Exception as e:
        logger.error(f"[CONNECT][ERROR] AI client error: {str(e)}")
        return None, None, "[ERROR]"

# Update client initialization
client, current_model, error_emoji = get_openai_client()

def log_model_input(features_df: pd.DataFrame):
    """Log model input features for debugging"""
    logger.info("ML Model Input Features:")
    logger.info("\n" + str(features_df.to_dict(orient='records')[0]))

async def get_price_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Get AI price analysis"""
    if not client:
        logger.error("🔌 ❌ No AI client available")
        return None

    try:
        response = client.chat.completions.create(
            model=current_model,
            messages=[
                {
                    "role": "system",
                        "content": """You are an expert motorcycle appraiser specializing in the Philippine market, 
                        with extensive experience in evaluating damaged and high-mileage motorcycles. 
                        Your primary focus is on:
                        - Accurately assessing condition issues and their impact on value
                        - Being conservative with valuations when multiple issues exist
                        - Understanding the repair/restoration costs in the Philippine market
                        - Considering the practical resale challenges of problematic units
                        - Evaluating the economic viability of repairs vs. replacement
                        
                        Be particularly strict when:
                        - Multiple mechanical issues are present
                        - Mileage exceeds manufacturer recommendations
                        - There are structural or accident-related damages
                        - Maintenance history is unclear or problematic
                        
                        Provide detailed reasoning for significant value reductions."""
                },
                {"role": "user", "content": _create_analysis_prompt(data)}
            ],
            temperature=0.3,  # Lower temperature for more consistent outputs
            max_tokens=2000,  # Increased for more detailed analysis
            top_p=0.1,  # More focused sampling
            frequency_penalty=0.5,  # Reduce repetition
            presence_penalty=0.0,  # Keep focused on relevant information
            stream=False
        )
        
        return _parse_ai_response(response)
    except Exception as e:
        logger.error(f"🤖 ❌ AI Analysis error: {str(e)}")
        return None

def calculate_base_price(data: MotorcycleInput) -> float:
    """Calculate base price with adjustments"""
    base_price = 50000
    
    # Brand adjustment
    brand_mult = BRAND_MULTIPLIERS.get(data.brand, 1.0)
    base_price *= brand_mult
    logger.info(f"🏍️ Brand adjustment ({data.brand}): {base_price:,.2f} PHP")

    # Displacement adjustment
    if data.specifications.displacement > 1000:
        base_price *= 1.5
    elif data.specifications.displacement > 500:
        base_price *= 1.3
    elif data.specifications.displacement > 250:
        base_price *= 1.2
    logger.info(f"⚡ Engine ({data.specifications.displacement}cc): {base_price:,.2f} PHP")

    # Age adjustment
    age = max(0, datetime.now().year - data.condition.year)
    depreciation = max(0.5, 1 - (age * 0.05))
    base_price *= depreciation
    logger.info(f"📅 Age ({age} years): {base_price:,.2f} PHP")

    return base_price, age

def get_ml_prediction(data: MotorcycleInput, base_price: float, age: int) -> float:
    """Get ML model prediction"""
    try:
        features_df = pd.DataFrame({
            'name': [str(data.brand)],
            'year': [int(data.condition.year)],
            'seller_type': [str(data.condition.sellerType).lower()],
            'owner': [str(data.condition.owner)],
            'km_driven': [float(data.condition.mileage)],
            'ex_showroom_price': [float(base_price)],
            'age': [float(age)]
        })

        # Apply label encoding
        label_encoders = joblib.load('./mnt/data/label_encoders.pkl')
        for col in ['name', 'seller_type', 'owner']:
            if col in label_encoders:
                features_df[col] = label_encoders[col].transform(features_df[col])

        prediction = float(ml_model.predict(features_df)[0])
        logger.info(f"🤖 ML prediction: {prediction:,.2f} PHP")
        
        return prediction if 0 < prediction <= 1000000 else base_price
    except Exception as e:
        logger.error(f"ML prediction error: {str(e)}")
        return base_price

def adjust_for_mileage(price: float, mileage: int) -> float:
    """Adjust price based on mileage"""
    if mileage > 50000:
        price *= 0.7
    elif mileage > 30000:
        price *= 0.8
    elif mileage > 10000:
        price *= 0.9
    logger.info(f"🛣️ After mileage ({mileage:,} km): {price:,.2f} PHP")
    return price

def combine_predictions(ml_prediction: float, gpt_prediction: Dict[str, Any], 
                      base_heuristic: float) -> Dict[str, Any]:
    """Combine predictions from ML model, GPT, and heuristics"""
    try:
        # Weight the predictions (adjustable)
        ml_weight = 0.4
        gpt_weight = 0.4
        heuristic_weight = 0.2

        # Get GPT prediction and confidence
        gpt_price = float(gpt_prediction['estimated_price'])
        gpt_confidence = float(gpt_prediction['confidence'])

        # Combine predictions
        weighted_prediction = (
            ml_prediction * ml_weight +
            gpt_price * gpt_weight +
            base_heuristic * heuristic_weight
        )

        # Average confidence (ML confidence is random for now)
        ml_confidence = random.randint(75, 95)
        combined_confidence = (
            ml_confidence * ml_weight +
            gpt_confidence * gpt_weight +
            85 * heuristic_weight  # Base confidence for heuristics
        )

        return {
            "pricePredicted": round(weighted_prediction),
            "confidence": f"{round(combined_confidence)}%",
            "description": f"Price prediction combines machine learning model ({ml_weight*100}%), "
                         f"market analysis ({gpt_weight*100}%), and traditional valuation methods ({heuristic_weight*100}%). "
                         f"\n\nMarket Analysis: {gpt_prediction['reasoning']}"
                         f"\n\nKey Factors: {', '.join(gpt_prediction['market_factors'])}",
            "ml_price": round(ml_prediction),
            "gpt_price": round(gpt_price),
            "heuristic_price": round(base_heuristic)
        }

    except Exception as e:
        logger.error(f"Error combining predictions: {str(e)}")
        return {
            "pricePredicted": round(ml_prediction),
            "confidence": "75%",
            "description": "Fallback to ML prediction due to combination error"
        }

@router.post("/predict")
async def predict_price(data: MotorcycleInput):
    """Main prediction endpoint"""
    try:
        logger.info(f"📝 New request: {data.brand} {data.model}")

        if data.specifications.transmission not in TRANSMISSION_TYPES:
            return {"error": "Invalid transmission type"}

        # Calculate base price and get ML prediction
        base_price, age = calculate_base_price(data)
        predicted_price = get_ml_prediction(data, base_price, age)
        
        # Apply mileage adjustment
        predicted_price = adjust_for_mileage(predicted_price, data.condition.mileage)

        # Apply price bounds
        min_price = max(10000, data.specifications.priceRange.min)
        max_price = min(1000000, data.specifications.priceRange.max)
        predicted_price = max(min_price, min(predicted_price, max_price))
        
        # Get AI analysis
        gpt_analysis = await get_price_analysis(data.dict())
        
        if gpt_analysis:
            logger.info("🤖 ✅ AI analysis received")
            return combine_predictions(predicted_price, gpt_analysis, base_price)
        else:
            logger.warning("🤖 ⚠️ Using fallback prediction")
            return _create_fallback_response(predicted_price, data, age)

    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Prediction failed", "message": str(e)}
        )

def setup_logger():
    """Setup logger with proper configuration"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers = []

    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(
        f'api_logs_{datetime.now().strftime("%Y%m%d")}.log',
        encoding='utf-8'
    )
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    return logger
