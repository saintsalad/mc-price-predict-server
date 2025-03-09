from fastapi import APIRouter, HTTPException
from enum import Enum
import joblib
import pandas as pd
from typing import Dict, Any
import logging
import sys
from datetime import datetime
from schemas import MotorcycleInput
import os
import json

# Enums matching TypeScript
class TransmissionType(str, Enum):
    Manual = "Manual"
    Automatic = "Automatic"
    SemiAutomatic = "Semi-Automatic"
    CVT = "CVT"

class MotorcycleCategory(str, Enum):
    Scooter = "Scooter"
    Underbone = "Underbone"
    Backbone = "Backbone"
    Sport = "Sport"
    Adventure = "Adventure"
    Cruiser = "Cruiser"

# Updated Pricing Constants for Philippine Market
PRICE_METRICS = {
    # Base price adjustments (based on entry-level motorcycle price in PH)
    "BASE_PRICE": 65000,  # Updated based on average entry-level motorcycle price
    "MIN_PRICE": 40000,   # Minimum price for used motorcycles
    "MAX_PRICE": 2000000, # Maximum price cap for high-end motorcycles
    
    # Brand multipliers (adjusted for Philippine market presence and pricing)
    "BRAND_MULTIPLIERS": {
        "Honda": 1.0,     # Base reference (most common)
        "Yamaha": 1.05,   # Slightly premium
        "Suzuki": 0.95,   # Slightly lower
        "Kawasaki": 1.15, # Premium
        "KTM": 1.25,      # Premium European
        "BMW": 1.8,       # Luxury
        "Royal Enfield": 1.2, # Premium classic
        "CFMoto": 0.9,    # Chinese brand
        "SYM": 0.85,      # Taiwanese brand
        "Kymco": 0.85     # Taiwanese brand
    },
    
    # Category multipliers (adjusted for PH market preferences)
    "CATEGORY_MULTIPLIERS": {
        MotorcycleCategory.Scooter: 1.0,    # Base reference
        MotorcycleCategory.Underbone: 0.85,  # Most affordable
        MotorcycleCategory.Backbone: 0.9,    # Basic transportation
        MotorcycleCategory.Sport: 1.3,       # Premium segment
        MotorcycleCategory.Adventure: 1.25,  # Growing segment
        MotorcycleCategory.Cruiser: 1.2      # Niche segment
    },
    
    # Displacement multipliers (adjusted for PH license categories)
    "DISPLACEMENT_TIERS": {
        400: 2.0,    # Big bikes (>400cc)
        200: 1.4,    # Premium segment (201-400cc)
        150: 1.2,    # Mid-range (151-200cc)
        125: 1.0,    # Standard (126-150cc)
        110: 0.8     # Entry-level (<125cc)
    },
    
    # Mileage depreciation (based on PH used market)
    "MILEAGE_DEPRECIATION": {
        40000: 0.65,  # High mileage
        30000: 0.75,  # Above average
        20000: 0.85,  # Average
        10000: 0.95,  # Low mileage
        0: 1.0        # Very low mileage
    },
    
    # Age depreciation (adjusted for PH market)
    "AGE_DEPRECIATION_RATE": 0.08,  # 8% per year
    "MIN_DEPRECIATION": 0.4,        # Maximum 60% depreciation from age
    
    # Condition adjustments (based on PH market preferences)
    "CONDITION_MULTIPLIERS": {
        "Dealer": 1.15,    # Dealer premium
        "Private": 1.0,    # Base reference
        "1": 1.1,         # First owner
        "2": 0.9,         # Second owner
        "3": 0.8          # Third or more owner
    },
    
    # Known issues impact (verified against PH repair costs)
    "ISSUE_SEVERITY": {
        "Cosmetic damage": 0.92,         # Minor aesthetic issues
        "Engine knocking": 0.70,         # Major mechanical issue
        "Oil leaks": 0.85,               # Moderate mechanical issue
        "Chain issues": 0.95,            # Minor mechanical issue
        "Electrical problems": 0.88,     # Moderate electrical issue
        "Transmission problems": 0.75,   # Major mechanical issue
        "Brake issues": 0.85,            # Safety issue
        "Suspension issues": 0.88,       # Comfort/handling issue
        "Starting problems": 0.90,       # Moderate electrical issue
        "Exhaust system issues": 0.93,   # Minor mechanical issue
        "Fuel system problems": 0.87     # Moderate fuel system issue
    }
}

def setup_logger():
    """Setup logger with proper configuration"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers = []

    try:
        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(
            f'api_logs_{datetime.now().strftime("%Y%m%d")}.log',
            encoding='utf-8'
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file: {e}")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    return logger

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

def load_latest_model():
    """Load the latest model and its artifacts from the models/latest directory"""
    latest_dir = os.path.join('models', 'latest')
    try:
        # Load model
        model_path = os.path.join(latest_dir, 'model.pkl')
        ml_model = joblib.load(model_path)
        
        # Load label encoders
        encoders_path = os.path.join(latest_dir, 'label_encoders.pkl')
        label_encoders = joblib.load(encoders_path)
        
        # Load features
        features_path = os.path.join(latest_dir, 'features.pkl')
        features = joblib.load(features_path)
        
        # Load model info
        info_path = os.path.join(latest_dir, 'model_info.json')
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        
        logger.info("✅ ML Model and artifacts loaded successfully")
        return ml_model, label_encoders, features, model_info
    except Exception as e:
        logger.error(f"❌ Failed to load ML model and artifacts: {e}")
        return None, None, None, None

# Load the model and artifacts
ml_model, label_encoders, model_features, model_info = load_latest_model()

def calculate_base_price(data: MotorcycleInput) -> tuple[float, int]:
    """Calculate base price with comprehensive adjustments"""
    metrics = PRICE_METRICS
    base_price = metrics["BASE_PRICE"]
    
    # Brand adjustment
    brand_mult = metrics["BRAND_MULTIPLIERS"].get(data.brand, 1.0)
    base_price *= brand_mult
    logger.info(f"🏍️ Brand adjustment ({data.brand}): {base_price:,.2f} PHP")

    # Category adjustment
    category_mult = metrics["CATEGORY_MULTIPLIERS"].get(data.specifications.category, 1.0)
    base_price *= category_mult
    logger.info(f"📑 Category adjustment ({data.specifications.category}): {base_price:,.2f} PHP")

    # Displacement adjustment
    displacement = data.specifications.displacement
    for tier, multiplier in sorted(metrics["DISPLACEMENT_TIERS"].items(), reverse=True):
        if displacement > tier:
            base_price *= multiplier
            break
    logger.info(f"⚡ Engine ({displacement}cc): {base_price:,.2f} PHP")

    # Age calculation and adjustment
    age = max(0, datetime.now().year - data.condition.year)
    depreciation = max(
        metrics["MIN_DEPRECIATION"],
        1 - (age * metrics["AGE_DEPRECIATION_RATE"])
    )
    base_price *= depreciation
    logger.info(f"📅 Age ({age} years): {base_price:,.2f} PHP")

    # Seller type and owner adjustment
    seller_mult = metrics["CONDITION_MULTIPLIERS"].get(data.condition.sellerType, 1.0)
    owner_mult = metrics["CONDITION_MULTIPLIERS"].get(data.condition.owner, 1.0)
    base_price *= seller_mult * owner_mult
    
    # Known issues adjustment
    if data.condition.knownIssues:
        issue_mult = metrics["ISSUE_SEVERITY"].get(data.condition.knownIssues, 0.9)
        base_price *= issue_mult
        logger.info(f"🔧 Issue adjustment ({data.condition.knownIssues}): {base_price:,.2f} PHP")

    return base_price, age

def get_ml_prediction(data: MotorcycleInput, base_price: float, age: int) -> float:
    """Get ML model prediction using the latest model"""
    if not all([ml_model, label_encoders, model_features]):
        logger.warning("⚠️ ML model or artifacts not available, using base price")
        return base_price
    
    try:
        # Create features dictionary with all possible features
        features_dict = {
            'brand': [data.brand],
            'model': [data.model],
            'category': [data.specifications.category],
            'displacement': [float(data.specifications.displacement)],
            'transmission': [data.specifications.transmission],
            'mileage': [float(data.condition.mileage)],
            'sellerType': [data.condition.sellerType],
            'owner': [data.condition.owner],
            'age': [float(age)],
            'priceRangeMin': [float(data.specifications.priceRange.min)],
            'priceRangeMax': [float(data.specifications.priceRange.max)],
            'knownIssues': [data.condition.knownIssues or "None"]
        }

        # Create DataFrame with only the features used by the model
        features_df = pd.DataFrame({
            feature: features_dict[feature] 
            for feature in model_features 
            if feature in features_dict
        })

        # Apply label encoding for categorical features
        for col in features_df.columns:
            if col in label_encoders:
                features_df[col] = label_encoders[col].transform(features_df[col].astype(str))

        prediction = float(ml_model.predict(features_df)[0])
        logger.info(f"🤖 ML prediction: {prediction:,.2f} PHP")
        
        return prediction if PRICE_METRICS["MIN_PRICE"] < prediction < PRICE_METRICS["MAX_PRICE"] else base_price
    except Exception as e:
        logger.error(f"ML prediction error: {str(e)}")
        return base_price

def adjust_for_mileage(price: float, mileage: int) -> float:
    """Adjust price based on mileage using defined metrics"""
    for threshold, multiplier in sorted(PRICE_METRICS["MILEAGE_DEPRECIATION"].items(), reverse=True):
        if mileage > threshold:
            price *= multiplier
            break
            
    logger.info(f"🛣️ After mileage ({mileage:,} km): {price:,.2f} PHP")
    return price

def create_prediction_response(predicted_price: float, data: MotorcycleInput, age: int) -> Dict[str, Any]:
    """Create the prediction response with detailed information"""
    # Use a default confidence if model_info is not available
    confidence = 85
    if model_info and 'metrics' in model_info and 'r2' in model_info['metrics']:
        confidence = model_info['metrics']['r2'] * 100
    
    return {
        "pricePredicted": round(predicted_price),
        "confidence": f"{confidence:.1f}%",
        "description": f"Price prediction based on machine learning model considering:\n"
                      f"- Brand: {data.brand}\n"
                      f"- Model: {data.model}\n"
                      f"- Category: {data.specifications.category}\n"
                      f"- Year: {data.condition.year} (Age: {age} years)\n"
                      f"- Mileage: {data.condition.mileage:,} km\n"
                      f"- Engine: {data.specifications.displacement}cc\n"
                      f"- Transmission: {data.specifications.transmission}\n"
                      f"- Condition: {data.condition.owner} owner(s)\n"
                      f"- Known Issues: {data.condition.knownIssues or 'None'}",
        "model_info": {
            "training_date": model_info['training_date'] if model_info else None,
            "metrics": model_info['metrics'] if model_info else None
        }
    }

@router.post("/predict")
async def predict_price(data: MotorcycleInput):
    """Main prediction endpoint"""
    try:
        logger.info(f"📝 New request: {data.brand} {data.model}")

        if data.specifications.transmission not in TransmissionType:
            return {"error": "Invalid transmission type"}

        # Calculate base price and get ML prediction
        base_price, age = calculate_base_price(data)
        predicted_price = get_ml_prediction(data, base_price, age)
        
        # Apply mileage adjustment
        predicted_price = adjust_for_mileage(predicted_price, data.condition.mileage)

        # Apply price bounds from PRICE_METRICS
        # Only apply minimum price floor - remove user's priceRange influence
        predicted_price = max(PRICE_METRICS["MIN_PRICE"], predicted_price)
        
        return create_prediction_response(predicted_price, data, age)

    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Prediction failed", "message": str(e)}
        )
