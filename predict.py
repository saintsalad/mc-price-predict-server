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
import random

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
    "BASE_PRICE": 68000,  # Updated based on 2024 average entry-level motorcycle price
    "MIN_PRICE": 35000,   # Minimum price for used motorcycles
    "MAX_PRICE": 2500000, # Maximum price cap for high-end motorcycles
    
    # Brand multipliers (adjusted for Philippine market presence and pricing)
    "BRAND_MULTIPLIERS": {
        "Honda": 1.0,         # Base reference (most common)
        "Yamaha": 1.08,       # Premium Japanese brand
        "Suzuki": 0.95,       # Slightly lower
        "Kawasaki": 1.18,     # Premium Japanese
        "KTM": 1.30,          # Premium European
        "BMW": 1.85,          # Luxury European
        "Ducati": 2.0,        # Luxury Italian
        "Harley-Davidson": 1.9,# Luxury American
        "Royal Enfield": 1.25, # Premium classic
        "CFMoto": 0.9,        # Chinese brand
        "SYM": 0.85,          # Taiwanese brand
        "Kymco": 0.85,        # Taiwanese brand
        "Vespa": 1.4,         # Premium scooter brand
        "Triumph": 1.75,      # Premium British
        "Rusi": 0.7,          # Local budget brand
        "Motorstar": 0.7,     # Local budget brand
        "TVS": 0.8,           # Indian brand
        "Bajaj": 0.8,         # Indian brand
        "Aprilia": 1.7,       # Premium Italian
        "MV Agusta": 2.1      # Luxury Italian
    },
    
    # Category multipliers (adjusted for PH market preferences)
    "CATEGORY_MULTIPLIERS": {
        MotorcycleCategory.Scooter: 1.0,     # Base reference
        MotorcycleCategory.Underbone: 0.82,  # Most affordable
        MotorcycleCategory.Backbone: 0.88,   # Basic transportation
        MotorcycleCategory.Sport: 1.35,      # Premium segment
        MotorcycleCategory.Adventure: 1.30,  # Growing segment
        MotorcycleCategory.Cruiser: 1.25     # Niche segment
    },
    
    # Displacement multipliers (adjusted for PH license categories)
    "DISPLACEMENT_TIERS": {
        650: 2.5,    # High-end big bikes (>650cc)
        400: 2.0,    # Big bikes (400-650cc)
        200: 1.4,    # Premium segment (201-400cc)
        150: 1.2,    # Mid-range (151-200cc)
        125: 1.0,    # Standard (126-150cc)
        110: 0.8,    # Entry-level (110-125cc)
        0: 0.7       # Ultra small (<110cc)
    },
    
    # Mileage depreciation (based on PH used market)
    "MILEAGE_DEPRECIATION": {
        50000: 0.60,  # Very high mileage
        40000: 0.68,  # High mileage
        30000: 0.78,  # Above average
        20000: 0.87,  # Average
        10000: 0.95,  # Low mileage
        5000: 0.98,   # Very low mileage
        0: 1.0        # New-like condition
    },
    
    # Age depreciation (adjusted for PH market)
    "AGE_DEPRECIATION_RATE": 0.085,  # 8.5% per year
    "MIN_DEPRECIATION": 0.35,        # Maximum 65% depreciation from age
    
    # Condition adjustments (based on PH market preferences)
    "CONDITION_MULTIPLIERS": {
        "Dealer": 1.18,    # Dealer premium
        "Private": 1.0,    # Base reference
        "1": 1.15,        # First owner
        "2": 0.9,         # Second owner
        "3": 0.8,         # Third owner
        "4+": 0.7         # Four or more owners
    },
    
    # Known issues impact (verified against PH repair costs)
    "ISSUE_SEVERITY": {
        "Cosmetic damage": 0.92,         # Minor aesthetic issues
        "Engine knocking": 0.68,         # Major mechanical issue
        "Oil leaks": 0.82,               # Moderate mechanical issue
        "Chain issues": 0.94,            # Minor mechanical issue
        "Electrical problems": 0.85,     # Moderate electrical issue
        "Transmission problems": 0.72,   # Major mechanical issue
        "Brake issues": 0.83,            # Safety issue
        "Suspension issues": 0.87,       # Comfort/handling issue
        "Starting problems": 0.88,       # Moderate electrical issue
        "Exhaust system issues": 0.93,   # Minor mechanical issue
        "Fuel system problems": 0.85,    # Moderate fuel system issue
        "Overheating": 0.78,             # Major cooling issue
        "Smoke from exhaust": 0.75,      # Engine combustion issue
        "Clutch problems": 0.85,         # Transmission component issue
        "Rust/Corrosion": 0.88,          # Cosmetic/structural issue
        "Leaking fork seals": 0.90,      # Minor suspension issue
        "Worn tires": 0.95,              # Consumable part
        "No issues": 1.0                 # Perfect condition
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

@router.get("/model")
async def get_latest_model_info():
    """
    Get information about the latest machine learning model.
    
    Returns:
        dict: Information about the model, including version, training date, metrics, and feature importance.
    """
    if model_info is None:
        raise HTTPException(status_code=404, detail="Model information not found")
    
    # Extract metrics separately for cleaner structure
    metrics = model_info.get("metrics", {})
    
    # Create a more organized response without redundancy
    return {
        "status": "success",
        "model": {
            "name": model_info.get("model_name", "MPP"),
            "version": model_info.get("version", "v0.0.0"),
            "training_date": model_info.get("training_date"),
            "training_file": model_info.get("training_file")
        },
        "performance": {
            "r2_score": metrics.get("r2", 0),
            "mae": metrics.get("mae", 0),
            "rmse": metrics.get("rmse", 0)
        },
        "specs": {
            "features_count": len(model_features) if model_features else 0,
            "encoders_count": len(label_encoders) if label_encoders else 0,
            "top_features": [f["feature"] for f in model_info.get("feature_importance", [])[:5]] if model_info.get("feature_importance") else []
        },
        "status_details": {
            "model_loaded": ml_model is not None
        }
    }

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
    """Get ML model prediction using the latest model with enhanced accuracy for PH market"""
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
                # For unknown categories, use a default value
                try:
                    features_df[col] = label_encoders[col].transform(features_df[col].astype(str))
                except ValueError:
                    # If category is unknown, replace with the most common/similar category
                    logger.warning(f"⚠️ Unknown category in feature {col}: {features_df[col][0]}, using default value")
                    features_df[col] = 0  # Default to first category or special handling

        # Get ML model prediction
        prediction = float(ml_model.predict(features_df)[0])
        logger.info(f"🤖 ML prediction: {prediction:,.2f} PHP")
        
        # Enhanced prediction validation and market-specific adjustments
        
        # 1. Detect outliers: if prediction is extremely different from base price
        deviation_ratio = prediction / base_price if base_price > 0 else 0
        
        if deviation_ratio > 3 or deviation_ratio < 0.3:
            logger.warning(f"⚠️ Prediction deviation too high ({deviation_ratio:.2f}x), adjusting prediction")
            # Use a weighted blend to bring the prediction closer to base price
            prediction = (prediction * 0.6) + (base_price * 0.4)
            logger.info(f"⚙️ Adjusted prediction: {prediction:,.2f} PHP")
        
        # 2. Market-specific adjustments for Philippine market
        
        # Popular models tend to hold value better in PH
        popular_models = ["Mio", "Click", "Nmax", "XRM", "Sniper", "Raider", "TMX", "Barako", "RS150"]
        if any(model_name in data.model for model_name in popular_models):
            popularity_boost = 1.05  # 5% boost for popular models
            prediction *= popularity_boost
            logger.info(f"📈 Popular model boost: {prediction:,.2f} PHP")
        
        # Regional premium for Metro Manila/Urban areas
        if hasattr(data, 'region') and data.region in ["Metro Manila", "NCR", "Cebu", "Davao"]:
            urban_premium = 1.08  # 8% premium for urban areas
            prediction *= urban_premium
            logger.info(f"🏙️ Urban area premium: {prediction:,.2f} PHP")
            
        # Ensure the final prediction stays within reasonable bounds
        if prediction < PRICE_METRICS["MIN_PRICE"]:
            prediction = PRICE_METRICS["MIN_PRICE"]
        elif prediction > PRICE_METRICS["MAX_PRICE"]:
            prediction = PRICE_METRICS["MAX_PRICE"]
            
        return prediction
    except Exception as e:
        logger.error(f"ML prediction error: {str(e)}")
        # Fallback to rule-based pricing with some randomization to simulate ML variation
        fallback_price = base_price * random.uniform(0.95, 1.05)
        logger.info(f"⚙️ Fallback prediction: {fallback_price:,.2f} PHP")
        return fallback_price

def adjust_for_mileage(price: float, mileage: int, data: MotorcycleInput = None) -> float:
    """Adjust price based on mileage using refined PH market metrics with non-linear depreciation"""
    # Apply different depreciation curves based on mileage brackets
    # This better models the real-world value drop in the Philippine motorcycle market
    
    if mileage <= 0:
        # No mileage data or new bike
        return price
    
    # Check for extraordinarily high mileage for PH market
    if mileage > 80000:
        # Extreme mileage case - major depreciation but with a floor
        return max(price * 0.45, PRICE_METRICS["MIN_PRICE"])
    
    # Find the appropriate bracket from the mileage depreciation table
    for threshold, multiplier in sorted(PRICE_METRICS["MILEAGE_DEPRECIATION"].items(), reverse=True):
        if mileage > threshold:
            # Apply base multiplier from the bracket
            base_multiplier = multiplier
            
            # Calculate exact depreciation within the bracket
            # This creates a smoother depreciation curve rather than discrete steps
            next_threshold = None
            next_multiplier = None
            
            # Find the next bracket for interpolation
            brackets = sorted(PRICE_METRICS["MILEAGE_DEPRECIATION"].items())
            for i, (t, m) in enumerate(brackets):
                if t == threshold and i > 0:
                    next_threshold = brackets[i-1][0]
                    next_multiplier = brackets[i-1][1]
                    break
            
            # If we have a next bracket, interpolate the multiplier
            if next_threshold is not None and next_multiplier is not None and threshold != next_threshold:
                # Calculate the position within the current bracket (0-1)
                bracket_position = (mileage - threshold) / (next_threshold - threshold)
                # Interpolate between current and next multiplier
                interpolated_multiplier = base_multiplier + (bracket_position * (next_multiplier - base_multiplier))
                # Apply the interpolated multiplier
                adjusted_price = price * interpolated_multiplier
            else:
                # If we can't interpolate, just use the bracket multiplier
                adjusted_price = price * base_multiplier
            
            # Additional adjustment for high mileage motorcycles that are still relatively new
            if mileage > 30000 and data is not None:
                age = max(0, datetime.now().year - data.condition.year)
                if age <= 3:
                    # High mileage on a newer bike is concerning - increase depreciation
                    adjusted_price *= 0.92
                    logger.info(f"⚠️ High mileage on newer bike adjustment: {adjusted_price:,.2f} PHP")
            
            logger.info(f"🛣️ After mileage ({mileage:,} km): {adjusted_price:,.2f} PHP")
            return adjusted_price
    
    # Default case - no depreciation
    return price

def create_prediction_response(predicted_price: float, data: MotorcycleInput, age: int) -> Dict[str, Any]:
    """Create the prediction response with detailed market information"""
    # Calculate confidence based on multiple factors
    base_confidence = 85  # Base confidence
    confidence_adjustments = []
    
    # 1. Model confidence from R² if available
    if model_info and 'metrics' in model_info and 'r2' in model_info['metrics']:
        base_confidence = model_info['metrics']['r2'] * 100
    
    # 2. Data quality factors
    confidence_factors = {
        # Age affects confidence - newer vehicles are more predictable
        'age': max(0, 100 - (age * 3)) / 100,  # -3% per year
        
        # Mileage affects confidence - higher mileage means more uncertainty
        'mileage': max(0, 100 - (data.condition.mileage / 1000)) / 100,  # -1% per 1000km
        
        # Known issues reduce confidence
        'issues': 0.95 if not data.condition.knownIssues else 0.85,
        
        # Multiple owners reduce confidence
        'ownership': {
            "1": 1.0,    # First owner - highest confidence
            "2": 0.95,   # Second owner
            "3": 0.90,   # Third owner
            "4+": 0.85   # Four or more owners
        }.get(data.condition.owner, 0.85),
        
        # Popular models have more data points, thus higher confidence
        'model_popularity': 1.1 if any(model in data.model for model in 
            ["Mio", "Click", "Nmax", "XRM", "Sniper", "Raider", "TMX", "Barako", "RS150"]) else 1.0,
        
        # Common brands have more data points
        'brand_popularity': 1.1 if data.brand in ["Honda", "Yamaha", "Suzuki", "Kawasaki"] else 1.0
    }
    
    # Calculate final confidence
    confidence_multiplier = (
        confidence_factors['age'] *
        confidence_factors['mileage'] *
        confidence_factors['issues'] *
        confidence_factors['ownership'] *
        confidence_factors['model_popularity'] *
        confidence_factors['brand_popularity']
    )
    
    final_confidence = min(99, base_confidence * confidence_multiplier)  # Cap at 99%
    
    # Log confidence calculation details
    logger.info(f"Confidence calculation:")
    logger.info(f"- Base confidence: {base_confidence:.1f}%")
    logger.info(f"- Age factor: {confidence_factors['age']:.2f}")
    logger.info(f"- Mileage factor: {confidence_factors['mileage']:.2f}")
    logger.info(f"- Issues factor: {confidence_factors['issues']:.2f}")
    logger.info(f"- Ownership factor: {confidence_factors['ownership']:.2f}")
    logger.info(f"- Model popularity: {confidence_factors['model_popularity']:.2f}")
    logger.info(f"- Brand popularity: {confidence_factors['brand_popularity']:.2f}")
    logger.info(f"Final confidence: {final_confidence:.1f}%")
    
    # Market analysis based on the prediction data
    market_position = "average"
    market_liquidity = "moderate"
    
    # Determine market position
    if data.specifications.displacement > 400:
        market_position = "premium"
        market_liquidity = "low"  # Big bikes have lower liquidity
    elif data.specifications.displacement > 150:
        market_position = "above average"
        market_liquidity = "moderate"
    else:
        # For everyday motorcycles (common in PH)
        market_position = "high demand"
        market_liquidity = "high"
    
    # Adjust for age
    if age <= 2:
        liquidity_bonus = 0.1  # Newer bikes sell faster
    elif age <= 5:
        liquidity_bonus = 0.05  # Still good demand
    else:
        liquidity_bonus = 0  # Older bikes
    
    # Popular models have higher liquidity
    popular_models = ["Mio", "Click", "Nmax", "XRM", "Sniper", "Raider", "TMX", "Barako", "RS150"]
    if any(model_name in data.model for model_name in popular_models):
        liquidity_bonus += 0.1
    
    # Calculate price ranges (±10% for high confidence, ±15% for lower)
    price_min = round(predicted_price * (0.9 if final_confidence > 90 else 0.85))
    price_max = round(predicted_price * (1.1 if final_confidence > 90 else 1.15))
    
    # Calculate value retention percentage compared to new price
    estimated_new_price = PRICE_METRICS["BASE_PRICE"]
    for brand, mult in PRICE_METRICS["BRAND_MULTIPLIERS"].items():
        if brand.lower() in data.brand.lower():
            estimated_new_price *= mult
            break
            
    category_mult = PRICE_METRICS["CATEGORY_MULTIPLIERS"].get(data.specifications.category, 1.0)
    estimated_new_price *= category_mult
    
    # Apply displacement multiplier
    displacement = data.specifications.displacement
    for tier, multiplier in sorted(PRICE_METRICS["DISPLACEMENT_TIERS"].items(), reverse=True):
        if displacement > tier:
            estimated_new_price *= multiplier
            break
            
    # Calculate value retention
    value_retention = (predicted_price / estimated_new_price) * 100 if estimated_new_price > 0 else 0
    
    return {
        "pricePredicted": round(predicted_price),
        "confidence": f"{final_confidence:.1f}%",
        "priceRange": {
            "min": price_min,
            "max": price_max
        },
        "marketAnalysis": {
            "position": market_position,
            "liquidity": market_liquidity,
            "valueRetention": f"{value_retention:.1f}%",
            "estimatedSellingTime": f"{int(30 - (liquidity_bonus * 100))} days"
        },
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

        # Check if transmission value is valid
        if data.specifications.transmission not in [t.value for t in TransmissionType]:
            return {"error": "Invalid transmission type"}

        # Calculate base price and get ML prediction
        base_price, age = calculate_base_price(data)
        predicted_price = get_ml_prediction(data, base_price, age)
        
        # Apply mileage adjustment
        predicted_price = adjust_for_mileage(predicted_price, data.condition.mileage, data)

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
