import logging
import pandas as pd
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import sys
from datetime import datetime
from schemas import MotorcycleInput

logger = logging.getLogger(__name__)

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

def log_model_input(features_df: pd.DataFrame):
    """Log model input features for debugging"""
    logger.info("ML Model Input Features:")
    logger.info("\n" + str(features_df.to_dict(orient='records')[0]))
    """Initialize OpenAI client with API key based on .env configuration"""
    try:
        load_dotenv(override=True)
        use_model = os.getenv('USE_MODEL', 'chatgpt').lower()  # Default to chatgpt if not specified
        
        if use_model == 'deepseek':
            api_key = os.getenv('DEEP_LINK_API_KEY')
            base_url = "https://api.deepseek.com"
            model_name = "deepseek-reasoner"
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
        return client, model_name, "[ERROR]"
    except Exception as e:
        logger.error(f"[CONNECT][ERROR] AI client error: {str(e)}")
        return None, None, "[ERROR]"

def _parse_ai_response(response) -> Dict[str, Any]:
    """Parse and validate AI response"""
    try:
        content = response.choices[0].message.content.strip()
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            analysis = json.loads(json_str)
            
            required_fields = ['estimated_price', 'confidence', 'reasoning', 'market_factors']
            if all(field in analysis for field in required_fields):
                analysis['estimated_price'] = float(str(analysis['estimated_price']).replace(',', ''))
                analysis['confidence'] = float(str(analysis['confidence']).replace('%', ''))
                return analysis
            
        return None
    except Exception as e:
        logger.error(f"Failed to parse AI response: {str(e)}")
        return None

def _create_fallback_response(price: float, data: MotorcycleInput, age: int) -> Dict[str, Any]:
    """Create fallback response when AI analysis fails"""
    return {
        "pricePredicted": round(price),
        "confidence": "80%",
        "description": (
            f"Prediction based on ML model and traditional valuation methods. "
            f"Brand: {data.brand}, Model: {data.model}\n"
            f"Year: {data.condition.year} (Age: {age} years)\n"
            f"Displacement: {data.specifications.displacement}cc\n"
            f"Mileage: {data.condition.mileage:,} km\n"
            f"Condition: {data.condition.knownIssues or 'No issues reported'}"
        )
    } 