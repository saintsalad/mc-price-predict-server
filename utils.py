import logging
import pandas as pd # type: ignore
from typing import Dict, Any
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

def create_fallback_response(price: float, data: MotorcycleInput, age: int) -> Dict[str, Any]:
    """Create fallback response when prediction has issues"""
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