# Motorcycle Price Prediction API

## Overview

This API provides accurate motorcycle price predictions for the Philippine market using both rule-based algorithms and machine learning. The system combines traditional pricing factors with advanced ML techniques to deliver reliable valuations for motorcycles based on their specifications and condition.

## How the Price Prediction Works

The prediction process involves multiple steps to ensure accuracy and reliability:

### 1. Base Price Calculation

The system first establishes a base price using the following factors:

- **Brand Adjustment**: Each motorcycle brand has a multiplier (e.g., Honda = 1.0, BMW = 1.85) that adjusts the base price according to brand value in the Philippines.
- **Category Adjustment**: Different motorcycle categories (Scooter, Sport, Adventure, etc.) have specific multipliers reflecting their market positioning.
- **Engine Displacement**: Larger engines command higher prices, with specific tiers based on Philippine motorcycle license categories (e.g., >650cc = 2.5x multiplier).
- **Age Depreciation**: The system calculates the motorcycle's age and applies graduated depreciation (approximately 8.5% per year).
- **Seller Type & Ownership**: Adjustments based on whether the seller is a dealer or private individual, and how many previous owners the motorcycle has had.
- **Known Issues**: Mechanical or cosmetic problems reduce the price based on severity (e.g., engine knocking = 68% of normal value).

### 2. Machine Learning Prediction

The base price is then refined using a trained machine learning model:

- **Feature Preparation**: The system converts motorcycle attributes into a format suitable for the ML model.
- **Categorical Encoding**: Features like brand, model, and transmission type are encoded.
- **Prediction**: The ML model generates a predicted price based on trained patterns from historical data.
- **Validation**: The prediction is checked for reasonableness; if it deviates significantly from the base price, a weighted blend is used.
- **Market-Specific Adjustments**: Popular models in the Philippines receive a small boost, and regional premiums are applied for areas like Metro Manila.

### 3. Mileage Adjustment

The prediction is further refined based on the motorcycle's mileage:

- **Graduated Depreciation**: Different mileage brackets have specific depreciation multipliers.
- **Interpolation**: For mileage between bracket levels, the system calculates a precise depreciation value.
- **Special Cases**: Very high mileage motorcycles (>80,000 km) receive additional depreciation, while high mileage on newer bikes triggers extra adjustments.

### 4. Confidence Calculation

The system calculates a confidence score for each prediction:

- **Base Confidence**: Starts with the ML model's R² value or a default of 85%.
- **Age Factor**: Newer motorcycles have more predictable prices, so confidence decreases by ~3% per year.
- **Mileage Factor**: Higher mileage increases uncertainty, reducing confidence by ~1% per 1000km.
- **Known Issues**: Reduces confidence as mechanical problems add uncertainty.
- **Ownership History**: Multiple previous owners reduce confidence.
- **Market Popularity**: Common brands and models have more data points, increasing confidence.

### 5. Market Analysis

The system provides additional market insights:

- **Market Position**: Indicates whether the motorcycle is in a premium, average, or high-demand segment.
- **Market Liquidity**: Estimates how quickly the motorcycle might sell.
- **Value Retention**: Calculates what percentage of the new price the motorcycle has retained.
- **Price Range**: Provides minimum and maximum expected price based on confidence level.
- **Estimated Selling Time**: Predicts how many days it might take to sell the motorcycle.

## Response Format

The API returns a comprehensive response with:

```json
{
  "pricePredicted": 44633,
  "confidence": "87.3%",
  "priceRange": {
    "min": 37938,
    "max": 51328
  },
  "marketAnalysis": {
    "position": "high demand",
    "liquidity": "high",
    "valueRetention": "65.6%",
    "estimatedSellingTime": "20 days"
  },
  "description": "Price prediction based on machine learning model considering:\n- Brand: Honda\n- Model: Click 125i\n- Category: Scooter\n- Year: 2019 (Age: 6 years)\n- Mileage: 31,600 km\n- Engine: 125.0cc\n- Transmission: Automatic\n- Condition: 3 owner(s)\n- Known Issues: Oil leaks",
  "model_info": {
    "training_date": "2025-03-09T18:27:43.113086",
    "metrics": {
      "mae": 3462.29,
      "mse": 19273044.76,
      "rmse": 4390.11,
      "r2": 0.982
    }
  }
}
```

## Technical Implementation

The prediction system uses:

- **FastAPI**: For the REST API interface
- **Scikit-learn**: For the machine learning models
- **Pandas**: For data manipulation
- **Joblib**: For model loading/saving

## Development and Training

The ML model is periodically retrained using real Philippine motorcycle market data to ensure accuracy. The system uses a combination of regression algorithms with hyperparameter tuning to optimize predictions.

## Confidence Metrics

The API includes confidence metrics to help users understand prediction reliability:

- **R²**: Coefficient of determination, indicating how well the model fits historical data
- **MAE**: Mean Absolute Error in PHP
- **RMSE**: Root Mean Squared Error in PHP
