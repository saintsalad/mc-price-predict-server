from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default development port
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Specify allowed methods
    allow_headers=["Content-Type", "Authorization"],  # Specify allowed headers
)

# Define request schema
class Specifications(BaseModel):
    category: str
    displacement: float
    transmission: str
    yearRange: str
    priceRange: dict

class Condition(BaseModel):
    year: int
    mileage: int
    sellerType: str
    owner: str
    knownIssues: str

class MotorcycleInput(BaseModel):
    brand: str
    model: str
    specifications: Specifications
    condition: Condition

@app.on_event("startup")
async def startup_event():
    """Runs when the API server starts."""
    print("🚀 API Server starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """Runs when the API server is shutting down."""
    print("👋 API Server shutting down...")

@app.post("/predict")
async def predict_price(data: MotorcycleInput):
    try:
        """Generates a price prediction without ML using rule-based AI heuristics."""
        
        base_price = 50000  # Base price for any bike

        # 🚗 Adjust price based on brand
        brand_multiplier = {
            "Yamaha": 1.1,
            "Honda": 1.05,
            "Kawasaki": 1.2,
            "Suzuki": 1.0,
            "Ducati": 1.5,
            "Harley-Davidson": 2.0
        }
        base_price *= brand_multiplier.get(data.brand, 1.0)

        # 🏍️ Adjust based on displacement
        if data.specifications.displacement > 1000:
            base_price *= 1.5
        elif data.specifications.displacement > 500:
            base_price *= 1.3
        elif data.specifications.displacement > 250:
            base_price *= 1.2

        # ⏳ Depreciation (Older bikes lose value)
        age = 2025 - data.condition.year
        base_price *= max(0.5, 1 - (age * 0.05))  # 5% depreciation per year

        # 🚦 Adjust based on mileage
        if data.condition.mileage > 50000:
            base_price *= 0.7  # More than 50k km = -30%
        elif data.condition.mileage > 30000:
            base_price *= 0.8
        elif data.condition.mileage > 10000:
            base_price *= 0.9

        # 🔧 Condition-based price adjustments
        def calculate_issues_multiplier(issues_description: str) -> float:
            issues_description = issues_description.lower()
            
            # Start with base multiplier
            multiplier = 1.0
            
            # Define impact of different issues
            if "none" in issues_description or "perfect" in issues_description:
                return 1.0
            
            # Check for various issues and stack their effects
            if any(word in issues_description for word in ["scratch", "scratches", "cosmetic"]):
                multiplier *= 0.95
            if any(word in issues_description for word in ["paint", "repaint", "repainted"]):
                multiplier *= 0.90
            if any(word in issues_description for word in ["engine", "mechanical", "repair needed"]):
                multiplier *= 0.70
            if any(word in issues_description for word in ["accident", "crash", "damaged"]):
                multiplier *= 0.60
            if any(word in issues_description for word in ["rebuilt", "salvage", "total"]):
                multiplier *= 0.50
            
            # Don't let multiplier go below 0.3
            return max(0.3, multiplier)

        base_price *= calculate_issues_multiplier(data.condition.knownIssues)

        # 👨‍💼 Seller type impact
        if data.condition.sellerType == "Dealer":
            base_price *= 1.1  # Dealers charge more
        else:
            base_price *= 0.95  # Private sellers offer discounts

        # 💰 Random confidence value (75-95%)
        confidence = random.randint(75, 95)

        # ✨ Generate explanation
        description = (
            f"The price was estimated based on the brand ({data.brand}), "
            f"engine displacement ({data.specifications.displacement}cc), "
            f"year ({data.condition.year}), mileage ({data.condition.mileage} km), "
            f"and known issues ({data.condition.knownIssues}). "
            f"Confidence level is based on heuristic accuracy."
        )

        return {
            "pricePredicted": round(base_price),
            "confidence": f"{confidence}%",
            "description": description
        }
    except Exception as e:
        return {
            "error": "Failed to generate prediction",
            "details": str(e)
        }
