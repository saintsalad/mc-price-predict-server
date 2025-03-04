from pydantic import BaseModel

class PriceRange(BaseModel):
    min: float
    max: float

class Specifications(BaseModel):
    category: str
    displacement: float
    transmission: str
    yearRange: str
    priceRange: PriceRange

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