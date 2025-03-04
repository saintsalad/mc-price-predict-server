from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sqlite3
from typing import List, Optional
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Schema definitions
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

class TrainingRecord(BaseModel):
    id: Optional[int] = None
    brand: str
    model: str
    specifications: Specifications
    condition: Condition
    predictedPrice: Optional[float] = None
    created_at: Optional[str] = None

def get_db():
    """Create a database connection"""
    conn = sqlite3.connect('motorcycle_training.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with required tables"""
    conn = get_db()
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS training_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand TEXT NOT NULL,
                model TEXT NOT NULL,
                specifications JSON NOT NULL,
                condition_data JSON NOT NULL,
                predicted_price REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise
    finally:
        conn.close()

def dict_to_db_record(record: TrainingRecord) -> tuple:
    """Convert TrainingRecord to database format"""
    return (
        record.brand,
        record.model,
        json.dumps(record.specifications.dict()),
        json.dumps(record.condition.dict()),
        record.predictedPrice,
    )

def db_record_to_dict(record: sqlite3.Row) -> dict:
    """Convert database record to TrainingRecord format"""
    record_dict = dict(record)
    return {
        "id": record_dict["id"],
        "brand": record_dict["brand"],
        "model": record_dict["model"],
        "specifications": json.loads(record_dict["specifications"]),
        "condition": json.loads(record_dict["condition_data"]),
        "predictedPrice": record_dict["predicted_price"],
        "created_at": record_dict["created_at"]
    }

# API Endpoints with Documentation

@router.post("/training", response_model=TrainingRecord)
async def create_training_record(record: TrainingRecord):
    """
    Create a new training record
    
    Route: POST /api/training
    
    Example payload:
    {
        "brand": "Honda",
        "model": "Click 125i",
        "specifications": {
            "category": "Scooter",
            "displacement": 125,
            "transmission": "CVT",
            "yearRange": "2018-2024",
            "priceRange": {
                "min": 77900,
                "max": 82900
            }
        },
        "condition": {
            "year": 2024,
            "mileage": 1000,
            "sellerType": "Dealer",
            "owner": "1",
            "knownIssues": ""
        },
        "predictedPrice": 80000
    }
    
    Returns:
    - 200: Created record with ID and timestamp
    - 500: Server error
    """
    try:
        conn = get_db()
        cursor = conn.execute('''
            INSERT INTO training_records (brand, model, specifications, condition_data, predicted_price)
            VALUES (?, ?, ?, ?, ?)
        ''', dict_to_db_record(record))
        
        conn.commit()
        
        # Fetch the created record
        created_record = conn.execute(
            'SELECT * FROM training_records WHERE id = ?', 
            (cursor.lastrowid,)
        ).fetchone()
        
        return db_record_to_dict(created_record)
    except Exception as e:
        logger.error(f"Error creating training record: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@router.get("/training", response_model=List[TrainingRecord])
async def get_training_records(
    limit: int = 10,
    offset: int = 0,
    brand: Optional[str] = None,
    category: Optional[str] = None
):
    """
    Get training records with optional filtering
    
    Route: GET /api/training
    
    Query Parameters:
    - limit: int (default: 10) - Number of records to return
    - offset: int (default: 0) - Number of records to skip
    - brand: string (optional) - Filter by motorcycle brand
    - category: string (optional) - Filter by motorcycle category
    
    Example requests:
    - GET /api/training
    - GET /api/training?limit=5&offset=0
    - GET /api/training?brand=Honda
    - GET /api/training?category=Scooter&brand=Honda
    
    Returns:
    - 200: List of training records
    - 500: Server error
    """
    try:
        conn = get_db()
        query = 'SELECT * FROM training_records WHERE 1=1'
        params = []

        if brand:
            query += ' AND brand = ?'
            params.append(brand)

        if category:
            query += ' AND json_extract(specifications, "$.category") = ?'
            params.append(category)

        query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])

        records = conn.execute(query, params).fetchall()
        return [db_record_to_dict(record) for record in records]
    except Exception as e:
        logger.error(f"Error fetching training records: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@router.get("/training/{record_id}", response_model=TrainingRecord)
async def get_training_record(record_id: int):
    """
    Get a specific training record
    
    Route: GET /api/training/{record_id}
    
    Parameters:
    - record_id: int - ID of the training record
    
    Example request:
    GET /api/training/1
    
    Returns:
    - 200: Training record
    - 404: Record not found
    - 500: Server error
    """
    try:
        conn = get_db()
        record = conn.execute(
            'SELECT * FROM training_records WHERE id = ?', 
            (record_id,)
        ).fetchone()
        
        if record is None:
            raise HTTPException(status_code=404, detail="Training record not found")
            
        return db_record_to_dict(record)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching training record: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@router.put("/training/{record_id}", response_model=TrainingRecord)
async def update_training_record(record_id: int, record: TrainingRecord):
    """
    Update a training record
    
    Route: PUT /api/training/{record_id}
    
    Parameters:
    - record_id: int - ID of the training record
    
    Example payload:
    {
        "brand": "Honda",
        "model": "Click 125i",
        "specifications": {
            "category": "Scooter",
            "displacement": 125,
            "transmission": "CVT",
            "yearRange": "2018-2024",
            "priceRange": {
                "min": 77900,
                "max": 82900
            }
        },
        "condition": {
            "year": 2024,
            "mileage": 1500,
            "sellerType": "Dealer",
            "owner": "1",
            "knownIssues": "Minor scratches"
        },
        "predictedPrice": 78000
    }
    
    Returns:
    - 200: Updated record
    - 404: Record not found
    - 500: Server error
    """
    try:
        conn = get_db()
        # Check if record exists
        existing = conn.execute(
            'SELECT id FROM training_records WHERE id = ?', 
            (record_id,)
        ).fetchone()
        
        if existing is None:
            raise HTTPException(status_code=404, detail="Training record not found")
        
        # Update record
        conn.execute('''
            UPDATE training_records 
            SET brand=?, model=?, specifications=?, condition_data=?, predicted_price=?
            WHERE id=?
        ''', dict_to_db_record(record) + (record_id,))
        
        conn.commit()
        
        # Fetch updated record
        updated = conn.execute(
            'SELECT * FROM training_records WHERE id = ?', 
            (record_id,)
        ).fetchone()
        
        return db_record_to_dict(updated)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating training record: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@router.delete("/training/{record_id}")
async def delete_training_record(record_id: int):
    """
    Delete a training record
    
    Route: DELETE /api/training/{record_id}
    
    Parameters:
    - record_id: int - ID of the training record
    
    Example request:
    DELETE /api/training/1
    
    Returns:
    - 200: {"message": "Training record deleted successfully"}
    - 404: Record not found
    - 500: Server error
    """
    try:
        conn = get_db()
        # Check if record exists
        record = conn.execute(
            'SELECT id FROM training_records WHERE id = ?', 
            (record_id,)
        ).fetchone()
        
        if record is None:
            raise HTTPException(status_code=404, detail="Training record not found")
            
        conn.execute('DELETE FROM training_records WHERE id = ?', (record_id,))
        conn.commit()
        
        return {"message": "Training record deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting training record: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# Add a count endpoint for pagination
@router.get("/training/count")
async def get_training_records_count(
    brand: Optional[str] = None,
    category: Optional[str] = None
):
    """
    Get total count of training records with optional filtering
    
    Route: GET /api/training/count
    
    Query Parameters:
    - brand: string (optional) - Filter by motorcycle brand
    - category: string (optional) - Filter by motorcycle category
    
    Example requests:
    - GET /api/training/count
    - GET /api/training/count?brand=Honda
    - GET /api/training/count?category=Scooter
    
    Returns:
    - 200: {"total": number}
    - 500: Server error
    """
    try:
        conn = get_db()
        query = 'SELECT COUNT(*) as total FROM training_records WHERE 1=1'
        params = []

        if brand:
            query += ' AND brand = ?'
            params.append(brand)

        if category:
            query += ' AND json_extract(specifications, "$.category") = ?'
            params.append(category)

        result = conn.execute(query, params).fetchone()
        return {"total": result["total"]}
    except Exception as e:
        logger.error(f"Error counting training records: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close() 