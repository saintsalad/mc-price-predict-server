from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import sqlite3
from typing import List, Optional
import json
import logging
from datetime import datetime
import csv
from io import StringIO

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

class BulkDeleteRequest(BaseModel):
    ids: List[int]

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
    model: Optional[str] = None,
    category: Optional[str] = None,
    categories: Optional[str] = None
):
    """
    Get training records with optional filtering
    
    Route: GET /api/training
    
    Query Parameters:
    - limit: int (default: 10) - Number of records to return
    - offset: int (default: 0) - Number of records to skip
    - brand: string (optional) - Filter by motorcycle brand
    - model: string (optional) - Filter by motorcycle model
    - category: string (optional) - Filter by single motorcycle category
    - categories: string (optional) - Filter by multiple categories (comma-separated)
    
    Example requests:
    - GET /api/training
    - GET /api/training?limit=5&offset=0
    - GET /api/training?brand=Honda&model=Click
    - GET /api/training?category=Scooter
    - GET /api/training?categories=Scooter,Sport,Naked
    - GET /api/training?brand=Honda&categories=Scooter,Sport
    
    Returns:
    - 200: List of training records
    - 500: Server error
    """
    try:
        conn = get_db()
        query = 'SELECT * FROM training_records WHERE 1=1'
        params = []

        # Brand filter
        if brand:
            query += ' AND brand = ?'
            params.append(brand)

        # Model filter (with LIKE for partial matches)
        if model:
            query += ' AND model LIKE ?'
            params.append(f'%{model}%')

        # Category filters
        if category:
            query += ' AND json_extract(specifications, "$.category") = ?'
            params.append(category)
        elif categories:
            category_list = [cat.strip() for cat in categories.split(',')]
            placeholders = ','.join(['?' for _ in category_list])
            query += f' AND json_extract(specifications, "$.category") IN ({placeholders})'
            params.extend(category_list)

        # Add sorting and pagination
        query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])

        # Log the query for debugging
        logger.debug(f"Executing query: {query} with params: {params}")

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
@router.get("/training-count")
async def get_training_records_count(
    brand: Optional[str] = None,
    model: Optional[str] = None,
    category: Optional[str] = None,
    categories: Optional[str] = None
):
    """
    Get total count of training records with optional filtering
    
    Route: GET /api/training/count
    
    Query Parameters:
    - brand: string (optional) - Filter by motorcycle brand
    - model: string (optional) - Filter by motorcycle model
    - category: string (optional) - Filter by single motorcycle category
    - categories: string (optional) - Filter by multiple categories (comma-separated)
    
    Example requests:
    - GET /api/training/count
    - GET /api/training/count?brand=Honda&model=Click
    - GET /api/training/count?category=Scooter
    - GET /api/training/count?categories=Scooter,Sport
    
    Returns:
    - 200: {"total": number}
    - 500: Server error
    """
    try:
        conn = get_db()
        query = 'SELECT COUNT(*) as total FROM training_records WHERE 1=1'
        params = []

        # Brand filter
        if brand:
            query += ' AND brand = ?'
            params.append(brand)

        # Model filter (with LIKE for partial matches)
        if model:
            query += ' AND model LIKE ?'
            params.append(f'%{model}%')

        # Category filters
        if category:
            query += ' AND json_extract(specifications, "$.category") = ?'
            params.append(category)
        elif categories:
            category_list = [cat.strip() for cat in categories.split(',')]
            placeholders = ','.join(['?' for _ in category_list])
            query += f' AND json_extract(specifications, "$.category") IN ({placeholders})'
            params.extend(category_list)

        # Log the query for debugging
        logger.debug(f"Executing count query: {query} with params: {params}")

        result = conn.execute(query, params).fetchone()
        return {"total": result["total"]}
    except Exception as e:
        logger.error(f"Error counting training records: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@router.post("/training/bulk", response_model=dict)
async def bulk_insert_training_records(file: UploadFile = File(...)):
    """
    Bulk insert training records from a CSV file
    
    Route: POST /api/training/bulk
    
    Parameters:
    - file: CSV file upload
    
    CSV Format Required Headers:
    brand,model,category,displacement,transmission,yearRange,priceRangeMin,priceRangeMax,year,mileage,sellerType,owner,knownIssues,predictedPrice
    
    Example CSV content:
    brand,model,category,displacement,transmission,yearRange,priceRangeMin,priceRangeMax,year,mileage,sellerType,owner,knownIssues,predictedPrice
    Honda,Click 125i,Scooter,125,CVT,2018-2024,77900,82900,2024,1000,Dealer,1,,80000
    Yamaha,NMAX,Scooter,155,CVT,2020-2024,119900,126900,2023,500,Dealer,1,Minor scratches,122000
    
    Returns:
    - 200: {
        "status": "success",
        "message": "Successfully inserted X records",
        "errors": [...],
        "total_rows": X,
        "successful_inserts": X,
        "failed_inserts": X
    }
    - 400: {
        "status": "error",
        "message": "CSV validation failed",
        "detail": "error details",
        "errors": [...]
    }
    - 500: {
        "status": "error",
        "message": "Server error",
        "detail": "error details",
        "errors": [...]
    }
    """
    try:
        # Read CSV file content
        content = await file.read()
        csv_content = StringIO(content.decode())
        
        try:
            csv_reader = csv.DictReader(csv_content)
            # Validate headers
            required_headers = {'brand', 'model', 'category', 'displacement', 'transmission', 
                              'yearRange', 'priceRangeMin', 'priceRangeMax', 'year', 'mileage', 
                              'sellerType', 'owner', 'knownIssues', 'predictedPrice'}
            actual_headers = set(csv_reader.fieldnames) if csv_reader.fieldnames else set()
            missing_headers = required_headers - actual_headers
            
            if missing_headers:
                error_msg = f"Missing required headers: {', '.join(missing_headers)}"
                logger.error(f"CSV validation error: {error_msg}")
                return {
                    "status": "error",
                    "message": "CSV validation failed",
                    "detail": error_msg,
                    "errors": [error_msg]
                }
            
        except Exception as e:
            error_msg = f"Invalid CSV format: {str(e)}"
            logger.error(f"CSV parsing error: {str(e)}")
            return {
                "status": "error",
                "message": "CSV parsing failed",
                "detail": error_msg,
                "errors": [error_msg]
            }
        
        successful_inserts = 0
        errors = []
        conn = get_db()
        total_rows = 0
        
        # Count total rows
        csv_content.seek(0)
        total_rows = sum(1 for row in csv_reader) - 1  # Subtract 1 for header
        csv_content.seek(0)
        next(csv_reader)  # Skip header row
        
        for row_idx, row in enumerate(csv_reader, start=1):
            try:
                # Log the row being processed for debugging
                logger.debug(f"Processing row {row_idx}: {row}")
                
                # Validate numeric fields before conversion
                numeric_validations = {
                    'displacement': row['displacement'],
                    'priceRangeMin': row['priceRangeMin'],
                    'priceRangeMax': row['priceRangeMax'],
                    'year': row['year'],
                    'mileage': row['mileage'],
                    'predictedPrice': row['predictedPrice']
                }
                
                for field, value in numeric_validations.items():
                    if not value or not str(value).strip():
                        raise ValueError(f"Missing required numeric value for {field}")
                
                # Create TrainingRecord object from CSV row
                record = TrainingRecord(
                    brand=row['brand'],
                    model=row['model'],
                    specifications=Specifications(
                        category=row['category'],
                        displacement=float(row['displacement']),
                        transmission=row['transmission'],
                        yearRange=row['yearRange'],
                        priceRange=PriceRange(
                            min=float(row['priceRangeMin']),
                            max=float(row['priceRangeMax'])
                        )
                    ),
                    condition=Condition(
                        year=int(row['year']),
                        mileage=int(row['mileage']),
                        sellerType=row['sellerType'],
                        owner=row['owner'],
                        knownIssues=row['knownIssues']
                    ),
                    predictedPrice=float(row['predictedPrice'])
                )
                
                # Insert record
                conn.execute('''
                    INSERT INTO training_records (brand, model, specifications, condition_data, predicted_price)
                    VALUES (?, ?, ?, ?, ?)
                ''', dict_to_db_record(record))
                
                successful_inserts += 1
                
            except ValueError as ve:
                error_msg = f"Row {row_idx}: Invalid data format - {str(ve)}"
                logger.error(error_msg)
                errors.append(error_msg)
            except KeyError as ke:
                error_msg = f"Row {row_idx}: Missing required field - {str(ke)}"
                logger.error(error_msg)
                errors.append(error_msg)
            except Exception as e:
                error_msg = f"Row {row_idx}: Unexpected error - {str(e)}"
                logger.error(f"Detailed error for row {row_idx}: {str(e)}\nRow data: {row}")
                errors.append(error_msg)
        
        conn.commit()
        
        # Log summary
        logger.info(f"Bulk insert completed: {successful_inserts} successful, {len(errors)} failed")
        if errors:
            logger.info(f"Errors encountered: {errors}")
        
        return {
            "status": "success",
            "message": f"Successfully inserted {successful_inserts} records",
            "total_rows": total_rows,
            "successful_inserts": successful_inserts,
            "failed_inserts": len(errors),
            "errors": errors
        }
        
    except Exception as e:
        error_msg = f"Error in bulk insert: {str(e)}"
        logger.error(error_msg, exc_info=True)  # This logs the full stack trace
        return {
            "status": "error",
            "message": "Server error during bulk insert",
            "detail": error_msg,
            "errors": [str(e)]
        }
    finally:
        conn.close()

@router.post("/training/delete-bulk", response_model=dict)
async def bulk_delete_training_records(request: BulkDeleteRequest):
    """
    Bulk delete multiple training records by their IDs
    
    Route: DELETE /api/training/bulk
    
    Parameters:
    - request: JSON payload containing list of record IDs to delete
    
    Example payload:
    {
        "ids": [1, 2, 3, 4, 5]
    }
    
    Returns:
    - 200: {
        "status": "success",
        "message": "Successfully deleted X records",
        "total_requested": X,
        "deleted_count": X,
        "not_found": [...],
        "errors": [...]
    }
    - 400: {
        "status": "error",
        "message": "Invalid request",
        "detail": "error details",
        "errors": [...]
    }
    - 500: {
        "status": "error",
        "message": "Server error",
        "detail": "error details",
        "errors": [...]
    }
    """
    try:
        if not request.ids:
            return {
                "status": "error",
                "message": "Invalid request",
                "detail": "No IDs provided for deletion",
                "errors": ["Empty ID list"]
            }

        conn = get_db()
        not_found = []
        errors = []
        deleted_count = 0
        total_requested = len(request.ids)

        # Check which IDs exist
        for record_id in request.ids:
            try:
                record = conn.execute(
                    'SELECT id FROM training_records WHERE id = ?', 
                    (record_id,)
                ).fetchone()
                
                if record is None:
                    not_found.append(record_id)
                    logger.warning(f"Record ID {record_id} not found")
                    continue

                # Delete the record
                conn.execute('DELETE FROM training_records WHERE id = ?', (record_id,))
                deleted_count += 1
                
            except Exception as e:
                error_msg = f"Error deleting record ID {record_id}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        conn.commit()
        
        # Log summary
        logger.info(f"Bulk delete completed: {deleted_count} deleted, {len(not_found)} not found, {len(errors)} errors")
        
        return {
            "status": "success",
            "message": f"Successfully deleted {deleted_count} records",
            "total_requested": total_requested,
            "deleted_count": deleted_count,
            "not_found": not_found,
            "errors": errors
        }

    except Exception as e:
        error_msg = f"Error in bulk delete: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "status": "error",
            "message": "Server error during bulk delete",
            "detail": error_msg,
            "errors": [str(e)]
        }
    finally:
        conn.close() 