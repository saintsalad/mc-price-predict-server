import sqlite3
import csv
import json
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def flatten_record(record):
    """Flatten the nested JSON structure into CSV-friendly format"""
    specifications = json.loads(record['specifications'])
    condition = json.loads(record['condition_data'])
    
    return {
        'id': record['id'],
        'brand': record['brand'],
        'model': record['model'],
        'category': specifications['category'],
        'displacement': specifications['displacement'],
        'transmission': specifications['transmission'],
        'yearRange': specifications['yearRange'],
        'priceRangeMin': specifications['priceRange']['min'],
        'priceRangeMax': specifications['priceRange']['max'],
        'year': condition['year'],
        'mileage': condition['mileage'],
        'sellerType': condition['sellerType'],
        'owner': condition['owner'],
        'knownIssues': condition['knownIssues'],
        'predictedPrice': record['predicted_price'],
        'created_at': record['created_at']
    }

def ensure_output_directory(directory='training_data'):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
    return directory

def get_output_filename(directory='training_data'):
    """Generate filename with timestamp in the specified directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'training_records_{timestamp}.csv'
    return os.path.join(directory, filename)

def sqlite_to_csv(database_path='motorcycle_training.db', output_file=None):
    # Ensure training_data directory exists
    data_dir = ensure_output_directory()
    
    if output_file is None:
        output_file = get_output_filename()
    else:
        # If custom filename provided, still put it in training_data directory
        output_file = os.path.join(data_dir, output_file)
        
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(database_path)
        conn.row_factory = sqlite3.Row
        
        # Get all records
        cursor = conn.execute('SELECT * FROM training_records')
        records = cursor.fetchall()
        
        if not records:
            logger.warning("No records found in the database")
            return
        
        # Define CSV headers
        headers = [
            'id', 'brand', 'model', 'category', 'displacement', 'transmission',
            'yearRange', 'priceRangeMin', 'priceRangeMax', 'year', 'mileage',
            'sellerType', 'owner', 'knownIssues', 'predictedPrice', 'created_at'
        ]
        
        # Write to CSV file
        with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()
            
            for record in records:
                try:
                    flattened_record = flatten_record(record)
                    writer.writerow(flattened_record)
                except Exception as e:
                    logger.error(f"Error processing record {record['id']}: {str(e)}")
                    continue
        
        logger.info(f"Successfully exported {len(records)} records to {output_file}")
        
    except Exception as e:
        logger.error(f"Error exporting database: {str(e)}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    sqlite_to_csv()

# run: py sqlite_to_csv.py