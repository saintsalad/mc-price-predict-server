import os
import subprocess
import json
import datetime
from fastapi import APIRouter, BackgroundTasks
from typing import Dict, Any

# Create a router for the training endpoints
router = APIRouter()

# Global variable to track training status
training_status = {
    "is_training": False,
    "last_training": None,
    "status": "idle",
    "message": "No training has been initiated yet"
}

def run_training_process():
    """Run the training process by executing sqlite_to_csv.py and train_model.py"""
    global training_status
    
    try:
        # Update status
        training_status["is_training"] = True
        training_status["status"] = "running"
        training_status["message"] = "Exporting data from SQLite to CSV..."
        
        # Step 1: Execute sqlite_to_csv.py
        sqlite_result = subprocess.run(
            ["python", "sqlite_to_csv.py"], 
            capture_output=True,
            text=True,
            check=True
        )
        
        # Update status
        training_status["message"] = "Training model with exported data..."
        
        # Step 2: Execute train_model.py
        train_result = subprocess.run(
            ["python", "train_model.py"], 
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract version info if possible
        version_info = "Unknown"
        try:
            # Try to read the version from the version_info.json file
            with open(os.path.join('models', 'version_info.json'), 'r') as f:
                version_data = json.load(f)
                version_info = f"v{version_data['major']}.{version_data['minor']}.{version_data['patch']}"
        except:
            pass
        
        # Update status on completion
        training_status["is_training"] = False
        training_status["last_training"] = datetime.datetime.now().isoformat()
        training_status["status"] = "completed"
        training_status["message"] = f"Training completed successfully. New model version: MPP_{version_info}"
        
    except subprocess.CalledProcessError as e:
        # Handle errors
        error_msg = f"Error during {'data export' if 'sqlite_to_csv' in str(e.cmd) else 'model training'}"
        if e.stderr:
            error_msg += f": {e.stderr}"
            
        # Update status on error
        training_status["is_training"] = False
        training_status["status"] = "failed"
        training_status["message"] = error_msg
    except Exception as e:
        # Handle any other errors
        training_status["is_training"] = False
        training_status["status"] = "failed"
        training_status["message"] = f"Unexpected error: {str(e)}"

@router.post("/train")
async def train_model(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """API endpoint to trigger model training process"""
    global training_status
    
    # Check if training is already in progress
    if training_status["is_training"]:
        return {
            "status": "error",
            "message": "Training is already in progress",
            "current_status": training_status
        }
    
    # Start training in background
    background_tasks.add_task(run_training_process)
    
    return {
        "status": "success",
        "message": "Training process has been started",
        "training_status": training_status
    } 