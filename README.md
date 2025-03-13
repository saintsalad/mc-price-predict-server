# Motorcycle Price Predictor (MPP)

A machine learning model API for predicting motorcycle prices in the Philippine market.

## Model Versioning System

The MPP uses a semantic versioning system with versioned model artifacts. Models are named in the format `MPP_vX.Y.Z` where:

- X = Major version (significant algorithm changes)
- Y = Minor version (feature additions or improvements)
- Z = Patch version (bug fixes or minor adjustments)

### Managing Model Versions

You can manage model versions using the `version_manager.py` script:

```bash
# Show current version
python version_manager.py show

# Increment patch version (for small fixes)
python version_manager.py increment patch

# Increment minor version (for feature enhancements)
python version_manager.py increment minor

# Increment major version (for significant changes)
python version_manager.py increment major

# Set to a specific version
python version_manager.py set --major 2 --minor 1 --patch 0
```

The versioning system automatically updates when training a new model with `train_model.py`.

## API Endpoints

### GET /model

Returns information about the latest model.

Example response:

```json
{
  "status": "success",
  "model": {
    "name": "MPP_v1.0.0",
    "version": "v1.0.0",
    "training_date": "2025-03-09T20:02:31.449692",
    "training_file": "training_data/training_records_20250309_200219.csv"
  },
  "performance": {
    "r2_score": 0.9834521715424374,
    "mae": 3496.6307391775126,
    "rmse": 4232.950643992336
  },
  "specs": {
    "features_count": 11,
    "encoders_count": 5,
    "top_features": [
      "priceRangeMax",
      "priceRangeMin",
      "displacement",
      "mileage",
      "model"
    ]
  },
  "status_details": {
    "model_loaded": true
  }
}
```

### POST /predict

Predicts the price of a motorcycle based on its specifications.

## Working with Model Versions

Models are stored in the following directories:

- `models/MPP_vX.Y.Z/` - Version-specific model artifacts
- `models/latest/` - Always points to the latest model for API use

Each time you train a new model, a new directory is created with the incremented version number.

## When to Update Version Numbers

- **Patch version (Z)**: Small fixes, minor adjustments to parameters, update training data
- **Minor version (Y)**: Add new features, improve preprocessing, enhance model quality
- **Major version (X)**: Change algorithm type, major architecture changes, significant preprocessing changes

## Training Utilities

### API Endpoints for Training

The application provides an API endpoint for training models from the web interface:

- `POST /train` - Start a training process in the background and returns the current training status

Example request:

```bash
curl -X POST http://localhost:8000/train
```

Example response:

```json
{
  "status": "success",
  "message": "Training process has been started",
  "training_status": {
    "is_training": true,
    "last_training": null,
    "status": "running",
    "message": "Exporting data from SQLite to CSV..."
  }
}
```

The training process runs in the background and updates its status internally. When the process completes, the training_status will reflect the result of the training, including the new model version created.

### Command Line Training Utility

## Running the Server

```bash
python main.py
```

The server will load the latest model and make it available via the API.
