import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import logging
import os
from datetime import datetime
import json
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_latest_training_file():
    """Get the most recent training data file from training_data directory."""
    training_dir = 'training_data'
    files = [f for f in os.listdir(training_dir) if f.startswith('training_records_') and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No training data files found in training_data directory")
    
    latest_file = max(files)
    return os.path.join(training_dir, latest_file)

def setup_model_directory():
    """Create a new timestamped directory for the current model version."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = 'models'
    version_dir = os.path.join(base_dir, f'version_{timestamp}')
    
    # Create directories
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(version_dir, exist_ok=True)
    
    # Create a latest symlink/directory
    latest_dir = os.path.join(base_dir, 'latest')
    if os.path.exists(latest_dir):
        if os.path.islink(latest_dir):
            os.unlink(latest_dir)
        else:
            shutil.rmtree(latest_dir)
    
    # Create new latest directory
    shutil.copytree(version_dir, latest_dir)
    
    return version_dir, latest_dir

def preprocess_data(df):
    """Preprocess the motorcycle training data."""
    df = df.copy(deep=True)
    
    # Extract numeric values from price fields
    df['priceRangeMin'] = pd.to_numeric(df['priceRangeMin'])
    df['priceRangeMax'] = pd.to_numeric(df['priceRangeMax'])
    df['predictedPrice'] = pd.to_numeric(df['predictedPrice'])
    
    # Convert specifications from string to structured data if needed
    df['displacement'] = pd.to_numeric(df['displacement'])
    df['mileage'] = pd.to_numeric(df['mileage'])
    df['year'] = pd.to_numeric(df['year'])
    
    # Calculate age
    current_year = datetime.now().year
    df['age'] = current_year - df['year']
    
    # Convert categorical variables
    categorical_columns = ['brand', 'model', 'category', 'transmission', 'sellerType', 'owner']
    label_encoders = {}
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column].astype(str))
    
    return df, label_encoders

def train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate the Random Forest model."""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, metrics, feature_importance

def save_model_artifacts(version_dir, latest_dir, model, label_encoders, features, metrics, feature_importance, training_file):
    """Save all model-related artifacts."""
    # Save model
    joblib.dump(model, os.path.join(version_dir, 'model.pkl'))
    joblib.dump(model, os.path.join(latest_dir, 'model.pkl'))
    
    # Save label encoders
    joblib.dump(label_encoders, os.path.join(version_dir, 'label_encoders.pkl'))
    joblib.dump(label_encoders, os.path.join(latest_dir, 'label_encoders.pkl'))
    
    # Save features list
    joblib.dump(features, os.path.join(version_dir, 'features.pkl'))
    joblib.dump(features, os.path.join(latest_dir, 'features.pkl'))
    
    # Save metrics and feature importance as JSON
    model_info = {
        'training_file': training_file,
        'training_date': datetime.now().isoformat(),
        'metrics': metrics,
        'feature_importance': feature_importance.to_dict(orient='records')
    }
    
    with open(os.path.join(version_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=4)
    with open(os.path.join(latest_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=4)

def main():
    try:
        # Get latest training file
        training_file = get_latest_training_file()
        logger.info(f"Using training data from: {training_file}")
        
        # Setup model directories
        version_dir, latest_dir = setup_model_directory()
        logger.info(f"Created model version directory: {version_dir}")
        
        # Load and preprocess data
        df = pd.read_csv(training_file)
        logger.info(f"Loaded {len(df)} records")
        
        df_processed, label_encoders = preprocess_data(df)
        logger.info("Data preprocessing completed")
        
        # Define features and target
        features = ['brand', 'model', 'category', 'displacement', 'transmission', 
                   'mileage', 'sellerType', 'owner', 'age', 'priceRangeMin', 'priceRangeMax']
        
        X = df_processed[features]
        y = df_processed['predictedPrice']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info("Data split completed")
        
        # Train model
        logger.info("Training model...")
        model, metrics, feature_importance = train_and_evaluate_model(
            X_train, X_test, y_train, y_test, features
        )
        
        # Log metrics
        logger.info("\nModel Performance Metrics:")
        logger.info(f"Mean Absolute Error: {metrics['mae']:,.2f} PHP")
        logger.info(f"Root Mean Squared Error: {metrics['rmse']:,.2f} PHP")
        logger.info(f"R² Score: {metrics['r2']:.4f}")
        
        # Log feature importance
        logger.info("\nFeature Importance:")
        for _, row in feature_importance.iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")
        
        # Save all model artifacts
        save_model_artifacts(
            version_dir,
            latest_dir,
            model,
            label_encoders,
            features,
            metrics,
            feature_importance,
            training_file
        )
        
        logger.info(f"\nModel and artifacts saved in:")
        logger.info(f"Version directory: {version_dir}")
        logger.info(f"Latest directory: {latest_dir}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()

#run: py train_model.py