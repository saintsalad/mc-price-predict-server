import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_numeric(x):
    """Clean numeric strings by removing commas, units, and converting to float."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        # Remove any non-numeric characters except dots and commas
        # First, remove commas used as thousand separators
        x = x.replace(',', '')
        # Extract only numbers and decimal points
        numeric_str = re.search(r'[\d.]+', x)
        if numeric_str:
            return float(numeric_str.group())
    return np.nan

def preprocess_data(df):
    """Preprocess the motorcycle data."""
    # Create a deep copy to avoid warnings
    df = df.copy(deep=True)
    
    # Rename the price column to remove spaces
    df = df.rename(columns={'selling_price in Ph': 'selling_price'})
    
    # Clean numeric columns
    numeric_columns = ['selling_price', 'km_driven', 'ex_showroom_price']
    for col in numeric_columns:
        df[col] = df[col].apply(clean_numeric)
        # Log the number of NaN values after cleaning
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            logger.info(f"Column {col} has {nan_count} NaN values after cleaning")
    
    # Convert year to age
    current_year = pd.Timestamp.now().year
    df['age'] = current_year - df['year']
    
    # Handle missing values in ex_showroom_price
    median_price = df.groupby('name')['ex_showroom_price'].transform('median')
    df = df.assign(ex_showroom_price=df['ex_showroom_price'].fillna(median_price))
    
    # If still any NaN (for unique names), fill with overall median
    overall_median = df['ex_showroom_price'].median()
    df = df.assign(ex_showroom_price=df['ex_showroom_price'].fillna(overall_median))
    
    # Convert categorical variables
    categorical_columns = ['name', 'seller_type', 'owner']
    label_encoders = {}
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column].astype(str))
    
    # Save label encoders for prediction
    joblib.dump(label_encoders, './mnt/data/label_encoders.pkl')
    
    # Log data statistics
    logger.info("\nData Statistics after preprocessing:")
    for col in numeric_columns + ['age']:
        logger.info(f"{col}:")
        logger.info(f"  Min: {df[col].min():,.2f}")
        logger.info(f"  Max: {df[col].max():,.2f}")
        logger.info(f"  Mean: {df[col].mean():,.2f}")
        logger.info(f"  Median: {df[col].median():,.2f}")
    
    return df

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Train and evaluate the Random Forest model."""
    # Initialize model with optimized parameters
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics
    logger.info(f"Mean Absolute Error: {mae:,.2f} PHP")
    logger.info(f"Root Mean Squared Error: {rmse:,.2f} PHP")
    logger.info(f"R² Score: {r2:.4f}")
    
    # Log feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    logger.info("\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        logger.info(f"{row['feature']}: {row['importance']:.4f}")
    
    return model

def main():
    try:
        # Load the data
        logger.info("Loading data...")
        file_path = "./mnt/data/MOTORCYCLE DETAILS.csv"
        df = pd.read_csv(file_path)
        
        # Log initial data info
        logger.info(f"\nInitial dataset shape: {df.shape}")
        logger.info("Initial missing values:")
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                logger.info(f"  {col}: {missing} missing values")
        
        # Preprocess the data
        logger.info("\nPreprocessing data...")
        df_processed = preprocess_data(df)
        
        # Define features and target
        features = ['name', 'year', 'seller_type', 'owner', 'km_driven', 
                   'ex_showroom_price', 'age']
        X = df_processed[features]
        y = df_processed['selling_price']
        
        # Log data info
        logger.info(f"\nProcessed dataset shape: {df_processed.shape}")
        logger.info(f"Number of unique motorcycles: {df_processed['name'].nunique()}")
        
        # Split the data
        logger.info("\nSplitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train and evaluate the model
        logger.info("\nTraining model...")
        model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        
        # Save the model
        model_filename = "./mnt/data/motorcycle_price_model.pkl"
        joblib.dump(model, model_filename)
        logger.info(f"\nModel saved as {model_filename}")
        
        # Save feature names for prediction
        joblib.dump(features, './mnt/data/model_features.pkl')
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
