import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import sys

def train_and_save_model(model_path=None):
    """
    Train a RandomForest model on e-commerce sales data and save it to disk
    """
    try:
        print("Loading data...")
        # Get the path to the notebook directory
        notebook_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RANDOM_FOREST_ML.ipynb')
        
        # Check if the notebook exists
        if not os.path.exists(notebook_path):
            print("Error: Notebook not found. Make sure the notebook is in the correct location.")
            return False
        
        # Use the CSV file path from the notebook or load a default path
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Amazon Sale Report.csv')
        
        # Check if CSV exists, if not create synthetic data
        if not os.path.exists(csv_path):
            print("CSV file not found. Creating synthetic data for demonstration...")
            # Create synthetic data
            dates = pd.date_range(start='2022-01-01', periods=180)
            np.random.seed(42)
            
            # Create synthetic sales with weekly pattern
            base = 500000
            weekly_pattern = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 0.7])  # Mon-Sun
            noise = np.random.normal(0, 50000, len(dates))
            trend = np.linspace(0, 100000, len(dates))
            
            sales = [base * weekly_pattern[i % 7] + noise[i] + trend[i] for i in range(len(dates))]
            
            df_daily = pd.DataFrame({
                'Date': dates,
                'Amount': sales
            })
        else:
            # Load actual data
            print(f"Loading data from {csv_path}...")
            df = pd.read_csv(csv_path)
            
            # Preprocess data
            df['Date'] = pd.to_datetime(df['Date'])
            df_daily = df.groupby('Date')['Amount'].sum().reset_index()
        
        print("Preparing features...")
        # Create features
        df_daily = df_daily.set_index('Date')
        df_daily['day_of_week'] = df_daily.index.dayofweek
        df_daily['is_weekend'] = (df_daily['day_of_week'] >= 5).astype(int)
        
        # Create lag features
        df_daily['lag_1'] = df_daily['Amount'].shift(1)
        df_daily['lag_7'] = df_daily['Amount'].shift(7)  # Weekly seasonality
        df_daily['rolling_3_mean'] = df_daily['Amount'].rolling(window=3).mean()
        
        # Drop rows with missing values
        df_model = df_daily.dropna()
        
        # Feature Selection
        X = df_model[['lag_1', 'lag_7', 'rolling_3_mean', 'day_of_week']]
        y = df_model['Amount']
        
        print("Training model...")
        # Train model with reasonable defaults
        model = RandomForestRegressor(
            n_estimators=50,  # Fewer trees for faster inference
            max_depth=10,     # Limit depth to avoid overfitting
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        
        # Calculate metrics on training data
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        print(f"Model trained with MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        # Save model
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'random_forest_model.pkl')
        
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        return True
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return False

if __name__ == "__main__":
    train_and_save_model()
