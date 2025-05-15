import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os
import joblib

def train_and_log_model(tracking_uri="http://localhost:5000"):
    """
    Train a Random Forest model, log metrics, parameters, and model to MLflow
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create experiment
    mlflow_experiment_name = "sales_prediction_random_forest"
    try:
        experiment_id = mlflow.create_experiment(mlflow_experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(mlflow_experiment_name).experiment_id
    
    # Set the experiment
    mlflow.set_experiment(mlflow_experiment_name)
    
    print("Loading data...")
    
    # Code to load data
    try:
        # Try to load the actual data file
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Amazon Sale Report.csv')
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df_daily = df.groupby('Date')['Amount'].sum().reset_index()
    except:
        # If not found, create synthetic data
        print("CSV file not found. Creating synthetic data for demonstration...")
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
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Model parameters to try
    n_estimators_list = [50, 100]
    max_depth_list = [10, 20]
    min_samples_split_list = [2, 5]
    
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for min_samples_split in min_samples_split_list:
                # Start an MLflow run
                with mlflow.start_run():
                    # Log model parameters
                    mlflow.log_param("n_estimators", n_estimators)
                    mlflow.log_param("max_depth", max_depth)
                    mlflow.log_param("min_samples_split", min_samples_split)
                    
                    print(f"Training model with params: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")
                    
                    # Train model
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    # Calculate metrics
                    mae_train = mean_absolute_error(y_train, y_pred_train)
                    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    r2_train = r2_score(y_train, y_pred_train)
                    
                    mae_test = mean_absolute_error(y_test, y_pred_test)
                    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    r2_test = r2_score(y_test, y_pred_test)
                    
                    # Log metrics
                    mlflow.log_metric("mae_train", mae_train)
                    mlflow.log_metric("rmse_train", rmse_train)
                    mlflow.log_metric("r2_train", r2_train)
                    mlflow.log_metric("mae_test", mae_test)
                    mlflow.log_metric("rmse_test", rmse_test)
                    mlflow.log_metric("r2_test", r2_test)
                    
                    # Print metrics
                    print(f"Training MAE: {mae_train:.2f}, RMSE: {rmse_train:.2f}, R²: {r2_train:.4f}")
                    print(f"Test MAE: {mae_test:.2f}, RMSE: {rmse_test:.2f}, R²: {r2_test:.4f}")
                    
                    # Log feature importance
                    feature_importances = model.feature_importances_
                    for i, feature in enumerate(X.columns):
                        mlflow.log_metric(f"importance_{feature}", feature_importances[i])
                    
                    # Log model
                    mlflow.sklearn.log_model(model, "random_forest_model")
                    
                    # Save a copy of the model
                    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'random_forest_model.pkl')
                    joblib.dump(model, model_path)
                    print(f"Model saved to {model_path}")
    
    print("MLflow tracking completed. View results at {tracking_uri}")
    return model

def load_best_model_from_mlflow(tracking_uri="http://localhost:5000", metric="rmse_test"):
    """
    Load the best model from MLflow based on a specified metric
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    # Get the experiment
    experiment = mlflow.get_experiment_by_name("sales_prediction_random_forest")
    if not experiment:
        print("Experiment not found.")
        return None
    
    # Get all runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        print("No runs found.")
        return None
    
    # Find the best run based on the metric
    if metric.startswith("mae") or metric.startswith("rmse"):
        # Lower is better for MAE and RMSE
        best_run_id = runs.loc[runs[f"metrics.{metric}"].idxmin()]["run_id"]
    else:
        # Higher is better for R²
        best_run_id = runs.loc[runs[f"metrics.{metric}"].idxmax()]["run_id"]
    
    print(f"Loading model from run: {best_run_id}")
    
    # Load the model
    model_uri = f"runs:/{best_run_id}/random_forest_model"
    model = mlflow.sklearn.load_model(model_uri)
    
    return model

if __name__ == "__main__":
    # Train and log model
    train_and_log_model()
