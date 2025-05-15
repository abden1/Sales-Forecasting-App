from flask import Flask, render_template, request, jsonify, redirect
import numpy as np
import pandas as pd
import pickle
import os
import joblib
import mlflow
import mlflow.sklearn
import uuid
import time

from datetime import datetime, timedelta

# Set up MLflow tracking
MLFLOW_TRACKING_URI = "http://localhost:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Create or get the experiment
MLFLOW_EXPERIMENT_NAME = "sales_prediction_web_app"
try:
    mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
except:
    pass  # Experiment already exists

# Set the active experiment
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

app = Flask(__name__)

# Load the trained model
# Load the model - try MLflow first, fall back to local file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'random_forest_model.pkl')
try:
    # Try to load the best model from MLflow
    mlflow.set_tracking_uri("http://localhost:5001")
    from mlflow_integration import load_best_model_from_mlflow
    model = load_best_model_from_mlflow(tracking_uri="http://localhost:5001", metric="r2_test")
    print("Loaded best model from MLflow")
except Exception as e:
    print(f"Error loading model from MLflow: {e}")
    print("Falling back to local model file")
    # Check if model exists, otherwise train it
    if not os.path.exists(MODEL_PATH):
        from model import train_and_save_model
        train_and_save_model(MODEL_PATH)
    model = joblib.load(MODEL_PATH)


@app.route('/')
def home():
    """Render the home page with default prediction values"""
    # Use default of 7 days for forecasting
    days = 7
    
    # Generate a forecast using default values
    forecast_data = generate_forecast(days=days)
    
    # Get historical data with matching days
    historical_data = get_historical_data(days=days)
    
    return render_template('index.html', 
                           forecast_data=forecast_data, 
                           historical_data=historical_data)

@app.route('/predict', methods=['POST'])
def predict():
    """Generate prediction based on user inputs"""
    try:
        # Get the form data
        lag_1 = float(request.form.get('lag_1', 500000))
        lag_7 = float(request.form.get('lag_7', 500000))
        rolling_3_mean = float(request.form.get('rolling_3_mean', 500000))
        day_of_week = int(request.form.get('day_of_week', 0))
        
        # Generate a unique run ID for this prediction
        run_id = str(uuid.uuid4())
        
        # Start an MLflow run to log this prediction
        with mlflow.start_run(run_name=f"single_prediction_{run_id}"):
            # Log input parameters
            mlflow.log_param("lag_1", lag_1)
            mlflow.log_param("lag_7", lag_7)
            mlflow.log_param("rolling_3_mean", rolling_3_mean)
            mlflow.log_param("day_of_week", day_of_week)
            mlflow.log_param("prediction_type", "single")
            
            # Make prediction for a single day
            prediction = make_single_prediction(lag_1, lag_7, rolling_3_mean, day_of_week)
            
            # Log metrics
            mlflow.log_metric("prediction_value", prediction)
            mlflow.log_metric("timestamp", time.time())
            
            # Set tag for source
            mlflow.set_tag("source", "web_app")
            mlflow.set_tag("user_agent", request.headers.get('User-Agent', 'Unknown'))
            
        return jsonify({
            'success': True,
            'prediction': f"{prediction:.2f}",
            'mlflow_run_id': run_id
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/forecast', methods=['POST'])
def forecast():
    """Generate a forecast based on user inputs with variable duration"""
    try:
        # Get the form data
        lag_1 = float(request.form.get('lag_1', 500000))
        lag_7 = float(request.form.get('lag_7', 500000))
        rolling_3_mean = float(request.form.get('rolling_3_mean', 500000))
        day_of_week = int(request.form.get('day_of_week', 0))
        forecast_days = int(request.form.get('forecast_days', 7))  # Get forecast duration
        
        # Generate a unique run ID for this forecast
        run_id = str(uuid.uuid4())
        
        # Start an MLflow run to log this forecast
        with mlflow.start_run(run_name=f"forecast_{forecast_days}days_{run_id}"):
            # Log input parameters
            mlflow.log_param("lag_1", lag_1)
            mlflow.log_param("lag_7", lag_7)
            mlflow.log_param("rolling_3_mean", rolling_3_mean)
            mlflow.log_param("day_of_week", day_of_week)
            mlflow.log_param("forecast_days", forecast_days)
            mlflow.log_param("prediction_type", "forecast")
            
            # Generate forecast with specified duration
            forecast_data = generate_forecast(lag_1, lag_7, rolling_3_mean, day_of_week, days=forecast_days)
            
            # Get historical data with the same day pattern and matching the forecast days
            historical_data = get_historical_data(start_day=day_of_week, days=forecast_days)
            
            # Log metrics
            forecast_values = [item['raw_prediction'] for item in forecast_data]
            for i, value in enumerate(forecast_values):
                mlflow.log_metric(f"forecast_day_{i+1}", value)
            
            mlflow.log_metric("forecast_mean", np.mean(forecast_values))
            mlflow.log_metric("forecast_min", np.min(forecast_values))
            mlflow.log_metric("forecast_max", np.max(forecast_values))
            mlflow.log_metric("forecast_total", np.sum(forecast_values))
            mlflow.log_metric("timestamp", time.time())
            
            # Set tags for source and metadata
            mlflow.set_tag("source", "web_app")
            mlflow.set_tag("user_agent", request.headers.get('User-Agent', 'Unknown'))
            
            # Create and log forecast chart as artifact
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, forecast_days+1), forecast_values, 'g-', label='Forecast')
                if historical_data and 'training' in historical_data and len(historical_data['training']) > 0:
                    plt.plot(range(1, len(historical_data['training'])+1), historical_data['training'], 'b--', label='Training')
                if historical_data and 'test' in historical_data and len(historical_data['test']) > 0:
                    plt.plot(range(1, len(historical_data['test'])+1), historical_data['test'], 'r-.', label='Test')
                plt.title(f"{forecast_days}-Day Sales Forecast")
                plt.xlabel("Day")
                plt.ylabel("Sales (₹)")
                plt.legend()
                plt.grid(True)
                
                # Save the chart as an artifact
                chart_path = "forecast_chart.png"
                plt.savefig(chart_path)
                plt.close()
                
                # Log the chart as an artifact
                mlflow.log_artifact(chart_path)
                os.remove(chart_path)  # Clean up
            except Exception as chart_error:
                print(f"Error creating chart for MLflow: {chart_error}")
        
        return jsonify({
            'success': True,
            'forecast': forecast_data,
            'historical_data': historical_data,
            'mlflow_run_id': run_id
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def make_single_prediction(lag_1, lag_7, rolling_3_mean, day_of_week):
    """Make a prediction for a single day"""
    # Create input features
    X_pred = pd.DataFrame({
        'lag_1': [lag_1],
        'lag_7': [lag_7],
        'rolling_3_mean': [rolling_3_mean],
        'day_of_week': [day_of_week]
    })
    
    # Make prediction
    prediction = model.predict(X_pred)[0]
    
    return prediction

def generate_forecast(lag_1=500000, lag_7=500000, rolling_3_mean=500000, day_of_week=0, days=7):
    """Generate a forecast for the next 'days' number of days"""
    forecasts = []
    today = datetime.now()
    dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
    
    # Initial values
    current = {
        'lag_1': lag_1,
        'lag_7': lag_7,
        'rolling_3_mean': rolling_3_mean,
        'day_of_week': day_of_week
    }
    
    for i in range(days):
        # Prepare input features
        X_pred = pd.DataFrame({
            'lag_1': [current['lag_1']],
            'lag_7': [current['lag_7']],
            'rolling_3_mean': [current['rolling_3_mean']],
            'day_of_week': [current['day_of_week']]
        })
        
        # Make prediction
        pred = model.predict(X_pred)[0]
        
        # Store result
        forecasts.append({
            'day': i + 1,
            'date': dates[i],
            'prediction': f"₹{pred:.2f}",
            'raw_prediction': pred
        })
        
        # Update for next prediction
        current['lag_7'] = current['lag_1'] if i % 7 == 6 else current['lag_7']
        current['lag_1'] = pred
        current['day_of_week'] = (current['day_of_week'] + 1) % 7
        current['rolling_3_mean'] = (current['lag_1'] + rolling_3_mean) / 2  # Simple approximation
    
    return forecasts

def get_historical_data(start_day=0, days=7):
    """Get historical data for charts, split into training and test sets"""
    try:
        # Try to load the actual data
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Amazon Sale Report.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df_daily = df.groupby('Date')['Amount'].sum().reset_index()
        else:
            # If file doesn't exist, create synthetic data that follows realistic patterns
            dates = pd.date_range(start='2022-01-01', periods=180)
            np.random.seed(42)
            
            # Create sales with weekly pattern
            base = 500000
            weekly_pattern = np.array([1.0, 0.95, 1.05, 1.15, 1.25, 1.35, 0.85])  # Mon-Sun
            noise = np.random.normal(0, 25000, len(dates))
            trend = np.linspace(0, 50000, len(dates))
            
            sales = [base * weekly_pattern[i % 7] + noise[i] + trend[i] for i in range(len(dates))]
            
            df_daily = pd.DataFrame({
                'Date': dates,
                'Amount': sales
            })
        
        # Create features for historical data
        df_daily = df_daily.sort_values('Date').reset_index(drop=True)
        df_daily['day_of_week'] = df_daily['Date'].dt.dayofweek
        
        # Find a day that starts with the requested start_day
        start_indices = df_daily[df_daily['day_of_week'] == start_day].index
        if len(start_indices) > 5:  # Ensure enough data for a proper sample
            start_idx = start_indices[5]  # Take the 5th occurrence to have enough lag data
            # Make sure we have enough data for requested days * 2 (for both train and test)
            if start_idx + (days*2) <= len(df_daily):  
                train_data = df_daily.loc[start_idx:start_idx+days-1, 'Amount'].tolist()
                test_data = df_daily.loc[start_idx+days:start_idx+(days*2)-1, 'Amount'].tolist()
                
                return {
                    'training': train_data,
                    'test': test_data
                }
        
        # Fallback to creating data with the right pattern but realistic values
        base_value = df_daily['Amount'].mean() if not df_daily.empty else 500000
        day_patterns = {
            0: 1.0,   # Monday: baseline
            1: 0.95,  # Tuesday: slight dip
            2: 1.05,  # Wednesday: slight increase
            3: 1.15,  # Thursday: moderate increase
            4: 1.25,  # Friday: bigger increase
            5: 1.35,  # Saturday: peak
            6: 0.85   # Sunday: big dip
        }
        
        training_data = []
        test_data = []
        for i in range(days):
            day_idx = (start_day + i) % 7
            pattern = day_patterns[day_idx % 7]  # Make sure we loop through the patterns
            # Add some randomness to make it realistic
            training_data.append(round(base_value * pattern * (1 + np.random.normal(0, 0.05))))
            test_data.append(round(base_value * pattern * (1 + np.random.normal(0, 0.03))))
            
        return {
            'training': training_data,
            'test': test_data
        }
            
    except Exception as e:
        print(f"Error getting historical data: {e}")
        # Fallback to synthetic data
        return {
            'training': [520000, 495000, 528000, 610000, 650000, 710000, 420000],
            'test': [515000, 490000, 525000, 605000, 640000, 700000, 410000]
        }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
