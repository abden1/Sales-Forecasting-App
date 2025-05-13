from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import os
import joblib
from datetime import datetime, timedelta

# Import API blueprint
from api import api_blueprint

app = Flask(__name__)

# Register the API blueprint
app.register_blueprint(api_blueprint)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'random_forest_model.pkl')

# Check if model exists, otherwise train it
if not os.path.exists(MODEL_PATH):
    from model import train_and_save_model
    train_and_save_model(MODEL_PATH)

model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    """Render the home page with default prediction values"""
    # Generate a 7-day forecast using default values
    forecast_data = generate_forecast()
    return render_template('index.html', forecast_data=forecast_data)

@app.route('/performance')
def performance():
    """Render the model performance dashboard"""
    # In a real application, these metrics would be calculated from the model
    metrics = {
        'mae': 76736.00,
        'rmse': 114492.83,
        'r2': 0.87,
        'n_estimators': 50,
        'train_size': '142 days',
        'test_size': '36 days'
    }
    return render_template('performance.html', metrics=metrics)

@app.route('/docs')
def api_docs():
    """Render the API documentation page"""
    return render_template('api_docs.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Generate prediction based on user inputs"""
    try:
        # Get the form data
        lag_1 = float(request.form.get('lag_1', 500000))
        lag_7 = float(request.form.get('lag_7', 500000))
        rolling_3_mean = float(request.form.get('rolling_3_mean', 500000))
        day_of_week = int(request.form.get('day_of_week', 0))
        
        # Make prediction for a single day
        prediction = make_single_prediction(lag_1, lag_7, rolling_3_mean, day_of_week)
        
        return jsonify({
            'success': True,
            'prediction': f"{prediction:.2f}"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/forecast', methods=['POST'])
def forecast():
    """Generate a 7-day forecast based on user inputs"""
    try:
        # Get the form data
        lag_1 = float(request.form.get('lag_1', 500000))
        lag_7 = float(request.form.get('lag_7', 500000))
        rolling_3_mean = float(request.form.get('rolling_3_mean', 500000))
        day_of_week = int(request.form.get('day_of_week', 0))
        
        # Generate forecast
        forecast_data = generate_forecast(lag_1, lag_7, rolling_3_mean, day_of_week)
        
        return jsonify({
            'success': True,
            'forecast': forecast_data
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
