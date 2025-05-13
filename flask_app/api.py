from flask import Blueprint, request, jsonify
import pandas as pd
import joblib
import os

api_blueprint = Blueprint('api', __name__)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'random_forest_model.pkl')
model = joblib.load(MODEL_PATH)

@api_blueprint.route('/api/v1/predict', methods=['POST'])
def predict_api():
    """API endpoint for single prediction"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['lag_1', 'lag_7', 'rolling_3_mean', 'day_of_week']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Create input features
        X_pred = pd.DataFrame({
            'lag_1': [float(data['lag_1'])],
            'lag_7': [float(data['lag_7'])],
            'rolling_3_mean': [float(data['rolling_3_mean'])],
            'day_of_week': [int(data['day_of_week'])]
        })
        
        # Make prediction
        prediction = model.predict(X_pred)[0]
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'prediction_formatted': f"₹{prediction:.2f}"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_blueprint.route('/api/v1/forecast', methods=['POST'])
def forecast_api():
    """API endpoint for 7-day forecast"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['lag_1', 'lag_7', 'rolling_3_mean', 'day_of_week']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        days = int(data.get('days', 7))
        if days < 1 or days > 30:
            return jsonify({
                'success': False,
                'error': 'Days parameter must be between 1 and 30'
            }), 400
        
        # Initial values
        current = {
            'lag_1': float(data['lag_1']),
            'lag_7': float(data['lag_7']),
            'rolling_3_mean': float(data['rolling_3_mean']),
            'day_of_week': int(data['day_of_week'])
        }
        
        # Generate forecast
        forecasts = []
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
                'day_of_week': current['day_of_week'],
                'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][current['day_of_week']],
                'prediction': float(pred),
                'prediction_formatted': f"₹{pred:.2f}"
            })
            
            # Update for next prediction
            current['lag_7'] = current['lag_1'] if i % 7 == 6 else current['lag_7']
            current['lag_1'] = pred
            current['day_of_week'] = (current['day_of_week'] + 1) % 7
            current['rolling_3_mean'] = (current['lag_1'] + current['rolling_3_mean']) / 2  # Simple approximation
        
        return jsonify({
            'success': True,
            'forecast': forecasts
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
