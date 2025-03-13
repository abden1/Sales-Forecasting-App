from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from app.models.forecasting import ARIMAForecaster
from app.database import Database
from app.data_loader import initialize_database
import os

app = Flask(__name__, 
    static_url_path='',
    static_folder='static',
    template_folder='templates')
CORS(app)

db = Database()
forecaster = ARIMAForecaster()
forecaster.preprocess_data()

def initialize_app():
    """Initialize the application with data and pre-trained model"""
    initialize_database()
    try:
        forecaster.train()
        print("Model pre-trained successfully!")
    except Exception as e:
        print(f"Error in initialization: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        if data and 'order' in data:
            p, d, q = data['order']
            forecaster.order = (p, d, q)
        
        result = forecaster.train()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()
        horizon = data.get('horizon', 30)
        forecast_data = forecaster.predict(horizon)
        return jsonify(forecast_data)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    initialize_app()
    app.run(host='localhost', port=8000) 