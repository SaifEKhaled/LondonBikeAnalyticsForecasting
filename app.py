from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import requests
from io import StringIO
import os

app = Flask(__name__)

# Load the best model and scaler
model = joblib.load('bike_model.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')

def prepare_input(data):
    # Convert input data to DataFrame
    df = pd.DataFrame([data])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if 7 <= x <= 9 or 16 <= x <= 19 else 0)
    df['is_morning'] = df['hour'].apply(lambda x: 1 if 6 <= x <= 11 else 0)
    df['is_evening'] = df['hour'].apply(lambda x: 1 if 17 <= x <= 22 else 0)
    df['temp_hum'] = df['t1'] * df['hum']
    df['temp_wind'] = df['t1'] * df['wind_speed']
    
    # Create dummy variables for categorical features
    weather_dummies = pd.get_dummies(df['weather_code'], prefix='weather')
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    
    # Combine all features
    features = pd.concat([
        df[['t1', 't2', 'hum', 'wind_speed', 'is_holiday', 'is_weekend',
              'hour', 'day_of_week', 'month', 'day_of_year', 'is_rush_hour', 'is_morning', 'is_evening',
              'temp_hum', 'temp_wind']],
        weather_dummies,
        season_dummies
    ], axis=1)
    
    # Ensure all feature columns exist
    for col in feature_names:
        if col not in features.columns:
            features[col] = 0
    
    # Reorder columns to match training data
    features = features[feature_names]
    
    return features

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Prepare input data
        features = prepare_input(data)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        return jsonify({
            'predicted_bike_count': int(prediction),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/predict/csv', methods=['POST'])
def predict_csv():
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'status': 'error'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'status': 'error'
            }), 400
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Prepare input data
        features = prepare_input(df.iloc[0].to_dict())
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        return jsonify({
            'predicted_bike_count': int(prediction),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/predict/api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        endpoint = data.get('endpoint')
        api_key = data.get('apiKey')
        
        if not endpoint or not api_key:
            return jsonify({
                'error': 'Missing endpoint or API key',
                'status': 'error'
            }), 400
        
        # Make API request
        response = requests.get(endpoint, headers={'Authorization': f'Bearer {api_key}'})
        response.raise_for_status()
        
        # Parse API response
        weather_data = response.json()
        
        # Prepare input data
        features = prepare_input(weather_data)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        return jsonify({
            'predicted_bike_count': int(prediction),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/performance')
def performance():
    try:
        # Load model metrics
        with open('model_metrics.json', 'r') as f:
            metrics_data = json.load(f)
        
        if request.headers.get('Accept') == 'application/json':
            return jsonify(metrics_data)
        
        return render_template('performance.html', 
                             metrics=metrics_data['metrics'],
                             feature_importance=metrics_data['feature_importance'])
    except Exception as e:
        if request.headers.get('Accept') == 'application/json':
            return jsonify({'error': str(e)}), 500
        return render_template('performance.html', error=str(e))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 