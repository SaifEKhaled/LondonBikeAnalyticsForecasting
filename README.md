# London Bike Demand Prediction

This project uses machine learning to predict bike demand in London based on various weather and time-based features. The model is deployed as a Flask API.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python model.py
```

3. Run the Flask application:
```bash
python app.py
```

## API Usage

The API will be available at `http://localhost:5000`

### Making Predictions

Send a POST request to `/predict` with the following JSON format:

```json
{
    "time": "2024-03-20 14:00:00",
    "temp_real_c": 15.0,
    "temp_feels_like_c": 14.0,
    "humidity_percent": 0.7,
    "wind_speed": 10.0,
    "weather": "Clear",
    "is_holiday": 0,
    "is_weekend": 0,
    "season": "Spring"
}
```

### Weather Codes
- Clear
- Scattered clouds
- Broken clouds
- Cloudy
- Rain
- Rain with thunderstorm
- Snowfall

### Seasons
- Spring
- Summer
- Autumn
- Winter

## Model Details

The model uses a Random Forest Regressor with the following features:
- Time-based features (hour, day of week, month)
- Weather conditions
- Temperature (real and feels like)
- Humidity
- Wind speed
- Holiday and weekend indicators
- Season

## Project Structure

- `model.py`: Contains the machine learning model training code
- `app.py`: Flask application for serving predictions
- `requirements.txt`: Project dependencies
- `london_bikes_cleaned.xlsx`: Processed dataset
- `bike_model.joblib`: Trained model
- `scaler.joblib`: Feature scaler
- `feature_names.joblib`: Feature names for model input 