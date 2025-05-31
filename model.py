import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from datetime import datetime
import json

# Trying XGBoost, LightGBM, CatBoost
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

def load_and_preprocess_data(file_path='london_merged.csv'):
    """
    Load and preprocess the London bikes dataset with enhanced features
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if 7 <= x <= 9 or 16 <= x <= 19 else 0)
    df['is_morning'] = df['hour'].apply(lambda x: 1 if 6 <= x <= 11 else 0)
    df['is_evening'] = df['hour'].apply(lambda x: 1 if 17 <= x <= 22 else 0)
    df['temp_hum'] = df['t1'] * df['hum']
    df['temp_wind'] = df['t1'] * df['wind_speed']
    
    # Categorical Features
    weather_dummies = pd.get_dummies(df['weather_code'], prefix='weather')
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    
    features = pd.concat([
        df[['t1', 't2', 'hum', 'wind_speed', 'is_holiday', 'is_weekend',
              'hour', 'day_of_week', 'month', 'day_of_year', 'is_rush_hour', 'is_morning', 'is_evening',
              'temp_hum', 'temp_wind']],
        weather_dummies,
        season_dummies
    ], axis=1)
    target = df['cnt']
    return features, target

def evaluate_model(model, X, y, feature_names, scaler=None):
    """
    Evaluate model performance and return metrics
    """
    # Predicting
    y_pred = model.predict(X)
    
    # Metrics Calculation
    metrics = {
        'mse': mean_squared_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'r2': r2_score(y, y_pred)
    }
    
    # Feature Importance
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    elif hasattr(model, 'coef_'):
        feature_importance = dict(zip(feature_names, np.abs(model.coef_)))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    return metrics, feature_importance

def cross_val_rmse(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    return np.mean(np.sqrt(-scores))

def train_model():
    """
    Train the model and save necessary artifacts
    """
    X, y = load_and_preprocess_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    print(f"Cross-validated RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
    
    metrics, feature_importance = evaluate_model(model, X_test_scaled, y_test, X.columns)
    
    joblib.dump(model, 'bike_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(X.columns.tolist(), 'feature_names.joblib')
    
    with open('model_metrics.json', 'w') as f:
        json.dump({
            'metrics': metrics,
            'feature_importance': feature_importance
        }, f, indent=4)
    
    print("Model training completed and artifacts saved!")
    print("\nModel Performance Metrics:")
    print(f"Mean Squared Error: {metrics['mse']:.2f}")
    print(f"Root Mean Squared Error: {metrics['rmse']:.2f}")
    print(f"Mean Absolute Error: {metrics['mae']:.2f}")
    print(f"R² Score: {metrics['r2']:.4f}")
    
    return model, scaler, X.columns.tolist(), metrics, feature_importance

if __name__ == "__main__":
    train_model() 