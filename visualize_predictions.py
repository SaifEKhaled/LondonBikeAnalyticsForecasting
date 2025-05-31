import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from model import load_and_preprocess_data
from sklearn.metrics import mean_squared_error

# Load the model and scaler
model = joblib.load('bike_model.joblib')
scaler = joblib.load('scaler.joblib')

# Load and preprocess data
X, y = load_and_preprocess_data()

# Load original data to get timestamps
df = pd.read_csv('london_merged.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Make predictions
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

# Function to remove outliers using IQR method
def remove_outliers(actual, predicted, timestamps, threshold=1.5):
    # Calculate IQR for actual values
    Q1 = np.percentile(actual, 25)
    Q3 = np.percentile(actual, 75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    # Create mask for non-outlier values
    mask = (actual >= lower_bound) & (actual <= upper_bound)
    
    return actual[mask], predicted[mask], timestamps[mask]

# Remove outliers
y_clean, y_pred_clean, timestamps_clean = remove_outliers(y, y_pred, df['timestamp'])

# Calculate MSE after outlier removal
mse = mean_squared_error(y_clean, y_pred_clean)
rmse = np.sqrt(mse)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Time series plot
ax1.plot(timestamps_clean, y_clean, label='Actual', alpha=0.7)
ax1.plot(timestamps_clean, y_pred_clean, label='Predicted', alpha=0.7)
ax1.set_xlabel('Time')
ax1.set_ylabel('Number of Bikes')
ax1.set_title('Actual vs Predicted Values Over Time (Outliers Removed)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Scatter plot
ax2.scatter(y_clean, y_pred_clean, alpha=0.5)
ax2.plot([y_clean.min(), y_clean.max()], [y_clean.min(), y_clean.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Values')
ax2.set_ylabel('Predicted Values')
ax2.set_title('Predicted vs Actual Values (Outliers Removed)')

# Add R² score to the scatter plot
r2 = np.corrcoef(y_clean, y_pred_clean)[0, 1] ** 2
ax2.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax2.transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))

# Add MSE and RMSE information
ax2.text(0.05, 0.85, f'MSE = {mse:.2f}\nRMSE = {rmse:.2f}', transform=ax2.transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

# Add information about removed outliers
outliers_removed = len(y) - len(y_clean)
ax2.text(0.05, 0.70, f'Outliers removed: {outliers_removed} points', transform=ax2.transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

# Adjust layout and save
plt.tight_layout()
plt.savefig('predicted_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close() 