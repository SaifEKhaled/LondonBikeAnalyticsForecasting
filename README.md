# London Bike Demand Prediction and Analytics

## Overview
This project focuses on analyzing and predicting bike sharing demand in London using machine learning techniques. The system provides both analytical insights and predictive capabilities for bike sharing demand based on various environmental and temporal factors.

## Objectives
- Develop a machine learning model to predict bike sharing demand
- Analyze patterns and trends in bike usage
- Identify key factors influencing bike sharing demand
- Create an interactive web application for predictions and analytics
- Provide comprehensive visualization of historical data and predictions

## Features
- **Demand Prediction**: Real-time predictions of bike sharing demand
- **Interactive Dashboard**: Visual analytics of historical data and trends
- **Feature Analysis**: Understanding key factors affecting bike usage
- **Performance Metrics**: Comprehensive model evaluation and metrics
- **Time Series Analysis**: Pattern recognition across different time periods

## Technical Stack
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Gradient Boosting
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Chart.js
- **Frontend**: HTML, CSS, Bootstrap
- **Deployment**: Heroku

## Model Architecture
The prediction model uses a Gradient Boosting Regressor with the following specifications:
- Number of estimators: 200
- Learning rate: 0.1
- Maximum depth: 5
- Loss function: Squared Error (MSE)

### Key Features Used
1. **Temporal Features**:
   - Hour of day
   - Day of week
   - Month
   - Day of year
   - Rush hour indicators
   - Morning/Evening periods

2. **Weather Features**:
   - Temperature
   - Humidity
   - Wind speed
   - Weather conditions
   - Season

3. **Special Events**:
   - Holidays
   - Weekends
   - Rush hours

## Model Performance
The model demonstrates strong predictive capabilities:
- **R² Score**: 96.5% (explains 96.5% of the variance in bike demand)
- **RMSE**: 210 bikes (average prediction error)
- **MAE**: 121 bikes (average absolute error)

### Key Findings
1. **Temporal Patterns**:
   - Peak demand during rush hours (7-9 AM and 4-7 PM)
   - Higher usage on weekdays compared to weekends
   - Seasonal variations in demand

2. **Weather Impact**:
   - Temperature has a significant positive correlation with demand
   - Rain and high wind speeds reduce bike usage
   - Optimal conditions: mild temperature, low wind, no rain

3. **Feature Importance**:
   - Time of day is the most significant predictor
   - Weather conditions rank second in importance
   - Special events (holidays, weekends) show moderate influence

## Project Structure
```
├── app.py                 # Flask application
├── model.py              # Machine learning model implementation
├── visualize_predictions.py  # Visualization scripts
├── requirements.txt      # Project dependencies
├── static/              # Static files (CSS, JS)
├── templates/           # HTML templates
└── data/               # Data files
    ├── london_merged.csv
    └── sample_data.csv
```

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/SaifEKhaled/LondonBikeAnalyticsForecasting.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

## Usage
1. **Web Interface**:
   - Access the dashboard at `http://localhost:5000`
   - Input parameters for prediction
   - View historical data and trends

2. **API Endpoints**:
   - `/predict`: Get bike demand predictions
   - `/performance`: View model performance metrics
   - `/analytics`: Access historical data analysis

## Future Improvements
1. **Model Enhancements**:
   - Incorporate more historical data
   - Add real-time weather data integration
   - Implement ensemble methods

2. **Feature Additions**:
   - User authentication
   - Customizable dashboards
   - Export functionality
   - Mobile responsiveness

3. **Analytics Expansion**:
   - Advanced time series analysis
   - Geographic distribution analysis
   - User behavior patterns

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Contact
For any queries or suggestions, please reach out to [saifkhaledd1511@gmail.com]

## Acknowledgments
- London Bike Sharing Dataset
- Scikit-learn documentation
- Flask framework
- Bootstrap framework 
