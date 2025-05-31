// Load performance metrics when the page loads
document.addEventListener('DOMContentLoaded', function() {
    loadPerformanceMetrics();
});

// Function to load performance metrics
async function loadPerformanceMetrics() {
    try {
        const response = await fetch('/performance');
        const data = await response.json();
        
        if (data.metrics) {
            document.getElementById('r2Score').textContent = data.metrics.r2.toFixed(4);
            document.getElementById('rmseScore').textContent = data.metrics.rmse.toFixed(2);
        }
    } catch (error) {
        console.error('Error loading performance metrics:', error);
    }
}

// Handle manual form submission
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    await makePrediction(getFormData());
});

// Handle CSV form submission
document.getElementById('csvForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const file = document.getElementById('csvFile').files[0];
    if (file) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/predict/csv', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            displayResults(data);
        } catch (error) {
            displayError(error.message);
        }
    }
});

// Handle API form submission
document.getElementById('apiForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const endpoint = document.getElementById('apiEndpoint').value;
    const apiKey = document.getElementById('apiKey').value;
    
    try {
        const response = await fetch('/predict/api', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ endpoint, apiKey })
        });
        const data = await response.json();
        displayResults(data);
    } catch (error) {
        displayError(error.message);
    }
});

// Function to get form data
function getFormData() {
    return {
        timestamp: document.getElementById('timestamp').value,
        t1: parseFloat(document.getElementById('t1').value),
        t2: parseFloat(document.getElementById('t2').value),
        hum: parseFloat(document.getElementById('hum').value),
        wind_speed: parseFloat(document.getElementById('wind_speed').value),
        weather_code: parseInt(document.getElementById('weather_code').value),
        is_holiday: parseInt(document.getElementById('is_holiday').value),
        is_weekend: parseInt(document.getElementById('is_weekend').value),
        season: parseInt(document.getElementById('season').value)
    };
}

// Function to make prediction
async function makePrediction(data) {
    showLoading();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            displayResults(result);
        } else {
            displayError(result.error || 'Prediction failed');
        }
    } catch (error) {
        displayError(error.message);
    }
}

// Function to display results
function displayResults(data) {
    const resultDiv = document.getElementById('result');
    
    if (data.status === 'success') {
        resultDiv.innerHTML = `
            <div class="text-center">
                <div class="prediction-value">${data.predicted_bike_count}</div>
                <div class="prediction-label">Predicted Bike Count</div>
            </div>
        `;
    } else {
        displayError(data.error || 'Prediction failed');
    }
}

// Function to display error
function displayError(message) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `
        <div class="text-center">
            <div class="error-message">
                <i class="fas fa-exclamation-circle me-2"></i>
                ${message}
            </div>
        </div>
    `;
}

// Function to show loading state
function showLoading() {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-3 text-muted">Calculating prediction...</p>
        </div>
    `;
}

// Set default timestamp to current time
document.getElementById('timestamp').value = new Date().toISOString().slice(0, 16); 