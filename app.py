from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import json
import pickle
from utils.data_processor import (load_model_and_dependencies, prepare_input_data, 
                                 calculate_derived_features, get_risk_factors_explanation,
                                 get_mitigation_strategies, load_dataset_stats)
from utils.model_utils import (predict_risk, get_similar_locations, 
                              get_risk_breakdown, get_feature_importance)
from dotenv import load_dotenv
from utils.recommendation_engine import generate_advanced_recommendations

# Load environment variables
load_dotenv()

# Define risk categorization function at the global scope
def categorize_risk(risk_score):
    if risk_score < 0.3:
        return "Low"
    elif risk_score < 0.7:
        return "Medium"
    elif risk_score < 1.0:
        return "High"
    else:
        return "Very High"

app = Flask(__name__)

# Create necessary directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('static/data', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# Copy the dataset to the data directory if it doesn't exist
if not os.path.exists('data/malappuram_construction_risk.csv'):
    import shutil
    shutil.copy('malappuram_construction_risk.csv', 'data/malappuram_construction_risk.csv')

# Check if model exists, if not, train it
if not os.path.exists('models/risk_assessment_model.pkl'):
    print("Training model...")
    try:
        # Import necessary libraries
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Load the dataset
        df = pd.read_csv('data/malappuram_construction_risk.csv')
        
        # Add risk category
        df['Risk Category'] = df['Risk Score'].apply(categorize_risk)
        
        # Calculate average temperature for reference
        avg_temp = df['Avg Temperature (°C)'].mean()
        
        # Feature Engineering
        # Environmental Risk
        df['Environmental Risk'] = (df['Flood Risk'] + df['Landslide Risk'] + df['Earthquake Risk']) / 3
        
        # Temperature Risk
        df['Temperature Risk'] = abs(df['Avg Temperature (°C)'] - avg_temp) / 10
        
        # Elevation Risk
        def elevation_risk(elevation):
            if elevation < 100:
                return 0
            elif elevation < 300:
                return 1
            elif elevation < 500:
                return 2
            else:
                return 3
        
        df['Elevation Risk'] = df['Elevation (m)'].apply(elevation_risk)
        
        # Rainfall Risk
        def rainfall_risk(rainfall):
            if rainfall < 2500:
                return 0
            elif rainfall < 3000:
                return 1
            elif rainfall < 3500:
                return 2
            else:
                return 3
        
        df['Rainfall Risk'] = df['Annual Rainfall (mm)'].apply(rainfall_risk)
        
        # Humidity Risk
        def humidity_risk(humidity):
            if humidity < 70:
                return 0
            elif humidity < 80:
                return 1
            elif humidity < 90:
                return 2
            else:
                return 3
        
        df['Humidity Risk'] = df['Humidity (%)'].apply(humidity_risk)
        
        # Total Natural Risk
        df['Total Natural Risk'] = (df['Flood Risk'] + df['Landslide Risk'] + 
                                   df['Earthquake Risk'] + df['Lightning Risk'] + 
                                   df['Elevation Risk'] + df['Rainfall Risk'] + 
                                   df['Humidity Risk'])
        
        # Prepare data for modeling
        features = ['Elevation (m)', 'Avg Temperature (°C)', 'Annual Rainfall (mm)', 'Humidity (%)',
                    'Flood Risk', 'Landslide Risk', 'Earthquake Risk', 'Lightning Risk',
                    'Environmental Risk', 'Temperature Risk', 'Elevation Risk', 'Rainfall Risk', 
                    'Humidity Risk', 'Total Natural Risk']
        
        X = df[features]
        y = df['Risk Score']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Save the model and dependencies
        with open('models/risk_assessment_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)
        
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        with open('models/feature_list.pkl', 'wb') as f:
            pickle.dump(features, f)
        
        with open('models/risk_categorizer.pkl', 'wb') as f:
            pickle.dump(categorize_risk, f)
        
        # Save map data
        map_data = df[['Location', 'Latitude', 'Longitude', 'Risk Score', 'Risk Category']].copy()
        map_data.to_csv('static/data/map_data.csv', index=False)
        
        print("Model training completed successfully!")
    except Exception as e:
        print(f"Error during model training: {e}")
        raise  # Re-raise the exception to stop execution if model training fails

# Load model and dependencies
try:
    model, scaler, features, risk_categorizer = load_model_and_dependencies()
    # Load dataset statistics
    stats, dataset = load_dataset_stats()
    # Get feature importance
    feature_importance = get_feature_importance(model, features)
    
    # Prepare map data
    map_data = dataset[['Location', 'Latitude', 'Longitude', 'Risk Score']].copy()
    
    # Add risk category
    map_data['Risk Category'] = map_data['Risk Score'].apply(risk_categorizer)
    
    # Save map data for the map view
    map_data.to_csv('static/data/map_data.csv', index=False)
    
    print("Model and dependencies loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Create dummy data for development
    model, scaler, features, risk_categorizer = None, None, [], lambda x: "Medium"
    stats = {'avg_temp': 28, 'avg_risk_score': 0.5, 'max_risk_score': 1.8, 'min_risk_score': 0, 'total_locations': 500}
    dataset = pd.DataFrame()
    feature_importance = {}
    map_data = pd.DataFrame()
    print("Using dummy data for development.")

@app.route('/')
def index():
    return render_template('index.html', stats=stats)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/assessment')
def assessment():
    return render_template('assessment.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form
    
    # Prepare input data
    location_data = prepare_input_data(form_data)
    
    # Calculate derived features
    location_data = calculate_derived_features(location_data, stats['avg_temp'])
    
    # Predict risk score
    risk_score = predict_risk(location_data, model, scaler, features)
    
    # Get risk category
    risk_category = risk_categorizer(risk_score)
    
    # Get risk factors explanation
    explanations = get_risk_factors_explanation(location_data)
    
    # Get basic mitigation strategies (we'll keep these as a fallback)
    basic_strategies = get_mitigation_strategies(risk_category, location_data)
    
    # Get similar locations
    similar_locations = get_similar_locations(location_data, dataset)
    
    # Get risk breakdown
    risk_breakdown = get_risk_breakdown(location_data)
    
    # Prepare data for advanced recommendations
    risk_data = {
        'risk_score': float(risk_score),
        'risk_category': risk_category,
        'location_data': location_data
    }
    
    # Get advanced recommendations using Groq
    try:
        advanced_recommendations = generate_advanced_recommendations(risk_data)
    except Exception as e:
        print(f"Error generating advanced recommendations: {e}")
        advanced_recommendations = None
    
    # Prepare result data
    result = {
        'location_name': form_data.get('location_name', 'New Location'),
        'latitude': form_data.get('latitude', ''),
        'longitude': form_data.get('longitude', ''),
        'risk_score': float(risk_score),
        'risk_category': risk_category,
        'explanations': explanations,
        'strategies': basic_strategies,
        'advanced_recommendations': advanced_recommendations,
        'similar_locations': similar_locations.to_dict('records'),
        'risk_breakdown': risk_breakdown,
        'feature_importance': {k: float(v) for k, v in list(feature_importance.items())[:5]}
    }
    
    return render_template('result.html', result=result)

@app.route('/map')
def map_view():
    # Load map data
    map_data = pd.read_csv('static/data/map_data.csv')
    
    # Convert to list of dictionaries for JSON
    locations = map_data.to_dict('records')
    
    return render_template('map.html', locations=json.dumps(locations))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Get JSON data
    data = request.get_json()
    
    # Prepare input data
    location_data = {
        'Elevation (m)': float(data.get('elevation', 0)),
        'Avg Temperature (°C)': float(data.get('temperature', 0)),
        'Annual Rainfall (mm)': float(data.get('rainfall', 0)),
        'Humidity (%)': float(data.get('humidity', 0)),
        'Flood Risk': int(data.get('flood_risk', 0)),
        'Landslide Risk': int(data.get('landslide_risk', 0)),
        'Earthquake Risk': int(data.get('earthquake_risk', 0)),
        'Lightning Risk': int(data.get('lightning_risk', 0))
    }
    
    # Calculate derived features
    location_data = calculate_derived_features(location_data, stats['avg_temp'])
    
    # Predict risk score
    risk_score = predict_risk(location_data, model, scaler, features)
    
    # Get risk category
    risk_category = risk_categorizer(risk_score)
    
    # Prepare result
    result = {
        'risk_score': float(risk_score),
        'risk_category': risk_category
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 