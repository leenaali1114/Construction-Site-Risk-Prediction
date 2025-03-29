import pandas as pd
import numpy as np
import pickle

def predict_risk(location_data, model, scaler, features):
    """
    Predict risk score for a new location
    
    Parameters:
    location_data (dict): Dictionary containing location features
    model: Trained model
    scaler: Fitted scaler
    features (list): List of features used by the model
    
    Returns:
    float: Predicted risk score
    """
    # Create a dataframe with the location data
    location_df = pd.DataFrame([location_data])
    
    # Select only the features used by the model
    X = location_df[features]
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    risk_score = model.predict(X_scaled)[0]
    
    return risk_score

def get_similar_locations(new_location, dataset, n=5):
    """
    Find similar locations in the dataset
    
    Parameters:
    new_location (dict): New location data
    dataset (DataFrame): Original dataset
    n (int): Number of similar locations to return
    
    Returns:
    DataFrame: Similar locations
    """
    # Create a copy of the dataset
    df = dataset.copy()
    
    # Calculate similarity score (Euclidean distance)
    features = ['Elevation (m)', 'Avg Temperature (°C)', 'Annual Rainfall (mm)', 
                'Humidity (%)', 'Flood Risk', 'Landslide Risk', 'Earthquake Risk', 
                'Lightning Risk']
    
    # Normalize the features for distance calculation
    for feature in features:
        if feature in df.columns:
            max_val = df[feature].max()
            min_val = df[feature].min()
            if max_val > min_val:
                df[f'{feature}_norm'] = (df[feature] - min_val) / (max_val - min_val)
                new_location[f'{feature}_norm'] = (new_location[feature] - min_val) / (max_val - min_val)
            else:
                df[f'{feature}_norm'] = 0
                new_location[f'{feature}_norm'] = 0
    
    # Calculate Euclidean distance
    df['distance'] = 0
    for feature in features:
        if f'{feature}_norm' in df.columns:
            df['distance'] += (df[f'{feature}_norm'] - new_location[f'{feature}_norm']) ** 2
    
    df['distance'] = np.sqrt(df['distance'])
    
    # Sort by distance and return top n
    similar_locations = df.sort_values('distance').head(n)
    
    return similar_locations[['Location', 'Risk Score', 'Latitude', 'Longitude', 'distance']]

def get_risk_breakdown(location_data):
    """
    Break down the risk factors for visualization
    
    Parameters:
    location_data (dict): Location data with all features
    
    Returns:
    dict: Risk breakdown for visualization
    """
    risk_breakdown = {
        'Environmental': (location_data['Flood Risk'] + location_data['Landslide Risk'] + 
                         location_data['Earthquake Risk']) / 3 * 100,
        'Climate': (location_data['Rainfall Risk'] + location_data['Humidity Risk'] + 
                   location_data['Temperature Risk'] / 5) / 3 * 100,
        'Geographical': location_data['Elevation Risk'] / 3 * 100,
        'Lightning': location_data['Lightning Risk'] / 2 * 100
    }
    
    return risk_breakdown

def get_feature_importance(model, features):
    """
    Get feature importance from the model
    
    Parameters:
    model: Trained model
    features (list): List of features used by the model
    
    Returns:
    dict: Feature importance
    """
    importance = model.feature_importances_
    feature_importance = dict(zip(features, importance))
    
    # Sort by importance
    feature_importance = {k: v for k, v in sorted(feature_importance.items(), 
                                                 key=lambda item: item[1], 
                                                 reverse=True)}
    
    return feature_importance 