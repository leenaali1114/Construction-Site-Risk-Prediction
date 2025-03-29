import pandas as pd
import numpy as np
import pickle

def load_model_and_dependencies():
    """Load the trained model and its dependencies"""
    with open('models/risk_assessment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('models/feature_list.pkl', 'rb') as f:
        features = pickle.load(f)
    
    with open('models/risk_categorizer.pkl', 'rb') as f:
        risk_categorizer = pickle.load(f)
    
    return model, scaler, features, risk_categorizer

def prepare_input_data(form_data):
    """
    Process form data and prepare it for prediction
    
    Parameters:
    form_data (dict): Form data from the web application
    
    Returns:
    dict: Processed data ready for prediction
    """
    # Extract and convert form data
    location_data = {
        'Elevation (m)': float(form_data.get('elevation', 0)),
        'Avg Temperature (°C)': float(form_data.get('temperature', 0)),
        'Annual Rainfall (mm)': float(form_data.get('rainfall', 0)),
        'Humidity (%)': float(form_data.get('humidity', 0)),
        'Flood Risk': int(form_data.get('flood_risk', 0)),
        'Landslide Risk': int(form_data.get('landslide_risk', 0)),
        'Earthquake Risk': int(form_data.get('earthquake_risk', 0)),
        'Lightning Risk': int(form_data.get('lightning_risk', 0))
    }
    
    return location_data

def calculate_derived_features(location_data, avg_temp):
    """
    Calculate derived features for prediction
    
    Parameters:
    location_data (dict): Basic location data
    avg_temp (float): Average temperature from the dataset
    
    Returns:
    dict: Location data with derived features
    """
    # Create a copy of the input data
    data = location_data.copy()
    
    # Calculate Environmental Risk
    data['Environmental Risk'] = data['Flood Risk'] + data['Landslide Risk'] + data['Earthquake Risk']
    
    # Calculate Temperature Risk
    data['Temperature Risk'] = abs(data['Avg Temperature (°C)'] - avg_temp)
    
    # Calculate Elevation Risk
    if data['Elevation (m)'] < 100:
        data['Elevation Risk'] = 0
    elif data['Elevation (m)'] < 300:
        data['Elevation Risk'] = 1
    elif data['Elevation (m)'] < 600:
        data['Elevation Risk'] = 2
    else:
        data['Elevation Risk'] = 3
    
    # Calculate Rainfall Risk
    if data['Annual Rainfall (mm)'] < 2500:
        data['Rainfall Risk'] = 0
    elif data['Annual Rainfall (mm)'] < 3000:
        data['Rainfall Risk'] = 1
    elif data['Annual Rainfall (mm)'] < 3500:
        data['Rainfall Risk'] = 2
    else:
        data['Rainfall Risk'] = 3
    
    # Calculate Humidity Risk
    if data['Humidity (%)'] < 70:
        data['Humidity Risk'] = 0
    elif data['Humidity (%)'] < 80:
        data['Humidity Risk'] = 1
    elif data['Humidity (%)'] < 90:
        data['Humidity Risk'] = 2
    else:
        data['Humidity Risk'] = 3
    
    # Calculate Total Natural Risk
    data['Total Natural Risk'] = (data['Flood Risk'] + data['Landslide Risk'] + 
                                data['Earthquake Risk'] + data['Lightning Risk'] + 
                                data['Elevation Risk'] + data['Rainfall Risk'] + 
                                data['Humidity Risk'])
    
    return data

def get_risk_factors_explanation(location_data):
    """
    Generate explanations for risk factors
    
    Parameters:
    location_data (dict): Location data with all features
    
    Returns:
    dict: Explanations for each risk factor
    """
    explanations = {}
    
    # Elevation risk explanation
    if location_data['Elevation Risk'] == 0:
        explanations['elevation'] = "Low elevation area with minimal risk."
    elif location_data['Elevation Risk'] == 1:
        explanations['elevation'] = "Moderate elevation with some construction challenges."
    elif location_data['Elevation Risk'] == 2:
        explanations['elevation'] = "High elevation area requiring special construction techniques."
    else:
        explanations['elevation'] = "Very high elevation with significant construction challenges and risks."
    
    # Temperature risk explanation
    if location_data['Temperature Risk'] < 2:
        explanations['temperature'] = "Temperature is close to regional average, posing minimal risk."
    elif location_data['Temperature Risk'] < 4:
        explanations['temperature'] = "Temperature variation may affect construction materials and schedules."
    else:
        explanations['temperature'] = "Significant temperature deviation from average, requiring special considerations."
    
    # Rainfall risk explanation
    if location_data['Rainfall Risk'] == 0:
        explanations['rainfall'] = "Low annual rainfall, minimal water-related risks."
    elif location_data['Rainfall Risk'] == 1:
        explanations['rainfall'] = "Moderate rainfall may cause occasional delays and requires proper drainage."
    elif location_data['Rainfall Risk'] == 2:
        explanations['rainfall'] = "High rainfall area with potential for water damage and construction delays."
    else:
        explanations['rainfall'] = "Very high rainfall requiring comprehensive water management systems."
    
    # Humidity risk explanation
    if location_data['Humidity Risk'] == 0:
        explanations['humidity'] = "Low humidity with minimal impact on construction."
    elif location_data['Humidity Risk'] == 1:
        explanations['humidity'] = "Moderate humidity may affect certain materials and finishing work."
    elif location_data['Humidity Risk'] == 2:
        explanations['humidity'] = "High humidity requiring moisture-resistant materials and techniques."
    else:
        explanations['humidity'] = "Very high humidity with significant impact on construction materials and methods."
    
    # Natural disaster risks
    disaster_risks = []
    if location_data['Flood Risk'] > 0:
        disaster_risks.append(f"Flood risk level {location_data['Flood Risk']}")
    if location_data['Landslide Risk'] > 0:
        disaster_risks.append(f"Landslide risk level {location_data['Landslide Risk']}")
    if location_data['Earthquake Risk'] > 0:
        disaster_risks.append(f"Earthquake risk level {location_data['Earthquake Risk']}")
    if location_data['Lightning Risk'] > 0:
        disaster_risks.append(f"Lightning risk level {location_data['Lightning Risk']}")
    
    if disaster_risks:
        explanations['natural_disasters'] = "Natural disaster risks include: " + ", ".join(disaster_risks)
    else:
        explanations['natural_disasters'] = "No significant natural disaster risks identified."
    
    return explanations

def get_mitigation_strategies(risk_category, risk_factors):
    """
    Generate mitigation strategies based on risk category and factors
    
    Parameters:
    risk_category (str): Risk category (Low, Medium, High, Very High)
    risk_factors (dict): Risk factors data
    
    Returns:
    list: Mitigation strategies
    """
    strategies = []
    
    # General strategies based on overall risk
    if risk_category == 'Low':
        strategies.append("Standard construction practices should be sufficient.")
        strategies.append("Regular monitoring and quality control procedures.")
    elif risk_category == 'Medium':
        strategies.append("Enhanced supervision and quality control measures.")
        strategies.append("Develop contingency plans for potential delays.")
        strategies.append("Consider weather-resistant construction materials.")
    elif risk_category == 'High':
        strategies.append("Comprehensive risk management plan required.")
        strategies.append("Increased budget contingency (15-20%) recommended.")
        strategies.append("Regular expert inspections during construction.")
        strategies.append("Advanced construction techniques may be necessary.")
    else:  # Very High
        strategies.append("Detailed geotechnical investigation essential before construction.")
        strategies.append("Specialized construction techniques required.")
        strategies.append("High budget contingency (20-30%) recommended.")
        strategies.append("Consider project redesign to mitigate extreme risks.")
        strategies.append("Expert consultation throughout project lifecycle.")
    
    # Specific strategies based on risk factors
    if risk_factors['Flood Risk'] > 0:
        strategies.append("Implement proper drainage systems and elevated foundations.")
        if risk_factors['Flood Risk'] > 1:
            strategies.append("Consider flood barriers and water-resistant materials.")
    
    if risk_factors['Landslide Risk'] > 0:
        strategies.append("Conduct slope stability analysis and implement retaining structures.")
        if risk_factors['Landslide Risk'] > 1:
            strategies.append("Consider soil reinforcement techniques and specialized foundations.")
    
    if risk_factors['Earthquake Risk'] > 0:
        strategies.append("Use earthquake-resistant design and construction methods.")
        strategies.append("Implement structural reinforcement and flexible connections.")
    
    if risk_factors['Lightning Risk'] > 1:
        strategies.append("Install comprehensive lightning protection systems.")
        strategies.append("Use surge protectors for electrical systems.")
    
    if risk_factors['Rainfall Risk'] > 1:
        strategies.append("Implement waterproofing measures and proper site drainage.")
        strategies.append("Schedule critical activities during drier periods.")
    
    if risk_factors['Humidity Risk'] > 1:
        strategies.append("Use moisture-resistant materials and anti-fungal treatments.")
        strategies.append("Implement proper ventilation systems.")
    
    if risk_factors['Elevation Risk'] > 1:
        strategies.append("Use specialized equipment for high-elevation construction.")
        strategies.append("Implement safety measures for working at heights.")
    
    return strategies

def load_dataset_stats():
    """Load the original dataset and calculate basic statistics"""
    df = pd.read_csv('data/malappuram_construction_risk.csv')
    stats = {
        'avg_temp': df['Avg Temperature (°C)'].mean(),
        'avg_risk_score': df['Risk Score'].mean(),
        'max_risk_score': df['Risk Score'].max(),
        'min_risk_score': df['Risk Score'].min(),
        'total_locations': len(df)
    }
    return stats, df 