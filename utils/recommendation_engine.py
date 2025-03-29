import os
import groq
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directly set the API key (not recommended for production)
api_key = "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"

# Initialize Groq client
try:
    client = groq.Client(api_key=api_key)
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    client = None

def generate_advanced_recommendations(risk_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Generate advanced recommendations using Groq API based on risk assessment data
    
    Parameters:
    risk_data (dict): Dictionary containing risk assessment results
    
    Returns:
    dict: Dictionary with categorized recommendations
    """
    # If client initialization failed, return fallback recommendations
    if client is None:
        print("Using fallback recommendations (Groq client not available)")
        return get_fallback_recommendations()
        
    # Extract key risk information
    risk_score = risk_data.get('risk_score', 0)
    risk_category = risk_data.get('risk_category', 'Medium')
    location_data = risk_data.get('location_data', {})
    
    # Create prompt for Groq
    prompt = f"""
    Generate detailed construction risk mitigation recommendations for a site with the following characteristics:
    
    - Risk Score: {risk_score} (Category: {risk_category})
    - Elevation: {location_data.get('Elevation (m)', 'N/A')} meters
    - Temperature: {location_data.get('Avg Temperature (°C)', 'N/A')}°C
    - Annual Rainfall: {location_data.get('Annual Rainfall (mm)', 'N/A')} mm
    - Humidity: {location_data.get('Humidity (%)', 'N/A')}%
    - Flood Risk: {location_data.get('Flood Risk', 'N/A')}
    - Landslide Risk: {location_data.get('Landslide Risk', 'N/A')}
    - Earthquake Risk: {location_data.get('Earthquake Risk', 'N/A')}
    - Lightning Risk: {location_data.get('Lightning Risk', 'N/A')}
    
    Provide specific, actionable recommendations in these categories:
    1. Structural Design
    2. Material Selection
    3. Construction Methods
    4. Safety Protocols
    5. Project Planning
    
    Format each category with 2-3 specific recommendations.
    """
    
    # Call Groq API
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # Using Llama 3 70B model
            messages=[
                {"role": "system", "content": "You are a construction risk management expert specializing in providing detailed, technical recommendations for construction projects based on risk assessments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        
        # Process the response
        recommendation_text = response.choices[0].message.content
        
        # Parse the recommendations into categories
        categories = ["Structural Design", "Material Selection", "Construction Methods", 
                     "Safety Protocols", "Project Planning"]
        
        recommendations = {}
        current_category = None
        
        for line in recommendation_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is a category header
            for category in categories:
                if category in line or line.startswith(f"{categories.index(category)+1}."):
                    current_category = category
                    recommendations[current_category] = []
                    break
                    
            # If we have a current category and this isn't a header, it's a recommendation
            if current_category and not any(category in line for category in categories) and not line.startswith(f"{categories.index(current_category)+1}."):
                # Remove bullet points or numbering
                clean_line = line
                if line.startswith('- '):
                    clean_line = line[2:]
                elif line.startswith('* '):
                    clean_line = line[2:]
                elif len(line) > 2 and line[0].isdigit() and line[1] == '.':
                    clean_line = line[2:].strip()
                    
                if clean_line and current_category in recommendations:
                    recommendations[current_category].append(clean_line)
        
        return recommendations
        
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        # Fallback to basic recommendations
        return get_fallback_recommendations()

def get_fallback_recommendations():
    """Provide fallback recommendations when Groq API is unavailable"""
    return {
        "Structural Design": [
            "Implement robust foundation design appropriate for the risk level",
            "Consider reinforced structural elements for high-risk areas"
        ],
        "Material Selection": [
            "Use weather-resistant materials appropriate for the climate conditions",
            "Select materials with appropriate thermal properties"
        ],
        "Construction Methods": [
            "Implement proper drainage systems",
            "Use appropriate construction techniques for the terrain"
        ],
        "Safety Protocols": [
            "Develop emergency response plans for identified risks",
            "Implement regular safety inspections"
        ],
        "Project Planning": [
            "Schedule construction phases accounting for seasonal weather patterns",
            "Build in contingency time for potential delays due to identified risks"
        ]
    } 