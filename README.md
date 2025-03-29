# AI-Based Risk Assessment for Construction Projects

This project implements an AI-based risk assessment system for construction projects in the Malappuram region. The system analyzes various environmental, geographical, and climate factors to predict risk scores and provide mitigation strategies.

## Features

- **Risk Prediction**: Predict risk scores for construction sites based on multiple factors
- **Risk Visualization**: Visualize risk factors and their impact on construction projects
- **Mitigation Strategies**: Get tailored recommendations to mitigate identified risks
- **Similar Locations**: Find similar locations and compare risk profiles
- **Interactive Map**: Explore risk distribution across the region

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn, Chart.js
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Mapping**: Leaflet.js

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/construction-risk-assessment.git
   cd construction-risk-assessment
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:5000`

## Project Structure

- `app.py`: Main Flask application
- `models/`: Saved ML models
- `static/`: CSS, JS, and other static files
- `templates/`: HTML templates
- `data/`: Data files
- `notebooks/`: Jupyter notebooks for analysis
- `utils/`: Utility functions

## Usage

1. Navigate to the Risk Assessment page
2. Enter the details of your construction site
3. Submit the form to get a comprehensive risk assessment
4. View the risk score, breakdown, and mitigation strategies
5. Explore similar locations and their risk profiles

## Model Training

The risk assessment model is trained using a Random Forest Regressor on historical data from the Malappuram region. The model considers the following factors:

- Elevation
- Temperature
- Rainfall
- Humidity
- Flood Risk
- Landslide Risk
- Earthquake Risk
- Lightning Risk

Additional derived features are created to improve model performance.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Data provided by [source]
- Inspired by [reference] 