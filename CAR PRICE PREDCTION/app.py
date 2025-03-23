from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Initialize Flask app
app = Flask(__name__)

# Sample car dataset (you can replace this with your actual dataset)
data = {
    'Brand': ['Toyota', 'Ford', 'BMW', 'Audi', 'Ford'],
    'Model': ['Camry', 'Focus', 'X5', 'A4', 'Fusion'],
    'Year': [2015, 2017, 2016, 2018, 2019],
    'Mileage': [50000, 30000, 40000, 20000, 25000],
    'Price': [15000, 12000, 35000, 25000, 18000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Preprocessing the data
le_brand = LabelEncoder()
df['Brand'] = le_brand.fit_transform(df['Brand'])
le_model = LabelEncoder()
df['Model'] = le_model.fit_transform(df['Model'])

# Features and target variable
X = df[['Brand', 'Model', 'Year', 'Mileage']]
y = df['Price']

# Train a model
model = LinearRegression()
model.fit(X, y)

# Save the model and label encoders
joblib.dump(model, 'car_price_model.pkl')
joblib.dump(le_brand, 'brand_encoder.pkl')
joblib.dump(le_model, 'model_encoder.pkl')

# Load the model
model = joblib.load('car_price_model.pkl')
le_brand = joblib.load('brand_encoder.pkl')
le_model = joblib.load('model_encoder.pkl')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data from request
    brand = data['brand']
    model_name = data['model']
    year = data['year']
    mileage = data['mileage']

    try:
        # Preprocess the input data
        brand_encoded = le_brand.transform([brand])[0]
        model_encoded = le_model.transform([model_name])[0]

        # Predict the price
        input_data = np.array([[brand_encoded, model_encoded, year, mileage]])
        predicted_price = model.predict(input_data)

        # Return the prediction result as JSON
        return jsonify({'predicted_price': round(float(predicted_price[0]),2)})

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except KeyError as e:
        return jsonify({'error': f'Missing key: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {e}'}), 500

if __name__ == '__main__