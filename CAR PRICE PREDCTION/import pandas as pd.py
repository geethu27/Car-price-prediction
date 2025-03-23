import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Sample dataset (Replace with actual dataset)
data = {
    'Age': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Mileage': [5000, 15000, 30000, 45000, 60000, 75000, 90000, 105000, 120000, 135000],
    'Price': [30000, 27000, 24000, 21000, 18000, 15000, 12000, 9000, 6000, 3000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Age', 'Mileage']]
y = df['Price']

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# Predict the price of a new car given Age and Mileage
age = 4  # Example input
mileage = 50000  # Example input
predicted_price = model.predict([[age, mileage]])
print(f"Predicted Price: {predicted_price[0]}")

# Plot the results
plt.scatter(df['Mileage'], df['Price'], color='blue', label='Actual Prices')
plt.scatter(X_test['Mileage'], y_pred, color='red', label='Predicted Prices')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.legend()
plt.title('Car Price Prediction')
plt.show()
