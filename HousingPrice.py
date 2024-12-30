import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset from the CSV file
data = pd.read_csv('CSV/HousingData.csv')

# Features and target variable
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# Save the model to a .pkl file
with open('models/HousingModel.pkl', 'wb') as file:
    pickle.dump(model, file)