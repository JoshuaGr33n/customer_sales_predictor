import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib

# Database connection using SQLAlchemy
db_config = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'database': 'ml_predictive_task'
}

engine = create_engine(
    f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")
query = "SELECT * FROM health_data"
df = pd.read_sql(query, engine)

# Data preprocessing
X = df[['age', 'bmi', 'blood_pressure']]
y = df['diabetes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Model training
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Check the size of the training and test sets
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

best_model = None
best_score = float('-inf')
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if name == 'Random Forest':  # Classification model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} - Accuracy: {accuracy}")
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_model_name = name
    else:  # Regression models
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} - Mean Squared Error: {mse}, R2 Score: {r2}")
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name

print(f"Best Model: {best_model_name} with score: {best_score}")

# Save the best model
joblib.dump(best_model, 'best_health_predictor.pkl')