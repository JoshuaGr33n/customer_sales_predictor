import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Database connection using SQLAlchemy
db_config = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'database': 'ml_predictive_task'
}

engine = create_engine(f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")
query = "SELECT * FROM sales_data"
df = pd.read_sql(query, engine)

print(df.head(2))

# Data preprocessing
df['date'] = pd.to_datetime(df['date'])
df['day_of_year'] = df['date'].dt.dayofyear
X = df[['day_of_year', 'marketing_spend']]
y = df['sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} Mean Squared Error: {mse}")

# Save the best model (optional)
best_model = min(models.items(), key=lambda x: mean_squared_error(y_test, x[1].predict(X_test)))[1]

print(f"Best Model: {best_model}")
joblib.dump(best_model, 'best_sales_predictor.pkl')