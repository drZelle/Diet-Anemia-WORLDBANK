# Elastic Net
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(full_file_path)
print(df.columns)

# Features
X = df[[ 'cereals', 'roots_tubers', 'vegetables', 'fruits', 'milk_equivalents',
'red_meat', 'poultry', 'eggs', 'seafood', 'legumes', 'nuts', 'oils',
'sugar']]  # Example features (can be any number of features)
y = df['Prevalence of anemia among pregnant women (%)']  # Target variable (sickness)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with a scaler and ElasticNet model
pipe = Pipeline([
    ('scaler', StandardScaler()),  # Scaling step
    ('elastic_net', ElasticNet())  # ElasticNet model without predefined alpha and l1_ratio
])

# Define parameter grid for grid search
param_grid = {
    'elastic_net__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],  # Regularization strength
    'elastic_net__l1_ratio': [0.01, 0.1, 0.7, 1.0]   # Mix between Lasso and Ridge
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Access the best model and hyperparameters
best_model = grid_search.best_estimator_
best_alpha = grid_search.best_params_['elastic_net__alpha']
best_l1_ratio = grid_search.best_params_['elastic_net__l1_ratio']
scaler = pipe.named_steps['scaler']
cv_strategy = grid_search.cv
print(f"Best alpha: {best_alpha}")
print(f"Best l1_ratio: {best_l1_ratio}")

# Fit the model
pipe.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipe.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared (R²)
r2 = r2_score(y_test, y_pred)

# Output the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
