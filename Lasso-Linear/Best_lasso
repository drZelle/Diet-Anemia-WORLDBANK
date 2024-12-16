from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

df = pd.read_csv(full_file_path)
print(df.columns)

# Features selected
X = df[['cereals', 'roots_tubers', 'vegetables', 'fruits', 'milk_equivalents',
'red_meat', 'poultry', 'eggs', 'seafood', 'legumes', 'nuts', 'oils',
'sugar']]
# Target
y = df['Prevalence of anemia among pregnant women (%)']  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for Lasso)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the range of alpha values to try
param_grid = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

# Perform grid search with cross-validation
grid_search = GridSearchCV(Lasso(), param_grid, cv=10)
grid_search.fit(X_train_scaled, y_train)
best_alpha = grid_search.best_params_['alpha']
cv_strategy = grid_search.cv
# Print the best alpha value
print("Best alpha (lambda):", best_alpha)

# Refit the Lasso model with the best alpha
best_lasso = grid_search.best_estimator_
y_pred = best_lasso.predict(X_test_scaled)

# Evaluate the best model
mse_best = mean_squared_error(y_test, y_pred)
r2_best = r2_score(y_test, y_pred)

print(f"Optimized Mean Squared Error: {mse_best:.2f}")
print(f"Optimized R-squared: {r2_best:.2f}")
