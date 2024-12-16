import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

df = pd.read_csv(full_file_path)
# Print the DataFrame
print(df.columns)

# Features selected
X = df[[ 'cereals', 'roots_tubers', 'vegetables', 'fruits', 'milk_equivalents',
'red_meat', 'poultry', 'eggs', 'seafood', 'legumes', 'nuts', 'oils',
'sugar']]
# tARGET
y = df['Prevalence of anemia among pregnant women (%)']

# Create a pipeline with StandardScaler, PolynomialFeatures, 
# and Ridge regression 
pipeline = Pipeline([ ('scaler', StandardScaler()), 
                     ('poly', PolynomialFeatures()), 
                     ('ridge', Ridge()) ])

# Define the parameter grid 
param_grid = { 
    'poly__degree': [1, 2, 3, 4], 
    'ridge__alpha': [0.01, 0.1, 1, 10, 100] }

# Perform GridSearchCV 
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error') 
grid_search.fit(X, y)

# Access the best parameters
best_params = grid_search.best_params_

# Extract the best degree and alpha
best_degree = best_params['poly__degree']
best_alpha = best_params['ridge__alpha']

# Best parameters and best score 
print("Best parameters:", best_params) 
print("Best score:", -grid_search.best_score_)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features before polynomial transformation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create polynomial features
poly = PolynomialFeatures(degree=best_degree)
X_poly_train = poly.fit_transform(X_train_scaled)  # Transform the training data
X_poly_test = poly.transform(X_test_scaled)  # Transform the test data

# Use Ridge Regression
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_poly_train, y_train)

# Make predictions on the test data
y_pred = ridge_model.predict(X_poly_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the performance metrics
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
