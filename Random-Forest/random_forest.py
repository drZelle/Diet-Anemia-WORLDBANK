import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score

df = pd.read_csv(full_file_path)
# Print the DataFrame
print(df.columns)

# Features
X = df[['cereals', 'roots_tubers', 'vegetables', 'fruits', 'milk_equivalents',
'red_meat', 'poultry', 'eggs', 'seafood', 'legumes', 'nuts', 'oils',
'sugar']] 
# Target
y = df['Prevalence of anemia among pregnant women (%)'] 

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Instantiate the Random Forest model
rf_model = RandomForestRegressor(random_state=42,
                                 verbose=0)

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Print the best parameters found
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate the best model
best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# Evaluate performance
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
cv_strategy = grid_search.cv
n_trees = best_rf_model.n_estimators
