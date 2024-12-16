import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
import warnings
import time

warnings.filterwarnings("ignore")

df = pd.read_csv(full_file_path)
print(df.columns)

# Features
X = df[[ 'cereals', 'roots_tubers', 'vegetables', 'fruits', 'milk_equivalents',
'red_meat', 'poultry', 'eggs', 'seafood', 'legumes', 'nuts', 'oils',
'sugar']] 
# Label
y = df['Prevalence of anemia among pregnant women (%)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

iter = 3000

# Create and train the MLP Regressor model
# Create MLP model
mlp_model = MLPRegressor(hidden_layer_sizes=(64,32), 
                         activation='logistic', 
                         max_iter=1, 
                         warm_start=True, 
                         random_state=42,
                         verbose=0  
                         )
# Store training and test losses
train_losses = []
test_losses = []
r2_scores = []

# Train the model with the warm_start option to allow partial training
for i in range(1, iter+1):
    mlp_model.fit(X_train, y_train)  # Train for 1 iteration
    train_pred = mlp_model.predict(X_train)
    test_pred = mlp_model.predict(X_test)
    
    # Calculate training and test loss (Mean Squared Error)
    train_loss = mean_squared_error(y_train, train_pred)
    test_loss = mean_squared_error(y_test, test_pred)
    
    # Calculate R² for the test set
    r2 = r2_score(y_test, test_pred)
    
    # Append the losses to the list
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    r2_scores.append(r2)
    
    # Print every 1/10 of the total iterations (i.e., every 100 iterations)
    if i % (iter // 10) == 0:
        print(f"Iteration {i}/{iter} - Training Loss: {train_loss:.4f}, \
              Test Loss: {test_loss:.4f}, R²: {r2:.4f}")


# Print final evaluation metrics
final_train_loss = train_losses[-1]
final_test_loss = test_losses[-1]
final_r2 = r2_scores[-1]

print(f"Final Training Loss (MSE): {final_train_loss:.4f}")
print(f"Final Test Loss (MSE): {final_test_loss:.4f}")
print(f"Final R²: {final_r2:.4f}")
