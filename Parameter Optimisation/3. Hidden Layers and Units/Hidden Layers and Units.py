#%% Dependencies

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from itertools import product
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score

#%% Initialisation

# Set Seed
SEED = 2025
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load preprocessed datasets
X_train = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/X_train.csv")
y_train = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/y_train.csv").values.ravel()
X_val = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/X_val.csv")
y_val = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/y_val.csv").values.ravel()

#%% Define Parameters

# Read best learning rate
with open("Parameter Optimisation/2. Learning Rate/best_refined_learning_rate.txt", "r") as f:
    best_refined_learning_rate = float(f.read().strip())

# Define Baseline Parameters
BASELINE_PARAMS = {
    "batch_size": 32,
    "epochs": 20,
    "threshold": 0.5
}

# Store Optimised Parameters
OPTIMISED_PARAMS = {
    "learning_rate": best_refined_learning_rate
}

# Define the range of hidden layers and units per layer
hidden_layers_range = list(range(1, 4))  # 1 to 3 hidden layers
units_per_layer = [32, 64, 128, 256]

# Generate all possible configurations
configurations = []
for layers in hidden_layers_range:
    for units in product(units_per_layer, repeat=layers):
        configurations.append(units)

num_trials = len(configurations)
print(f"Total Configurations to Test: {num_trials}")

#%% Define Model

# Function to create model with variable hidden layers and units
def create_model(hidden_units):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    for units in hidden_units:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=OPTIMISED_PARAMS["learning_rate"])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

#%% Search

# List to store results
results = []

# Loop through configurations
for i, hidden_units in enumerate(configurations, start=1):
    print(f"Trial {i}/{num_trials}: Testing hidden layers {hidden_units}")
    
    model = create_model(hidden_units)
    history = model.fit(X_train, y_train, epochs=BASELINE_PARAMS["epochs"], batch_size=BASELINE_PARAMS["batch_size"], verbose=1, validation_data=(X_val, y_val))
    
    # Store loss values
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    # Predict on validation set
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob >= BASELINE_PARAMS["threshold"]).astype(int)
    
    # Compute Metrics
    f1 = f1_score(y_val, y_pred)
    auc_roc = roc_auc_score(y_val, y_pred_prob)
    pr_auc = average_precision_score(y_val, y_pred_prob)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    
    # Store results
    results.append([hidden_units, f1, auc_roc, pr_auc, accuracy, precision, recall, final_train_loss, final_val_loss])
    print(f"Hidden Layers {hidden_units}, F1-score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}, PR-AUC: {pr_auc:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n")

# Convert results to DataFrame and save
results_df = pd.DataFrame(results, columns=['Hidden Layers', 'F1-Score', 'AUC-ROC', 'PR-AUC', 'Accuracy', 'Precision', 'Recall', 'Final Train Loss', 'Final Val Loss'])
results_df.to_csv("Parameter Optimisation/3. Hidden Layers and Units/hidden_layers_units_results.csv", index=False)

# Identify the best hidden layer configuration
best_idx = results_df['F1-Score'].idxmax()
best_config = results_df.loc[best_idx]

# Save the best configuration
with open("Parameter Optimisation/3. Hidden Layers and Units/best_hidden_layers_units.txt", "w") as f:
    f.write(str(best_config['Hidden Layers']))

print("\nBest Hidden Layer Configuration and Associated Metrics:")
print(best_config.to_string(index=True))

#%% Plotting for optimal trial - 67

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(range(num_trials), results_df['Final Train Loss'], marker='o', label='Training Loss')
plt.plot(range(num_trials), results_df['Final Val Loss'], marker='s', label='Validation Loss')
plt.xlabel("Trial Number")
plt.ylabel("Loss")
plt.title("Training and Validation Loss across Hidden Layer Configurations")
plt.legend()
plt.grid(True)
plt.savefig("Parameter Optimisation/3. Hidden Layers and Units/loss_vs_hidden_layers.png")
plt.show()

print("Done")
