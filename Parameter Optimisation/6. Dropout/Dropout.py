#%% Dependencies

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score
from itertools import product

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

# Load optimized parameters
with open("Parameter Optimisation/2. Learning Rate/best_refined_learning_rate.txt", "r") as f:
    best_refined_learning_rate = float(f.read().strip())

with open("Parameter Optimisation/3. Hidden Layers and Units/best_hidden_layers_units.txt", "r") as f:
    best_hidden_layers = eval(f.read().strip())  # Convert stored string back to tuple

with open("Parameter Optimisation/4. Batch Size/best_batch_size.txt", "r") as f:
    best_batch_size = int(float(f.read().strip()))

with open("Parameter Optimisation/5. Epochs and Early Stopping/best_epochs.txt", "r") as f:
    best_epochs = int(float(f.read().strip()))

with open("Parameter Optimisation/5. Epochs and Early Stopping/best_patience.txt", "r") as f:
    best_patience = int(float(f.read().strip()))

# Store optimized parameters
OPTIMISED_PARAMS = {
    "learning_rate": best_refined_learning_rate,
    "hidden_layers": best_hidden_layers,
    "batch_size": best_batch_size,
    "epochs": best_epochs,
    "patience": best_patience
}

# Define dropout rates to test
dropout_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
dropout_rates = list(product(dropout_options, repeat=len(best_hidden_layers))) # generate combinations
num_trials = len(dropout_rates)
print(f"Total Dropout Configurations to Test: {num_trials}")

#%% Define Model

# Function to create model with variable dropout per layer
def create_model(dropout_rate_per_layer):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    for units, dropout_rate in zip(OPTIMISED_PARAMS["hidden_layers"], dropout_rate_per_layer):
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))  # Apply layer-specific dropout
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=OPTIMISED_PARAMS["learning_rate"])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

#%% Search

# List to store results
results = []

# Loop through dropout configurations
for i, dropout_rate_per_layer in enumerate(dropout_rates, start=1):
    print(f"Trial {i}/{num_trials}: Testing dropout rates per layer={dropout_rate_per_layer}")
    
    model = create_model(dropout_rate_per_layer)
    early_stopping = EarlyStopping(monitor='val_loss', patience=OPTIMISED_PARAMS["patience"], restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=OPTIMISED_PARAMS["epochs"], batch_size=OPTIMISED_PARAMS["batch_size"],
                        verbose=1, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    # Store loss values
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    # Predict on validation set
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # Compute Metrics
    f1 = f1_score(y_val, y_pred)
    auc_roc = roc_auc_score(y_val, y_pred_prob)
    pr_auc = average_precision_score(y_val, y_pred_prob)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    
    # Store results
    results.append([dropout_rate_per_layer, f1, auc_roc, pr_auc, accuracy, precision, recall, final_train_loss, final_val_loss])
    print(f"Dropout {dropout_rate_per_layer}, F1-score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}, PR-AUC: {pr_auc:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n")

# Convert results to DataFrame and save
results_df = pd.DataFrame(results, columns=['Dropout Rates', 'F1-Score', 'AUC-ROC', 'PR-AUC', 'Accuracy', 'Precision', 'Recall', 'Final Train Loss', 'Final Val Loss'])
results_df.to_csv("Parameter Optimisation/6. Dropout/dropout_results.csv", index=False)

# Identify the best dropout configuration
best_idx = results_df['F1-Score'].idxmax()
best_config = results_df.loc[best_idx]

# Save the best dropout rate configuration
with open("Parameter Optimisation/6. Dropout/best_dropout.txt", "w") as f:
    f.write(str(best_config['Dropout Rates']))

print("\nBest Dropout Configuration and Associated Metrics:")
print(best_config.to_string(index=True))

#%% Plotting

# Plot Training and Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(range(num_trials), results_df['Final Train Loss'], marker='o', label='Training Loss')
plt.plot(range(num_trials), results_df['Final Val Loss'], marker='s', label='Validation Loss')
plt.xlabel("Dropout Configuration Index")
plt.ylabel("Loss")
plt.title("Training and Validation Loss vs Dropout Configurations")
plt.legend()
plt.grid(True)
plt.savefig("Parameter Optimisation/6. Dropout/loss_vs_dropout.png")
plt.show()

print("Dropout optimization completed. Results saved.")
