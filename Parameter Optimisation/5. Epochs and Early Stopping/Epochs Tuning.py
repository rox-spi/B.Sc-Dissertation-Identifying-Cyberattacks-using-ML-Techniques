#%% Dependencies

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
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

# Load optimized parameters
with open("Parameter Optimisation/2. Learning Rate/best_refined_learning_rate.txt", "r") as f:
    best_refined_learning_rate = float(f.read().strip())

with open("Parameter Optimisation/3. Hidden Layers and Units/best_hidden_layers_units.txt", "r") as f:
    best_hidden_layers = eval(f.read().strip())  # Convert stored string back to tuple

with open("Parameter Optimisation/4. Batch Size/best_batch_size.txt", "r") as f:
    best_batch_size = int(float(f.read().strip()))

# Store optimized parameters
OPTIMISED_PARAMS = {
    "learning_rate": best_refined_learning_rate,
    "hidden_layers": best_hidden_layers,
    "batch_size": best_batch_size
}

# Define range of epochs to test
epochs_to_test = list(range(10, 110, 10))  # Steps of 10 from 10 to 100
num_trials = len(epochs_to_test)
print(f"Total Epoch Configurations to Test: {num_trials}")

#%% Define Model

# Function to create model with best settings
def create_model():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    for units in OPTIMISED_PARAMS["hidden_layers"]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=OPTIMISED_PARAMS["learning_rate"])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

#%% Search

# List to store results
results = []

# Loop through different epochs
for i, num_epochs in enumerate(epochs_to_test, start=1):
    print(f"Trial {i}/{num_trials}: Testing {num_epochs} epochs")
    
    model = create_model()
    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=OPTIMISED_PARAMS["batch_size"], verbose=1, validation_data=(X_val, y_val))
    
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
    results.append([num_epochs, f1, auc_roc, pr_auc, accuracy, precision, recall, final_train_loss, final_val_loss])
    print(f"Epochs {num_epochs}, F1-score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}, PR-AUC: {pr_auc:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n")

# Convert results to DataFrame and save
results_df = pd.DataFrame(results, columns=['Epochs', 'F1-Score', 'AUC-ROC', 'PR-AUC', 'Accuracy', 'Precision', 'Recall', 'Final Train Loss', 'Final Val Loss'])
results_df.to_csv("Parameter Optimisation/5. Epochs and Early Stopping/epochs_results.csv", index=False)

# Identify the best epoch configuration
best_idx = results_df['F1-Score'].idxmax()
best_config = results_df.loc[best_idx]

# Save the best number of epochs
with open("Parameter Optimisation/5. Epochs and Early Stopping/best_epochs.txt", "w") as f:
    f.write(str(best_config['Epochs']))

print("\nBest Number of Epochs and Associated Metrics:")
print(best_config.to_string(index=True))

#%% Plotting

# Plot Training and Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(epochs_to_test, results_df['Final Train Loss'], marker='o', label='Training Loss')
plt.plot(epochs_to_test, results_df['Final Val Loss'], marker='s', label='Validation Loss')
plt.xlabel("Number of Epochs")
plt.xticks(epochs_to_test)
plt.ylabel("Loss")
plt.title("Training vs Validation Loss\nNumber of Epochs")
plt.legend()
plt.ylim(0.032, 0.04)  
plt.grid(True)
plt.tight_layout()
plt.savefig("Parameter Optimisation/5. Epochs and Early Stopping/loss_vs_epochs.png", dpi=600)
plt.show()

print("Epoch optimization completed. Results saved.")
