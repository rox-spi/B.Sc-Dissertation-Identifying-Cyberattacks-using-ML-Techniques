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

# Define Baseline Parameters
BASELINE_PARAMS = {
    "batch_size": 32,
    "epochs": 20,
    "threshold": 0.5,
    "hidden_layers": 1,
    "hidden_units": 128
}

# Define a refined range of learning rates around the best found value 0.002
refined_learning_rates = [0.0015, 0.002, 0.0025]
print("Refined Learning Rates:", refined_learning_rates)
num_trials = len(refined_learning_rates)

#%% Define Model

# Define function to create the baseline model
def create_model(learning_rate):
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(BASELINE_PARAMS["hidden_units"], activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

#%% Search

# List to store results
results = []

for i, lr in enumerate(refined_learning_rates, start=1):
    print(f"Trial {i}/{num_trials}: Training with learning rate: {lr}")
    
    model = create_model(lr)
    history = model.fit(X_train, y_train, epochs=BASELINE_PARAMS["epochs"], batch_size=BASELINE_PARAMS["batch_size"], verbose=1, validation_data=(X_val, y_val))
    
    # Store loss values
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"Final Train Loss: {final_train_loss:.4f}, Final Validation Loss: {final_val_loss:.4f}\n")
    
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
    results.append([lr, f1, auc_roc, pr_auc, accuracy, precision, recall, final_train_loss, final_val_loss])
    print(f"Learning Rate: {lr}, F1-score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}, PR-AUC: {pr_auc:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n")

# Convert results to DataFrame and save
results_df = pd.DataFrame(results, columns=['Learning Rate', 'F1-Score', 'AUC-ROC', 'PR-AUC', 'Accuracy', 'Precision', 'Recall', 'Final Train Loss', 'Final Val Loss'])
results_df.to_csv("Parameter Optimisation/2. Learning Rate/refined_lr_results.csv", index=False)

# Identify the best refined learning rate (based on highest F1-score)
best_idx = results_df['F1-Score'].idxmax()
best_lr_data = results_df.loc[best_idx]

# Save the best refined learning rate in a text file
with open("Parameter Optimisation/2. Learning Rate/best_refined_learning_rate.txt", "w") as f:
    f.write(str(best_lr_data["Learning Rate"]))

print("\nBest Refined Learning Rate and Associated Metrics:")
print(best_lr_data.to_string(index=True))

#%% Plotting

# Plot Training and Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(refined_learning_rates, results_df['Final Train Loss'], marker='o', label='Training Loss')
plt.plot(refined_learning_rates, results_df['Final Val Loss'], marker='s', label='Validation Loss')
plt.xscale('log')
plt.xticks(refined_learning_rates)
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss\nRefined Learning Rate")
plt.ylim(0.032, 0.04)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Parameter Optimisation/2. Learning Rate/loss_vs_refined_lr_plot.png", dpi=600)
plt.show()

print("Refined Learning Rate Optimization Completed.")