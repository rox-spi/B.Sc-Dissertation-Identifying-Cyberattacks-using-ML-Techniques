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

#%% Define Baseline Parameters

BASELINE_PARAMS = {
    "learning_rate": 0.001,   # Default learning rate recommended for Adam 
    "hidden_layers": [128],   # Single hidden layer with 128 units
    "batch_size": 32,         # Default batch size
    "epochs": 20,             # Fixed number of epochs
    "threshold": 0.5          # Standard classification threshold
}

#%% Function to Create Model with Different Activation Functions

def create_model(activation):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    for units in BASELINE_PARAMS["hidden_layers"]:
        model.add(Dense(units, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=BASELINE_PARAMS["learning_rate"])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

#%% Test ReLU, Sigmoid, and Tanh Activation Functions

activation_functions = ["relu", "sigmoid", "tanh"]
results = []

for activation in activation_functions:
    print(f"\nTraining model with {activation} activation function...\n")
    
    # Train the model
    model = create_model(activation)
    history = model.fit(X_train, y_train, epochs=BASELINE_PARAMS["epochs"], batch_size=BASELINE_PARAMS["batch_size"],
                        verbose=1, validation_data=(X_val, y_val))
    
    # Get predicted probabilities
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
    results.append({
        "Activation": activation,
        "F1-Score": f1,
        "AUC-ROC": auc_roc,
        "PR-AUC": pr_auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    })

    print(f"\nPerformance for {activation} activation:")
    print(f"F1-Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}, PR-AUC: {pr_auc:.4f}, "
          f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

#%% Save Results

results_df = pd.DataFrame(results)
results_df.to_csv("Parameter Optimisation/8. Baseline Model vs Optimised Model/activation_function_comparison.csv", index=False)

#%% Plot Comparison

plt.figure(figsize=(10, 6))
for metric in ["F1-Score", "AUC-ROC", "PR-AUC", "Accuracy", "Precision", "Recall"]:
    plt.plot(results_df["Activation"], results_df[metric], marker='o', label=metric)

plt.xlabel("Activation Function")
plt.ylabel("Metric Value")
plt.title("Comparison of Activation Functions")
plt.legend()
plt.grid()
plt.savefig("Parameter Optimisation/8. Baseline Model vs Optimised Model/activation_function_comparison.png", dpi=600)
plt.show()

print("Done")
