#%% Dependencies

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score, confusion_matrix

#%% Initialisation

# Set Seed
# SEED = 2025
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# Load preprocessed datasets
X_train = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/X_train.csv")
y_train = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/y_train.csv").values.ravel()
X_test = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/X_test.csv")
y_test = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/y_test.csv").values.ravel()

#%% Define Baseline Parameters

BASELINE_PARAMS = {
    "learning_rate": 0.001,   # Default learning rate recommended for Adam 
    "hidden_layers": [128],   # Single hidden layer with 128 units
    "batch_size": 32,         # Default batch size
    "epochs": 20,             # Fixed number of epochs
    "threshold": 0.5          # Standard classification threshold
}

#%% Define Model 

# Function to create the baseline model
def create_baseline_model():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    for units in BASELINE_PARAMS["hidden_layers"]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=BASELINE_PARAMS["learning_rate"])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

#%% Model Fitting

# Train the baseline model
baseline_model = create_baseline_model()
history = baseline_model.fit(X_train, y_train, epochs=BASELINE_PARAMS["epochs"], batch_size=BASELINE_PARAMS["batch_size"],
                             verbose=1, validation_data=(X_test, y_test))

# Get predicted probabilities
y_pred_prob = baseline_model.predict(X_test)
y_pred = (y_pred_prob >= BASELINE_PARAMS["threshold"]).astype(int)

#%% Get Metrics

# Compute Metrics
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_prob)
pr_auc = average_precision_score(y_test, y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Store baseline results
baseline_results = {
    "F1-Score": f1,
    "AUC-ROC": auc_roc,
    "PR-AUC": pr_auc,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall
}
pd.DataFrame([baseline_results]).to_csv("Parameter Optimisation/8. Baseline Model vs Optimised Model/baseline_results.csv", index=False)

print("\nBaseline Model Performance:")
for metric, value in baseline_results.items():
    print(f"{metric}: {value:.4f}")

#%% Plotting

# Plot Training and Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, BASELINE_PARAMS["epochs"] + 1), history.history['loss'], marker='o', label='Training Loss')
plt.plot(range(1, BASELINE_PARAMS["epochs"] + 1), history.history['val_loss'], marker='s', label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(range(1, BASELINE_PARAMS["epochs"] + 1))
plt.title("Baseline Model: Training and Validation Loss")
plt.legend()
plt.grid()
plt.savefig("Parameter Optimisation/8. Baseline Model vs Optimised Model/loss_vs_epochs_baseline.png", dpi=600)
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"], annot_kws={"size": 16})
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("Actual", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Confusion Matrix - Baseline NN Model", fontsize=18)
plt.savefig("Parameter Optimisation/8. Baseline Model vs Optimised Model/confusion_matrix_baseline.png", dpi=600)
plt.show()

# Save Confusion Matrix to csv
print(conf_matrix)
baseline_conf_matrix = pd.DataFrame(conf_matrix) 
baseline_conf_matrix.to_csv("Parameter Optimisation/8. Baseline Model vs Optimised Model/confusion_matrix_baseline.csv", index=False, header=False)


print("Done")
