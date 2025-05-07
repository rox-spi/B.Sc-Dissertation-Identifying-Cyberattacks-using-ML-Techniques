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

with open("Parameter Optimisation/6. Dropout/best_dropout.txt", "r") as f:
    best_dropout = eval(f.read().strip())  # Convert stored string back to tuple

# Store optimized parameters
OPTIMISED_PARAMS = {
    "learning_rate": best_refined_learning_rate,
    "hidden_layers": best_hidden_layers,
    "batch_size": best_batch_size,
    "epochs": best_epochs,
    "patience": best_patience,
    "dropout": best_dropout
}

# Define thresholds to test
threshold_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
num_trials = len(threshold_options)
print(f"Total Configurations to Test: {num_trials}")

#%% Define Model

# Function to create model with variable dropout per layer
def create_model():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    for units, dropout in zip(OPTIMISED_PARAMS["hidden_layers"], OPTIMISED_PARAMS["dropout"]):
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout))  # Apply layer-specific dropout
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=OPTIMISED_PARAMS["learning_rate"])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

#%% Train Model Once and Generate Probabilities

model = create_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=OPTIMISED_PARAMS["patience"], restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=OPTIMISED_PARAMS["epochs"],
    batch_size=OPTIMISED_PARAMS["batch_size"],
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[early_stopping]
)

# Predict probabilities
y_pred_prob = model.predict(X_val)

#%% Search

# List to store results
results = []

# Loop through threshold configurations
for i, threshold in enumerate(threshold_options, start=1):
    y_pred = (y_pred_prob >= threshold).astype(int)

    # Compute Metrics
    f1 = f1_score(y_val, y_pred)
    auc_roc = roc_auc_score(y_val, y_pred_prob)
    pr_auc = average_precision_score(y_val, y_pred_prob)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)

    # Store Results 
    results.append([threshold, f1, auc_roc, pr_auc, accuracy, precision, recall])
    print(f"Threshold {threshold:.2f}, F1-score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}, PR-AUC: {pr_auc:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n")

# Convert results to DataFrame and save
results_df = pd.DataFrame(results, columns=['Threshold', 'F1-Score', 'AUC-ROC', 'PR-AUC', 'Accuracy', 'Precision', 'Recall'])
results_df.to_csv("Parameter Optimisation/7. Threshold/threshold_results.csv", index=False)

# Identify the best configuration
best_idx = results_df['F1-Score'].idxmax()
best_config = results_df.loc[best_idx]

# Save the best threshold
with open("Parameter Optimisation/7. Threshold/best_threshold.txt", "w") as f:
    f.write(str(best_config['Threshold']))

print("\nBest Threshold and Associated Metrics:")
print(best_config.to_string(index=True))

#%% Plotting

plt.figure(figsize=(8, 6))
plt.plot(threshold_options, results_df['F1-Score'], marker='o', label='F1-Score')
plt.plot(threshold_options, results_df['Accuracy'], marker='s', label='Accuracy')
plt.plot(threshold_options, results_df['Precision'], marker='^', label='Precision')
plt.plot(threshold_options, results_df['Recall'], marker='d', label='Recall')
plt.plot(threshold_options, results_df['AUC-ROC'], marker='X', label='AUC-ROC')
plt.plot(threshold_options, results_df['PR_AUC'], marker='p', label='PR-AUC')
plt.xlabel("Threshold")
plt.ylabel("Results")
plt.title("Performance Metrics\nClassification Threshold")
plt.xticks(threshold_options)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.savefig("Parameter Optimisation/7. Threshold/metrics_vs_threshold.png", dpi=600)
plt.show()

print("Threshold tuning completed.")