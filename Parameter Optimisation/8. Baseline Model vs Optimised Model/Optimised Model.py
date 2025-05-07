#%% Dependencies

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
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

#%% Define Optimised Parameters

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
    
with open("Parameter Optimisation/7. Threshold/best_threshold.txt", "r") as f:
    best_threshold = float(f.read().strip())

# Store optimized parameters
OPTIMISED_PARAMS = {
    "learning_rate": best_refined_learning_rate,
    "hidden_layers": best_hidden_layers,
    "batch_size": best_batch_size,
    "epochs": best_epochs,
    "patience": best_patience,
    "dropout": best_dropout,
    "threshold": best_threshold
}

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

#%% Model Fitting

model = create_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=OPTIMISED_PARAMS["patience"], restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=OPTIMISED_PARAMS["epochs"],
    batch_size=OPTIMISED_PARAMS["batch_size"],
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[early_stopping]
)

# Predict probabilities
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob >= OPTIMISED_PARAMS["threshold"]).astype(int)

# Save trained model
model.save("Parameter Optimisation/optimised_model.keras")

#%% Get Metrics

# Compute Metrics
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_prob)
pr_auc = average_precision_score(y_test, y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Store baseline results
optimised_results = {
    "F1-Score": f1,
    "AUC-ROC": auc_roc,
    "PR-AUC": pr_auc,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall
}
pd.DataFrame([optimised_results]).to_csv("Parameter Optimisation/8. Baseline Model vs Optimised Model/optimised_results.csv", index=False)

print("\nOptimised Model Performance:")
for metric, value in optimised_results.items():
    print(f"{metric}: {value:.4f}")

#%% Plotting

epochs_ran = len(history.history['loss'])

# Plot Training and Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs_ran + 1), history.history['loss'], marker='o', label='Training Loss')
plt.plot(range(1, epochs_ran + 1), history.history['val_loss'], marker='s', label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(range(1, epochs_ran + 1))
plt.title("Optimised Model: Training and Validation Loss")
plt.legend()
plt.grid()
plt.savefig("Parameter Optimisation/8. Baseline Model vs Optimised Model/loss_vs_epochs_optimised.png", dpi=600)
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"], annot_kws={"size": 16})
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("Actual", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Confusion Matrix - Optimised NN Model", fontsize=18)
plt.savefig("Parameter Optimisation/8. Baseline Model vs Optimised Model/confusion_matrix_optimised.png", dpi=600)
plt.show()

# Save Confusion Matrix to csv
optimised_conf_matrix = pd.DataFrame(conf_matrix) 
optimised_conf_matrix.to_csv("Parameter Optimisation/8. Baseline Model vs Optimised Model/confusion_matrix_optimised.csv", index=False, header=False)

print("Done")
