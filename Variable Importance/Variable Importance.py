#%% Variable Importance for Optimised ANN Model

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score
import shap

#%% Load Optimised Model and Data
SEED = 2025
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load preprocessed data
X_train = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/X_train.csv")
y_train = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/y_train.csv").values.ravel()
X_test = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/X_test.csv")
y_test = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/y_test.csv").values.ravel()

# Load optimised model
model = load_model("Parameter Optimisation/optimised_model.keras")

#%% Permutation Feature Importance

# Get baseline F1-score
y_pred = (model.predict(X_test.values, verbose=1) > 0.7).astype(int).ravel()
baseline_score = f1_score(y_test, y_pred, zero_division=0)

# Compute PFI using F1
pfi_scores = []
for i, col in enumerate(X_test.columns):
    shuffled = X_test.copy()
    rng = np.random.default_rng(SEED + i)
    shuffled[col] = rng.permutation(shuffled[col])

    y_shuffled_pred = (model.predict(shuffled.values, verbose=1) > 0.7).astype(int).ravel()
    shuffled_score = f1_score(y_test, y_shuffled_pred, zero_division=0)
    importance = baseline_score - shuffled_score
    pfi_scores.append(importance)

# Format as DataFrame
pfi_scores = pd.DataFrame(pfi_scores, index=X_test.columns, columns=["Permutation Importance (F1)"])

#%% SHAP (DeepSHAP) 

X_train_sample = X_train[:100].values
X_test_sample = X_test[:100].values

explainer = shap.DeepExplainer(model, X_train_sample)
shap_values = explainer.shap_values(X_test_sample)

shap_array = shap_values[0].T
shap_importance = pd.DataFrame(
    np.mean(np.abs(shap_array), axis=0),
    index=X_test.columns,
    columns=["SHAP (DeepSHAP)"]
)

#%% Combine Results and Save
all_scores = pd.concat([pfi_scores, shap_importance], axis=1)
all_scores = all_scores.sort_values(by="SHAP (DeepSHAP)", ascending=False)

print("\n=== VARIABLE IMPORTANCE SCORES ===")
print(all_scores.round(4))

# Optional: Save to file
all_scores.to_csv("Variable Importance/Parameter Importance Results.csv")

#%% Convert results to %
pfi_percent = 100 * (all_scores["Permutation Importance (F1)"].abs() / all_scores["Permutation Importance (F1)"].abs().sum())
shap_percent = 100 * (all_scores["SHAP (DeepSHAP)"] / all_scores["SHAP (DeepSHAP)"].sum())

# Add to DataFrame
all_scores["PFI %"] = pfi_percent
all_scores["SHAP %"] = shap_percent

# Save updated CSV
all_scores.to_csv("Variable Importance/Parameter Importance Results (with Percentages).csv")

# Print top rows
print("\n=== VARIABLE IMPORTANCE (with Percentages) ===")
print(all_scores[["Permutation Importance (F1)", "PFI %", "SHAP (DeepSHAP)", "SHAP %"]].round(4).head(10))

#%% Plots
import matplotlib.pyplot as plt

# Function for splitting variables in 2 plots
def plot_bar_split(df, column, method_name, file_prefix):
    # Sort by importance
    sorted_df = df.sort_values(by=column, ascending=False)

    # Split into two halves
    midpoint = len(sorted_df) // 2
    top_half = sorted_df.iloc[:midpoint]
    bottom_half = sorted_df.iloc[midpoint:]

    # Plot Top Half
    plt.figure(figsize=(12, 8))
    top_half[column].plot(kind='barh')
    plt.title(f"{method_name} (Top Half)")
    plt.xlabel(column)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"Variable Importance/{file_prefix}_top_half.png", dpi=600)
    plt.close()

    # Plot Bottom Half
    plt.figure(figsize=(12, 8))
    bottom_half[column].plot(kind='barh')
    plt.title(f"{method_name} (Bottom Half)")
    plt.xlabel(column)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"Variable Importance/{file_prefix}_bottom_half.png", dpi=600)
    plt.close()

# Create Bar Graphs
plot_bar_split(all_scores, "Permutation Importance (F1)", "Permutation Feature Importance (F1)", "PFI_F1")
plot_bar_split(all_scores, "SHAP (DeepSHAP)", "SHAP (DeepSHAP)", "SHAP")

# Create Bar Graphs for PERCENTAGES
plot_bar_split(all_scores, "PFI %", "Permutation Importance (F1) – %", "PFI_percent")
plot_bar_split(all_scores, "SHAP %", "SHAP (DeepSHAP) – %", "SHAP_percent")
