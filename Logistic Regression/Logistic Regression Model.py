#%% Dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

#%% Initialisation

SEED = 2025

# Load data
X_train = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/X_train.csv")
y_train = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/y_train.csv")['Label']
X_val = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/X_val.csv")
y_val = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/y_val.csv")['Label']
X_test = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/X_test.csv")
y_test = pd.read_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/y_test.csv")

#%% Train model

# Train 
model = LogisticRegression(random_state=SEED, max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_val_pred = model.predict(X_val)
y_val_proba = model.predict_proba(X_val)[:, 1]

y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

#%% Obtain Metrics and Plots

# Define metric function
def evaluate(y_true, y_pred, y_proba):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'ROC AUC': roc_auc_score(y_true, y_proba),
        'PR AUC': average_precision_score(y_true, y_proba)
    }

# Get metrics
val_metrics = evaluate(y_val, y_val_pred, y_val_proba)
test_metrics = evaluate(y_test, y_test_pred, y_test_proba)
metrics_df = pd.DataFrame({'Validation': val_metrics, 'Test': test_metrics})

# Save metrics
metrics_df.to_csv("Logistic Regression/logreg_metrics.csv")

# Save classification reports
with open("Logistic Regression/logreg_classification_report_val.txt", "w") as f:
    f.write(classification_report(y_val, y_val_pred))

with open("Logistic Regression/logreg_classification_report_test.txt", "w") as f:
    f.write(classification_report(y_test, y_test_pred))

# Save confusion matrices
val_cm = confusion_matrix(y_val, y_val_pred)
test_cm = confusion_matrix(y_test, y_test_pred)
np.savetxt("Logistic Regression/logreg_conf_matrix_val.csv", val_cm, delimiter=",", fmt="%d")
np.savetxt("Logistic Regression/logreg_conf_matrix_test.csv", test_cm, delimiter=",", fmt="%d")

# Plot confusion matrices
def plot_confusion_matrix(cm, title, path):
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["Benign", "Attack"],
                yticklabels=["Benign", "Attack"],
                annot_kws = {'size':16})
    plt.title(title, fontsize=18)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=600)
    plt.close()
    
plot_confusion_matrix(val_cm, "Confusion Matrix - Logistic Regression", "Logistic Regression/logreg_conf_matrix_val.png")
plot_confusion_matrix(test_cm, "Confusion Matrix - Logistic Regression", "Logistic Regression/logreg_conf_matrix_test.png")
