# Identifying Cyberattacks using Machine Learning Techniques

This repository contains the code and supporting material for the undergraduate dissertation titled **"Identifying Cyberattacks using Machine Learning Techniques"**. The project applies Artificial Neural Networks (ANNs) and Variable Importance Techniques to a labelled cyberattack dataset HIKARI-2022, with a focus on classification performance and interpretability.

---

## 📚 Table of Contents

- [📁 Project Structure](#project-structure)
- [📂 External Data Files](#external-data-files)
- [📂 Dataset](#dataset)
- [🧠 Models and Methods](#models-and-methods)
- [📊 Output Highlights](#output-highlights)
- [🧪 Environment Setup](#environment-setup)

---

## Project Structure


```text
├── Logistic Regression/
│   ├── Logistic Regression Model.py
│   ├── Classification Reports
│   ├── Confusion Matrices
│   └── Metrics Summary
├── Parameter Optimisation/
│   ├── 0. Raw Data/
│   ├── 1. Preprocessing/
│   ├── 2-7. Hyperparameter Tuning Experiments/
│   ├── 8. Baseline vs Optimised Model/
│   ├── optimised_model.keras
├── Variable Importance/
│   ├── Variable Importance.py
│   ├── SHAP & PFI Graphs
│   └── Results Tables
```
---

## External Data Files

Due to GitHub file size limitations, the following data files are hosted externally:

- [`ALLFLOWMETER_HIKARI2022.csv`](https://drive.google.com/file/d/1vEjwOeRym9Rm31AaXo7B3Db_HGsg_-Or/view?usp=sharing) (119 MB)

  This file contains the raw data as explained [below](#dataset). Download and place it in `Parameter Optimisation/0. Raw Data/` as it is required for preprocessing.

- [`X_train.csv`](https://drive.google.com/file/d/1QsfljwP7oCdvb1CumvwsfN66XZ7lGZnj/view?usp=sharing) (404 MB)
- [`X_val.csv`](https://drive.google.com/file/d/104nDxyQJmej6qrG0ohhtc-KFPCM2jvY9/view?usp=sharing) (101 MB)
- [`train_data_with_smotenc_tracking.csv`](https://drive.google.com/file/d/1FCWGzMumYVCcbMWsAhjKfAtCQ3xtYdSe/view?usp=sharing) (172 MB)

  These files are required for training, and evaluation. Download and place them in `Parameter Optimisation/1. Preprocessing/SMOTENC/` as needed.

---

## Dataset

This project uses the **HIKARI-2022** dataset for evaluating cyberattack detection models. The dataset was proposed by:

> Ferriyan, A., Thamrin, A. H., Takeda, K., & Murai, J. (2021).  
> *Generating Network Intrusion Detection Dataset Based on Real and Encrypted Synthetic Attack Traffic.*  
> _Applied Sciences_, 11(17), 7868. [https://doi.org/10.3390/app11177868](https://doi.org/10.3390/app11177868)

The dataset is publicly available for download via [Zenodo](https://zenodo.org/record/6463389).

---

## Models and Methods

- **Artificial Neural Networks (ANNs)**  
  - Built with Keras and TensorFlow
  - Extensive hyperparameter tuning (learning rate, layers, dropout, etc.) with outputs for each stage
  - Final model saved as `optimised_model.keras`

- **Logistic Regression**  
  - Used as a baseline model
  - Includes classification reports, confusion matrices, and metrics

- **Variable Importance**  
  - Analysed using:
    - **Permutation Feature Importance (PFI)**
    - **SHAP values** using DeepSHAP explainer

---

## Output Highlights

- Performance comparisons between baseline and optimised models
- Evaluation metrics: F1-score, Precision, Accuracy, ROC-AUC, PR-AUC
- Clear visualisations of feature contributions using DeepSHAP and PFI
  
---

## Environment Setup

This project was developed with Python 3.11 using Spyder as the IDE of choice.
Required packages include:
- numpy, pandas, scikit-learn, tensorflow, keras, shap, imbalanced-learn, matplotlib, tqdm, h5py