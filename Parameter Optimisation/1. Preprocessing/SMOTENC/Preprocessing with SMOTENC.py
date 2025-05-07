#%% Dependencies

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

#%% Import Dataset

# Set Seed
SEED = 2025
np.random.seed(SEED)

# Load Dataset
path = 'Parameter Optimisation/0. Raw Data/ALLFLOWMETER_HIKARI2022.csv'
df = pd.read_csv(path)

#describe = df.describe()
df.info()
df.head()
df.tail()
datatypes = df.dtypes

#%% Set Data Types

# Set Data Types for Columns
integer_cols = [ df.columns[0], 'originp', 'responp', 'fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_data_pkts_tot', 'bwd_data_pkts_tot', 'fwd_header_size_tot',
                'fwd_header_size_min',  'fwd_header_size_max',   'bwd_header_size_tot',   'bwd_header_size_min', 
                'bwd_header_size_max',  'flow_FIN_flag_count',   'flow_SYN_flag_count',   'flow_RST_flag_count',
                'fwd_PSH_flag_count',   'bwd_PSH_flag_count',    'flow_ACK_flag_count',   'fwd_URG_flag_count',
                'bwd_URG_flag_count',   'flow_CWR_flag_count',   'flow_ECE_flag_count',   'fwd_pkts_payload.min',
                'fwd_pkts_payload.max', 'fwd_pkts_payload.tot',  'bwd_pkts_payload.min',  'bwd_pkts_payload.max',
                'bwd_pkts_payload.tot', 'flow_pkts_payload.min', 'flow_pkts_payload.max', 'flow_pkts_payload.tot',
                'fwd_init_window_size', 'bwd_init_window_size',  'fwd_last_window_size',  'bwd_last_window_size',
                'Label']

string_cols = ['uid', 'originh', 'responh', 'flow_duration', 'attack_category']

df[integer_cols] = df[integer_cols].astype(int)
df[string_cols] = df[string_cols].astype(object)

# Remaining columns as floats
remaining_cols = [col for col in df.columns if col not in integer_cols + string_cols]

# Convert remaining columns to float
df[remaining_cols] = df[remaining_cols].astype(float)
datatypes_new = df.dtypes
datatypes_new.to_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/datatypes.csv")

# Convert 'Flow Duration' to timedelta (seconds)
df['flow_duration'] = pd.to_timedelta(df['flow_duration'])
df['flow_duration'] = df['flow_duration'].dt.total_seconds()

#%% Initial Checks

# Check for Missing Data
missing = df.isnull().sum() # no missing values

# Check Cardinality of Each Feature
cardinality = df.nunique().sort_values(ascending=False)
cardinality.to_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/cardinality.csv")

# Drop select columns
columns_to_drop = [df.columns[0], 'uid', 'originh', 'responh', 'fwd_URG_flag_count', 'bwd_URG_flag_count', 'attack_category']
# Reasons: 1. Column 0: just a row number
#          2. uid: unique identifier per flow
#          3. originh: is the source host/ip 
#          4. responh: is the destination host/ip  
#          5. fwd_URG_flag_count: has one unique value
#          6. bwd_URG_flag_count: has one unique value 
#          7. attack_category: will be a dead giveaway of benign or malicious traffic

dfnew = df.drop(columns=columns_to_drop)

# Define Categorical Columns
init_categorical_cols = ['fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_data_pkts_tot', 'bwd_data_pkts_tot',
                    'flow_FIN_flag_count', 'flow_SYN_flag_count', 'flow_RST_flag_count', 
                    'fwd_PSH_flag_count', 'bwd_PSH_flag_count',	'flow_ACK_flag_count', 'flow_CWR_flag_count',
                    'flow_ECE_flag_count', 'Label']

# Verify Categorical Columns are of Integer Values
print(df.dtypes[init_categorical_cols])

#%% Checking Categorical Columns

### Categorical Descriptive Statistics ###
init_categorical_stats = {}

for col in init_categorical_cols:
    value_counts = dfnew[col].value_counts()
    percentage = (value_counts / len(dfnew[col])) * 100

    init_categorical_stats[col] = pd.DataFrame({
        "Category": value_counts.index,
        "Frequency": value_counts.values,
        "Percentage (%)": percentage.values
    })

    # Save each categorical column statistics separately
    init_categorical_stats[col].to_csv(f"Parameter Optimisation/1. Preprocessing/SMOTENC/Initial Categorical/init_categorical_summary_{col}.csv", index=False)

    print(f"\nInitial Categorical Summary for {col}:")
    print(init_categorical_stats[col])

# Save overall categorical summary for reference
init_categorical_summary_overall = pd.DataFrame({
    "Unique Categories": {col: dfnew[col].nunique() for col in init_categorical_cols}
})

init_categorical_summary_overall.to_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/Initial Categorical/init_categorical_summary_overall.csv")

print("\nInitial Overall Categorical Summary:")
print(init_categorical_summary_overall)

#%% Recoding Categorical Columns

# After going through each categorical variable, some are to be taken as numerical 
# due to the nature of their 'categories', others will undergo recoding of categories, 
# and some will be excluded due to having one dominating category.

# Variables to drop with 99.99% single category
drop_cols = ['flow_ECE_flag_count', 'flow_CWR_flag_count']
df_final = dfnew.drop(columns=drop_cols)

# Variables to be kept as categorical
categorical_cols = ['flow_RST_flag_count', 'flow_SYN_flag_count', 'flow_FIN_flag_count', 'fwd_data_pkts_tot', 'Label']

### Recoding Categories for Categorical Columns ###
def recode_column(value, thresholds):
    for i, threshold in enumerate(thresholds, start=1):
        if value < threshold:
            return i - 1
    return len(thresholds)

recode_thresholds = {
    'flow_RST_flag_count': [1, 2, 3], # categories will be 0,1,2,3+
    'flow_SYN_flag_count': [1, 2, 3], # categories will be 0,1,2,3+
    'flow_FIN_flag_count': [1, 2, 3], # categories will be 0,1,2,3+
    'fwd_data_pkts_tot': [1, 2, 3, 4] # categories will be 0,1,2,3,4+
}

for col, thresholds in recode_thresholds.items():
    df_final[col] = df_final[col].apply(lambda x: recode_column(x, thresholds))
    df_final[col] = df_final[col].astype('category')

### Categorical Columns Descriptive Statistics ###
categorical_stats = {}
for col in categorical_cols:
    value_counts = df_final[col].value_counts()
    percentage = (value_counts / len(df_final[col])) * 100
    categorical_stats[col] = pd.DataFrame({
        "Category": value_counts.index,
        "Frequency": value_counts.values,
        "Percentage (%)": percentage.values
    })
    categorical_stats[col].to_csv(f"Parameter Optimisation/1. Preprocessing/SMOTENC/Categorical/categorical_summary_{col}.csv", index=False)

    print(f"\nCategorical Summary for {col}:")
    print(categorical_stats[col])

# Save overall categorical summary for reference
categorical_summary_overall = pd.DataFrame({
    "Unique Categories": {col: df_final[col].nunique() for col in categorical_cols}
})

categorical_summary_overall.to_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/Categorical/categorical_summary_overall.csv")

print("\nInitial Overall Categorical Summary:")
print(categorical_summary_overall)
#%% Checking Numerical Columns

### Numerical Descriptive Statistics ###
numerical_summary = df_final.drop(columns=['Label']).describe().transpose()
#numerical_summary["median"] = dfnew.median()
numerical_summary["mode"] = df_final.mode().iloc[0]  # Get first mode for each column
numerical_summary["range"] = numerical_summary["max"] - numerical_summary["min"]

# Selecting relevant statistics
numerical_summary = numerical_summary[["mean", "mode", "std", "25%", "50%", "75%", "range"]]

# Save numerical summary
numerical_summary.to_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/numerical_summary.csv")

print("Numerical Summary:")
print(numerical_summary)

#%% Data Partitioning

# Split dataset into X (features) and y (target)
X = df_final.drop(columns=['Label'])
y = df_final['Label']

# 80-20 Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

# Count number of Samples in each partition
train_counts = y_train.value_counts()
test_counts = y_test.value_counts()
train_size = len(y_train)
test_size = len(y_test)

# Prepare summary
summary = pd.DataFrame({
    "Set": ["Train", "Test"],
    "Total Rows": [train_size, test_size],
    "Benign (0)": [train_counts.get(0, 0), test_counts.get(0, 0)],
    "Attack (1)": [train_counts.get(1, 0), test_counts.get(1, 0)]
})

print(summary)

#%% Using SMOTENC

# Identify Categorical Columns minus Label
categorical_cols.remove('Label')

# Add column to identify which samples are from dataset
X_train['is_synthetic'] = 0 # real samples marked by 0

# Applying SMOTENC
# Define categorical feature indices
categorical_indices = [X_train.columns.get_loc(col) for col in categorical_cols]

# Apply SMOTENC on Train Set (excluding the 'is_synthetic' column)
smote = SMOTENC(categorical_features=categorical_indices, random_state=SEED)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train.drop(columns=['is_synthetic']), y_train)

# Identify new synthetic samples
num_new_samples = len(X_train_resampled) - len(X_train)
synthetic_indices = range(len(X_train), len(X_train_resampled))

# Convert to Dataframe and mark synthetic rows
X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns[:-1])  # exclude 'is_synthetic' column
X_train_resampled['is_synthetic'] = 0  # Default all to real '0'
X_train_resampled.loc[synthetic_indices, 'is_synthetic'] = 1 # mark those synthetic with '1'

# Reattach labels
df_train_resampled = X_train_resampled.copy()
df_train_resampled['Label'] = y_train_resampled

# Save the resampled training set with tracking of synthetic samples
df_train_resampled.to_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/train_data_with_smotenc_tracking.csv", index=False)

# Prepare Summary after SMOTENC
train_counts_resampled = y_train_resampled.value_counts()

summary_after_smote = pd.DataFrame({
    "Set": ["Train (Resampled)", "Test"],
    "Total Rows": [len(y_train_resampled), len(y_test)],
    "Benign (0)": [train_counts_resampled.get(0, 0), test_counts.get(0, 0)],
    "Attack (1)": [train_counts_resampled.get(1, 0), test_counts.get(1, 0)]
})

print(summary_after_smote)

#%% Validation Split

# Split the resampled training set into training (80%) and validation (20%)
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_resampled, y_train_resampled, test_size=0.2, random_state=SEED, stratify=y_train_resampled)

# Count number of samples in each partition after validation split
train_counts_final = pd.Series(y_train_final).value_counts()
val_counts = pd.Series(y_val).value_counts()

# Prepare final summary
summary_final = pd.DataFrame({
    "Set": ["Train", "Validation", "Test"],
    "Total Rows": [len(y_train_final), len(y_val), len(y_test)],
    "Benign (0)": [train_counts_final.get(0, 0), val_counts.get(0, 0), test_counts.get(0, 0)],
    "Attack (1)": [train_counts_final.get(1, 0), val_counts.get(1, 0), test_counts.get(1, 0)]
})

print(summary_final)

#%% Scaling and Saving

# Scale the features (excluding categorical cols and synthetic tracking column)
scaler = StandardScaler()
numerical_cols = [col for col in X_train.columns if col not in categorical_cols + ['is_synthetic']]
X_train_final[numerical_cols] = scaler.fit_transform(X_train_final[numerical_cols])
X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Save scaled versions of sets
X_train_final.drop(columns=['is_synthetic']).to_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/X_train.csv", index=False)
pd.DataFrame(y_train_final, columns=['Label']).to_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/y_train.csv", index=False)
X_val.drop(columns=['is_synthetic']).to_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/X_val.csv", index=False)
pd.DataFrame(y_val, columns=['Label']).to_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/y_val.csv", index=False)
X_test.to_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/X_test.csv", index=False)
y_test.to_csv("Parameter Optimisation/1. Preprocessing/SMOTENC/y_test.csv", index=False)

print("Done")