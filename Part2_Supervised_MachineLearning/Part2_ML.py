import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report
)

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_PATH = "Part2_Supervised_MachineLearning"
PLOT_PATH = os.path.join(BASE_PATH, "plots")
os.makedirs(PLOT_PATH, exist_ok=True)


# ==========================================================
# Task 1: Load cleaned dataset and define target
# ==========================================================

print("\nTask 1: Load cleaned dataset")

df = pd.read_csv(
    "Part1_DataAcquisition_Cleaning_EDA/cleaned_data.csv"
)

print("\nDataset shape:", df.shape)
print("\nColumn data types:")
print(df.dtypes)

print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Remove rows with a missing target, if any
df = df.dropna(subset=["Churn"]).copy()

# Remove target and customer identifier from features
X = df.drop(columns=["Churn", "customerID"], errors="ignore").copy()

# Convert TotalCharges to numeric if it exists
if "TotalCharges" in X.columns:
    X["TotalCharges"] = pd.to_numeric(
        X["TotalCharges"], errors="coerce"
    )

# Encode target: No = 0, Yes = 1
y = df["Churn"].astype(str).str.strip().map({
    "No": 0,
    "Yes": 1
})

if y.isnull().any():
    raise ValueError(
        "Unexpected values found in Churn. "
        "Check the target column."
    )

y = y.astype(int)

print("\nTarget distribution:")
print(y.value_counts().rename(
    index={0: "No Churn", 1: "Churn"}
))

print("\nTarget distribution (%):")
print(
    (y.value_counts(normalize=True) * 100)
    .rename(index={0: "No Churn", 1: "Churn"})
    .round(2)
)


# ==========================================================
# Task 2: Encode categorical columns and preprocess features
# ==========================================================

print("\nTask 2: Encode categorical columns")

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)

# Inspect categorical values
for col in categorical_columns:
    print(f"\n{col}:")
    print(X[col].value_counts(dropna=False).head(10))

# Recalculate columns after TotalCharges conversion
categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

# Numerical preprocessing:
# Median imputation followed by standardization
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing:
# Most-frequent imputation followed by one-hot encoding
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        drop="first"
    ))
])

# Combine transformations
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns)
    ],
    remainder="drop"
)

print("\nPreprocessing pipeline created successfully.")
print("\nTask 2 completed.")


# ==========================================================
# Task 3: Stratified train-test split and preprocessing
# ==========================================================

print("\nTask 3: Train-test split and preprocessing")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining target percentages:")
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\nTesting target percentages:")
print((y_test.value_counts(normalize=True) * 100).round(2))

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)


# ==========================================================
# Task 4: Logistic Regression classification model
# ==========================================================

print("\nTask 4: Classification model - Logistic Regression")

model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE
)

model.fit(X_train_processed, y_train)

# Predictions
y_pred = model.predict(X_test_processed)
probability = model.predict_proba(X_test_processed)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, probability)

print("\nLogistic Regression Results")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"],
    zero_division=0
))


# ==========================================================
# Task 5a: ROC curve
# ==========================================================

print("\nTask 5a: ROC curve")

fpr, tpr, thresholds = roc_curve(y_test, probability)

plt.figure(figsize=(7, 5))
plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC={roc_auc:.3f})"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Telco Customer Churn - ROC Curve")
plt.grid(alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()

roc_path = os.path.join(PLOT_PATH, "roc_curve.png")
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"ROC curve saved to: {roc_path}")


# ==========================================================
# Task 5b: Decision-threshold sensitivity
# ==========================================================

print("\nTask 5b: Decision-threshold sensitivity")

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1 Score':<12}"
)

best_f1 = -1
best_threshold = 0.50

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:

    y_pred_threshold = (
        probability >= threshold
    ).astype(int)

    precision_threshold = precision_score(
        y_test, y_pred_threshold, zero_division=0
    )
    recall_threshold = recall_score(
        y_test, y_pred_threshold, zero_division=0
    )
    f1_threshold = f1_score(
        y_test, y_pred_threshold, zero_division=0
    )

    print(
        f"{threshold:<12.2f}"
        f"{precision_threshold:<12.2f}"
        f"{recall_threshold:<12.2f}"
        f"{f1_threshold:<12.2f}"
    )

    if f1_threshold > best_f1:
        best_f1 = f1_threshold
        best_threshold = threshold

print(f"\nBest tested threshold: {best_threshold:.2f}")
print(f"Corresponding F1 score: {best_f1:.4f}")

# The threshold comparison above is exploratory.
# Select the final threshold using validation data, then
# evaluate the selected threshold on the held-out test set.


# ==========================================================
# Task 6: Regularization experiment
# ==========================================================

print("\nTask 6: Regularization experiment on Logistic Regression")

regularized_model = LogisticRegression(
    C=0.01,
    max_iter=2000,
    random_state=RANDOM_STATE
)

regularized_model.fit(X_train_processed, y_train)

y_pred_regularized = regularized_model.predict(
    X_test_processed
)

y_prob_regularized = regularized_model.predict_proba(
    X_test_processed
)[:, 1]

accuracy_regularized = accuracy_score(
    y_test, y_pred_regularized
)
precision_regularized = precision_score(
    y_test, y_pred_regularized, zero_division=0
)
recall_regularized = recall_score(
    y_test, y_pred_regularized, zero_division=0
)
f1_regularized = f1_score(
    y_test, y_pred_regularized, zero_division=0
)
auc_regularized = roc_auc_score(
    y_test, y_prob_regularized
)

print("\nRegularized Logistic Regression (C=0.01)")
print(f"Accuracy : {accuracy_regularized:.4f}")
print(f"Precision: {precision_regularized:.4f}")
print(f"Recall   : {recall_regularized:.4f}")
print(f"F1 Score : {f1_regularized:.4f}")
print(f"ROC-AUC  : {auc_regularized:.4f}")

print("\nBaseline vs Regularized Model")
print(f"Baseline ROC-AUC   : {roc_auc:.4f}")
print(f"Regularized ROC-AUC: {auc_regularized:.4f}")


# ==========================================================
# Task 7: Bootstrap confidence interval for AUC difference
# ==========================================================

print("\nTask 7: Bootstrap confidence interval for AUC difference")

auc_difference = []
n = len(y_test)

rng = np.random.default_rng(RANDOM_STATE)
y_test_array = y_test.to_numpy()

for i in range(1000):

    # Sample test observations with replacement
    indices = rng.choice(n, size=n, replace=True)
    y_boot = y_test_array[indices]

    # ROC-AUC requires both classes in the sample
    if len(np.unique(y_boot)) < 2:
        continue

    prob_baseline = probability[indices]
    prob_regularized = y_prob_regularized[indices]

    auc_baseline_boot = roc_auc_score(
        y_boot, prob_baseline
    )
    auc_regularized_boot = roc_auc_score(
        y_boot, prob_regularized
    )

    auc_difference.append(
        auc_baseline_boot - auc_regularized_boot
    )

if len(auc_difference) == 0:
    print("No valid bootstrap samples were available.")

else:
    mean_difference = np.mean(auc_difference)

    lower = np.percentile(auc_difference, 2.5)
    upper = np.percentile(auc_difference, 97.5)

    print(f"Valid bootstrap samples: {len(auc_difference)}")
    print(f"Mean AUC Difference: {mean_difference:.4f}")
    print(
        f"95% Confidence Interval: [{lower:.4f}, {upper:.4f}]"
    )

    if lower > 0 or upper < 0:
        print("The 95% confidence interval excludes zero.")
    else:
        print("The 95% confidence interval includes zero.")

print("\nPart 2: Supervised Machine Learning completed.")