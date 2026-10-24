import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


print("\nPreprocessing Steps:")

# Load cleaned Telco dataset
df = pd.read_csv(
    "Part1_DataAcquisition_Cleaning_EDA/cleaned_data.csv"
)

print(df.dtypes)

# Define features and target
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Drop customer ID because it does not help predict churn
if "customerID" in X.columns:
    X = X.drop(columns=["customerID"])

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical columns:", categorical_columns)
print("Numerical columns:", numerical_columns)

# Fill missing values and encode categorical columns
X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

# Fill any remaining missing numerical values
X = X.fillna(X.median())

print("\nEncoded feature matrix:")
print(X.head())
print("Feature matrix shape:", X.shape)

# Classification train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Task 1: Baseline Decision Tree

print("\nTask 1: Baseline Decision Tree")

decision_tree_baseline = DecisionTreeClassifier(
    random_state=42
)

decision_tree_baseline.fit(X_train_scaled, y_train)

print(
    "Train Accuracy:",
    decision_tree_baseline.score(X_train_scaled, y_train)
)
print(
    "Test Accuracy:",
    decision_tree_baseline.score(X_test_scaled, y_test)
)


# Task 2: Controlled Decision Tree

print("\nTask 2: Controlled Decision Tree")

controlled_tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    random_state=42
)

controlled_tree.fit(X_train_scaled, y_train)

print(
    "Train Accuracy:",
    controlled_tree.score(X_train_scaled, y_train)
)
print(
    "Test Accuracy:",
    controlled_tree.score(X_test_scaled, y_test)
)


# Task 3: Gini vs Entropy

print("\nTask 3: Gini vs Entropy")

for criterion in ["gini", "entropy"]:

    model = DecisionTreeClassifier(
        max_depth=5,
        criterion=criterion,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    print("\nCriterion:", criterion)
    print("Test Accuracy:", accuracy_score(y_test, predictions))
    print("ROC-AUC:", roc_auc_score(y_test, probabilities))


# Task 4: Random Forest

print("\nTask 4: Random Forest")

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

random_forest.fit(X_train_scaled, y_train)

print("Train Accuracy:", random_forest.score(X_train_scaled, y_train))
print("Test Accuracy:", random_forest.score(X_test_scaled, y_test))

rf_probability = random_forest.predict_proba(X_test_scaled)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probability)

print("Random Forest ROC-AUC:", rf_auc)

# Top 5 important features
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest.feature_importances_
})

print("\nTop 5 important features:")
print(
    feature_importance.sort_values(
        by="Importance",
        ascending=False
    ).head(5)
)


# Task 4a: Gradient Boosting

print("\nTask 4a: Gradient Boosting")

gradient_boosting = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gradient_boosting.fit(X_train_scaled, y_train)

print("Train Accuracy:", gradient_boosting.score(X_train_scaled, y_train))
print("Test Accuracy:", gradient_boosting.score(X_test_scaled, y_test))

gb_probability = gradient_boosting.predict_proba(X_test_scaled)[:, 1]

print(
    "Gradient Boosting ROC-AUC:",
    roc_auc_score(y_test, gb_probability)
)


# Task 4b: Feature Ablation Study

print("\nTask 4b: Feature Ablation")

least_important = feature_importance.sort_values(
    by="Importance",
    ascending=True
).head(5)["Feature"].tolist()

print("Features removed:", least_important)

X_train_reduced = X_train.drop(columns=least_important)
X_test_reduced = X_test.drop(columns=least_important)

scaler_reduced = StandardScaler()

X_train_reduced_scaled = scaler_reduced.fit_transform(X_train_reduced)
X_test_reduced_scaled = scaler_reduced.transform(X_test_reduced)

reduced_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

reduced_rf.fit(X_train_reduced_scaled, y_train)

reduced_probability = reduced_rf.predict_proba(
    X_test_reduced_scaled
)[:, 1]

reduced_auc = roc_auc_score(y_test, reduced_probability)

print("Full Model ROC-AUC:", rf_auc)
print("Reduced Model ROC-AUC:", reduced_auc)


# Task 5: Cross-Validated Comparison

print("\nTask 5: Cross-Validated Comparison")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc"
    )

    print("\n", name)
    print("Mean AUC:", scores.mean())
    print("Standard Deviation:", scores.std())


# Task 6: Hyperparameter Tuning

print("\nTask 6: Hyperparameter Tuning")

pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5]
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:")
print(grid.best_params_)

print("Best Cross-Validated ROC-AUC:", grid.best_score_)

best_model = grid.best_estimator_

best_probability = best_model.predict_proba(X_test)[:, 1]

print(
    "Tuned Model Test ROC-AUC:",
    roc_auc_score(y_test, best_probability)
)


# Task 7: Manual Learning Curve

print("\nTask 7: Manual Learning Curve")

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

print(f"{'Training Fraction':<20}{'Training AUC':<20}{'Test AUC':<20}")

for fraction in fractions:

    n_samples = int(fraction * len(X_train))

    X_subset = X_train.iloc[:n_samples]
    y_subset = y_train.iloc[:n_samples]

    # Use a new model for each training fraction
    from sklearn.base import clone

    model = clone(best_model)
    model.fit(X_subset, y_subset)

    train_probability = model.predict_proba(X_subset)[:, 1]
    test_probability = model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_subset, train_probability)
    test_auc = roc_auc_score(y_test, test_probability)

    print(
        f"{fraction:<20.0%}"
        f"{train_auc:<20.4f}"
        f"{test_auc:<20.4f}"
    )


# Task 8: Save and Load the Best Model

print("\nTask 8: Save the Best Model")

from pathlib import Path

# Save files in the same folder as this Part 3 script.
PART3_DIR = Path(__file__).resolve().parent

MODEL_PATH = PART3_DIR / "best_model.pkl"
FEATURES_PATH = PART3_DIR / "feature_columns.pkl"

# Save the trained GridSearchCV-selected pipeline.
joblib.dump(best_model, MODEL_PATH)

# Save the exact feature names and their order used during training.
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, FEATURES_PATH)

print("Best model saved to:", MODEL_PATH)
print("Feature columns saved to:", FEATURES_PATH)

# Load and verify the saved model.
loaded_model = joblib.load(MODEL_PATH)
loaded_features = joblib.load(FEATURES_PATH)

print("Model loaded successfully!")
print("Model type:", type(loaded_model).__name__)
print("Number of features:", len(loaded_features))
print("First 10 features:", loaded_features[:10])

# Verify the saved model and feature schema are consistent.
if hasattr(loaded_model, "feature_names_in_"):
    model_features = list(loaded_model.feature_names_in_)

    if model_features != loaded_features:
        raise ValueError(
            "Saved feature columns do not match the model's "
            "training feature columns."
        )

# Predict on two test customers.
sample_data = X_test.iloc[:2].copy()

predictions = loaded_model.predict(sample_data)
probabilities = loaded_model.predict_proba(sample_data)[:, 1]

print("\nPredictions for two customers:")

for i, prediction in enumerate(predictions):
    churn_label = "Yes" if int(prediction) == 1 else "No"

    print(
        f"Customer {i + 1}: "
        f"Predicted Churn = {churn_label}, "
        f"Churn Probability = {probabilities[i]:.4f}"
    )

print("\nPart 3 completed successfully!")