import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load the dataset
print("\nTask 1: Load Telco Customer Churn Dataset")

BASE_PATH = "Part1_DataAcquisition_Cleaning_EDA"
df = pd.read_csv(f"{BASE_PATH}/Telco-Customer-Churn.csv")

print("First 5 rows:\n", df.head())
print("\nColumn datatypes:\n", df.dtypes)
print("\nDataFrame shape:", df.shape)
print("\nColumn names:\n", df.columns.tolist())


# Task 2: Data type correction and null value analysis
print("\nTask 2: Data Type Correction and Null Value Analysis")

# In this dataset, TotalCharges may contain blank strings.
# Convert them to NaN before analyzing missing values.
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

null_values_count = df.isnull().sum()
null_percentage = (null_values_count / len(df)) * 100

print("Null values count:\n", null_values_count)
print("\nMissing percentage:\n", null_percentage.round(2))

print("\nColumns with more than 20% missing values:")
print(null_percentage[null_percentage > 20])

print("\nColumns with some missing values:")
print(null_percentage[null_percentage > 0])


# Task 3: Duplicate detection
print("\nTask 3: Duplicate Detection")

print("Exact duplicate rows:", df.duplicated().sum())

# Check whether customerID is duplicated, if present.
if "customerID" in df.columns:
    print("Duplicate customer IDs:", df["customerID"].duplicated().sum())

# Do not automatically drop rows until duplicates are inspected.


# Task 4: Data types and memory usage
print("\nTask 4: Data Types and Memory Usage")

print("Memory before conversion:",
      df.memory_usage(deep=True).sum(), "bytes")

print("\nUnique values in object columns:")
for col in df.select_dtypes(include="object").columns:
    print(f"{col}: {df[col].nunique(dropna=False)}")

# Convert suitable string columns to category.
# Keep customerID as an identifier, not a model feature.
for col in df.select_dtypes(include="object").columns:
    if col != "customerID" and df[col].nunique() <= 20:
        df[col] = df[col].astype("category")

print("\nUpdated data types:\n", df.dtypes)
print("\nMemory after conversion:",
      df.memory_usage(deep=True).sum(), "bytes")


# Task 5: Descriptive statistics and skewness
print("\nTask 5: Descriptive Statistics and Skewness")

print(df.describe(include="number").T)

for col in df.select_dtypes(include="number").columns:
    print(f"Skewness of {col}: {df[col].skew():.2f}")

# Churn class distribution
if "Churn" in df.columns:
    print("\nChurn counts:\n", df["Churn"].value_counts(dropna=False))
    print("\nChurn percentage:\n",
          (df["Churn"].value_counts(normalize=True, dropna=False) * 100).round(2))


# Task 6: IQR outlier detection
print("\nTask 6: IQR Outlier Detection")

outlier_columns = [
    col for col in ["tenure", "MonthlyCharges", "TotalCharges"]
    if col in df.columns
]

for col in outlier_columns:
    series = df[col].dropna()

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_count = (
        (series < lower_bound) | (series > upper_bound)
    ).sum()

    print(f"\nColumn: {col}")
    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Outlier count: {outliers_count}")

# Do not remove these records automatically:
# valid high charges or long tenure may be legitimate.
    
                
# Task 7: Data Visualizations 
print("\nTask 7: Visualizations")

PLOT_PATH = f"{BASE_PATH}/plots"
os.makedirs(PLOT_PATH, exist_ok=True)

# Ensure TotalCharges is numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# A line plot of one numeric variable sorted by row index or time column (use plt.plot()). Add a title, x-label, and y-label.
plt.figure(figsize=(10, 5))
plt.plot(df["MonthlyCharges"].reset_index(drop=True), alpha=0.7)
plt.title("Monthly Charges Across Customers")
plt.xlabel("Customer Row Index")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig(f"{PLOT_PATH}/line_plot.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# A bar chart comparing the mean of one numeric column across categories of one categorical column (use plt.bar() or df.groupby().mean().plot.bar()). 
# Add a title and axis labels.

churn_rate = (
    df.groupby("Contract", observed=False)["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))
plt.bar(churn_rate.index.astype(str), churn_rate.values)
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Customers Who Churned (%)")
plt.tight_layout()
plt.savefig(f"{PLOT_PATH}/bar_chart.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# A histogram of the most skewed numeric column identified in Task 5 (use sns.histplot() with bins=20).
# Add a title and describe the shape of the distribution in the README.

numeric_df = df.select_dtypes(include="number")
skewness = numeric_df.skew().dropna()

if not skewness.empty:
    max_skew_col = skewness.abs().idxmax()

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x=max_skew_col,
        bins=20,
        kde=True,
        edgecolor="black"
    )
    plt.title(f"Distribution of {max_skew_col}")
    plt.xlabel(max_skew_col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/histogram.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Most skewed numeric column: {max_skew_col}")
    print(f"Skewness: {skewness[max_skew_col]:.3f}")

# A scatter plot between two numeric columns that you expect to be correlated (use sns.scatterplot()).
# Add a title and interpret the direction and approximate strength of the relationship in the README.

# 7d. Scatter plot: Tenure vs Total Charges
scatter_df = df.dropna(subset=["tenure", "TotalCharges"])

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=scatter_df,
    x="tenure",
    y="TotalCharges",
    hue="Churn",
    alpha=0.6
)
plt.title("Tenure vs Total Charges by Churn Status")
plt.xlabel("Tenure (Months)")
plt.ylabel("Total Charges")
plt.tight_layout()
plt.savefig(f"{PLOT_PATH}/scatterplot.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# A box plot of a numeric column split by a categorical column (use sns.boxplot()). 
# Add a title and describe any visible differences in median and spread across categories.
# 7e. Box plot: Monthly Charges by Churn status
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig(f"{PLOT_PATH}/boxplot.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# Task 8:Correlation heat map: 


print("\nTask 8: Correlation Heatmap")

corr_df = df.select_dtypes(include="number").copy()

# Exclude identifier if present
corr_df = corr_df.drop(columns=["customerID"], errors="ignore")

# Add churn as a numeric variable
corr_df["Churn_Encoded"] = df["Churn"].map({"No": 0, "Yes": 1})

corr_matrix = corr_df.corr()

plt.figure(figsize=(10, 7))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    vmin=-1,
    vmax=1,
    cmap="coolwarm"
)
plt.title("Correlation Matrix of Numeric Features and Churn")
plt.tight_layout()
plt.savefig(f"{PLOT_PATH}/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# Find the strongest absolute correlation with churn
churn_corr = corr_matrix["Churn_Encoded"].drop(
    labels=["Churn_Encoded"], errors="ignore"
).dropna()

if not churn_corr.empty:
    strongest_feature = churn_corr.abs().idxmax()
    print(
        f"Strongest numeric correlation with churn: {strongest_feature}"
    )
    print(
        f"Correlation: {churn_corr[strongest_feature]:.3f}"
    )

# Find the strongest correlation between two distinct numeric features
feature_corr = corr_matrix.drop(
    index=["Churn_Encoded"], columns=["Churn_Encoded"], errors="ignore"
)

max_corr = 0
col1, col2 = "", ""

for i in range(len(feature_corr.columns)):
    for j in range(i + 1, len(feature_corr.columns)):
        value = abs(feature_corr.iloc[i, j])
        if pd.notna(value) and value > max_corr:
            max_corr = value
            col1 = feature_corr.columns[i]
            col2 = feature_corr.columns[j]

if col1:
    print(f"Highest feature-to-feature correlation: {col1} <--> {col2}")
    print(f"Correlation value: {feature_corr.loc[col1, col2]:.3f}")



# Task 9:a. Imputation strategy comparison: 

print("\nTask 9a: Imputation Strategy Comparison")

numeric_cols = df.select_dtypes(include="number").columns
skewness = df[numeric_cols].skew().abs().dropna()
top_2_skewed = skewness.sort_values(ascending=False).head(2).index.tolist()

for col in top_2_skewed:
    print(f"\nColumn: {col}")
    print(f"Mean:   {df[col].mean():.2f}")
    print(f"Median: {df[col].median():.2f}")
    print(f"Missing values: {df[col].isna().sum()}")

    # Use median only if missing values actually exist.
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
        print(f"Missing after median imputation: {df[col].isna().sum()}")


# Task 9 :b. Spearman rank correlation: 


print("\nTask 9b: Spearman Rank Correlation")

corr_data = df.select_dtypes(include="number").drop(
    columns=["customerID"], errors="ignore"
)

pearson_corr = corr_data.corr(method="pearson")
spearman_corr = corr_data.corr(method="spearman")

difference = (spearman_corr - pearson_corr).abs()

pairs = []
for i in range(len(difference.columns)):
    for j in range(i + 1, len(difference.columns)):
        value = difference.iloc[i, j]
        if pd.notna(value):
            pairs.append([
                difference.columns[i],
                difference.columns[j],
                value
            ])

pairs.sort(key=lambda x: x[2], reverse=True)

print("Pearson correlation:")
print(pearson_corr.round(3))

print("\nSpearman correlation:")
print(spearman_corr.round(3))

print("\nTop 3 pairs with the largest absolute difference:")
for col_a, col_b, value in pairs[:3]:
    print(f"{col_a} <--> {col_b}: {value:.4f}")


# Task 9:c. Grouped aggregation: 
print("\nTask 9c: Grouped Aggregation")

# Create a numeric target: No = 0, Yes = 1
df["Churn_Encoded"] = df["Churn"].astype(str).map({
    "No": 0,
    "Yes": 1
})

# Verify the conversion before aggregation
print("\nChurn encoding check:")
print(df[["Churn", "Churn_Encoded"]].head())
print("Encoded column dtype:", df["Churn_Encoded"].dtype)
print("Missing encoded values:", df["Churn_Encoded"].isna().sum())

# Ensure the columns used for numeric calculations are numeric
df["MonthlyCharges"] = pd.to_numeric(
    df["MonthlyCharges"], errors="coerce"
)
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

# Aggregate by contract
grouped = df.groupby("Contract", observed=True)

customer_count = grouped["Churn_Encoded"].count().rename("customers")
churn_rate = grouped["Churn_Encoded"].mean().rename("churn_rate")
avg_monthly = grouped["MonthlyCharges"].mean().rename("avg_monthly_charges")
avg_tenure = grouped["tenure"].mean().rename("avg_tenure")

result = pd.concat(
    [customer_count, churn_rate, avg_monthly, avg_tenure],
    axis=1
)

result["churn_rate_pct"] = result["churn_rate"] * 100

print("\nChurn Analysis by Contract Type:")
print(result.round(2))

if not result.empty and result["churn_rate"].notna().any():
    highest = result["churn_rate"].idxmax()
    lowest = result["churn_rate"].idxmin()

    print("\nHighest churn contract:", highest)
    print(f"Churn rate: {result.loc[highest, 'churn_rate_pct']:.2f}%")

    print("\nLowest churn contract:", lowest)
    print(f"Churn rate: {result.loc[lowest, 'churn_rate_pct']:.2f}%")
else:
    print("No valid churn rates found. Check the Churn column values.")
# Task 10:Save cleaned dataset 

print("\nTask 10: Save Cleaned Dataset")

os.makedirs(BASE_PATH, exist_ok=True)

# Save a cleaned EDA copy, including TotalCharges converted to numeric.
# Keep the original Churn column as Yes/No for readability.
eda_df = df.drop(columns=["Churn_Encoded"], errors="ignore")
eda_df.to_csv(f"{BASE_PATH}/cleaned_data.csv", index=False)

print("cleaned_data.csv saved successfully.")
