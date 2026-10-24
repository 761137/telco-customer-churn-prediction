# Telco Customer Churn Prediction

## Project Overview

Customer churn is a major challenge for telecommunications companies because losing customers can impact revenue and business growth. This project focuses on exploring and preparing the Telco Customer Churn dataset as the first step toward building a machine learning model to predict customer churn.

The project includes data acquisition, data quality checks, data type conversion, exploratory data analysis (EDA), statistical analysis, data visualization, and data cleaning.

## Business Objective

The main objective is to understand customer characteristics and identify patterns associated with churn. These insights will support the development of a machine learning model that can predict customers who are likely to leave the company.

## Dataset Description

- *Dataset:* Telco Customer Churn
- *File name:* Telco-Customer-Churn.csv
- *Total records:* 7,043
- *Total columns:* 21
- *Target variable:* Churn

The dataset contains customer demographics, subscribed services, contract details, billing information, tenure, and churn status.

### Key Features

| Feature | Description |
|---|---|
| customerID | Unique customer identifier |
| gender | Customer gender |
| SeniorCitizen | Indicates whether the customer is a senior citizen |
| Partner | Whether the customer has a partner |
| Dependents | Whether the customer has dependents |
| tenure | Number of months the customer has stayed |
| PhoneService | Whether phone service is subscribed |
| InternetService | Type of internet service |
| Contract | Customer contract type |
| MonthlyCharges | Monthly amount charged |
| TotalCharges | Total amount charged |
| Churn | Whether the customer left the company |

## Technologies Used

- Python 3.9
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook / Python scripts
- Git and GitHub

## Project Structure

text
Part1_DataAcquisition_Cleaning_EDA/
├── Telco-Customer-Churn.csv
├── cleaned_data.csv
├── Part1_DataPrep_EDA.py
├── README.md
└── plots/
    ├── line_plot.png
    ├── bar_chart.png
    ├── histogram.png
    ├── scatterplot.png
    ├── boxplot.png
    └── correlation_heatmap.png


## Project Workflow

### Task 1: Data Acquisition

- Loaded the Telco Customer Churn dataset using Pandas.
- Examined the dataset structure and dimensions.
- Reviewed column names and sample records.

### Task 2: Data Quality Assessment

- Checked missing values across columns.
- Identified and reviewed data quality issues.
- Found 11 missing values in TotalCharges, approximately 0.16% of the dataset.
- Checked for duplicate rows and duplicate customer IDs.
- No duplicate rows or duplicate customer IDs were found.

### Task 3: Data Type Conversion

- Reviewed column data types.
- Converted TotalCharges from object to numeric format.
- Set Churn as a categorical variable.
- Ensured columns used appropriate data types for analysis.

### Task 4: Memory Optimization

- Compared DataFrame memory usage before and after optimization.
- Converted suitable columns to more memory-efficient data types.
- Reduced memory usage from 7,779,973 bytes to 814,638 bytes.
- Achieved approximately 89.5% memory reduction.

### Task 5: Statistical Analysis

- Calculated descriptive statistics for numerical variables.
- Examined the distribution and skewness of numerical features.
- Identified skewness in SeniorCitizen (1.83), TotalCharges (0.96), tenure (0.24), and MonthlyCharges (-0.22).
- Reviewed the distribution of the target variable.

### Task 6: Outlier Detection

- Used the Interquartile Range (IQR) method to examine numerical features.
- Checked tenure, MonthlyCharges, and TotalCharges.
- No IQR-based outliers were identified in these three columns.

### Task 7: Data Visualization

Generated visualizations to understand the dataset and relationships between variables:

- Line plot
- Bar chart
- Histogram
- Scatter plot
- Box plot
- Correlation heatmap

These visualizations support exploratory analysis and help identify patterns in customer data.

### Task 8: Correlation Analysis

- Examined correlations among numerical variables.
- Compared numerical features with the encoded churn target.
- Found that tenure had the strongest numeric correlation with encoded churn, at approximately -0.352.
- Found that tenure and TotalCharges had the strongest feature-to-feature correlation, at approximately 0.826.

The negative relationship between tenure and encoded churn indicates that customers with longer tenure tend to be less likely to churn.

### Task 9: Advanced Exploratory Data Analysis

#### 9a. Missing Value Imputation

- Reviewed missing values in numerical columns.
- Used median imputation for missing TotalCharges values.
- Verified that missing values were resolved after imputation.

#### 9b. Pearson and Spearman Correlation

- Compared Pearson and Spearman correlation coefficients for selected numerical feature pairs.
- Examined differences between linear and rank-based relationships.
- The largest observed absolute difference was approximately 0.0615 for tenure and TotalCharges.

#### 9c. Contract-Based Churn Analysis

Analyzed churn rate, customer count, average monthly charges, and average tenure by contract type.

| Contract Type | Customers | Churn Rate | Avg. Monthly Charges | Avg. Tenure |
|---|---:|---:|---:|---:|
| Month-to-month | 3,875 | 42.71% | 66.40 | 18.04 |
| One year | 1,473 | 11.27% | 65.05 | 42.04 |
| Two year | 1,695 | 2.83% | 60.77 | 56.74 |

*Key finding:* Month-to-month customers have the highest churn rate (42.71%), while two-year contract customers have the lowest churn rate (2.83%).

This suggests that contract type is an important variable to investigate when developing a churn prediction model. These are exploratory associations and do not, by themselves, establish causation.

### Task 10: Export Cleaned Dataset

- Saved the prepared dataset to cleaned_data.csv.
- Verified that the output file was generated successfully.
- Prepared the dataset for the next stage of machine learning development.

## Key Findings

1. The dataset contains 7,043 customer records and 21 columns.
2. Only 11 missing values were identified in TotalCharges.
3. No duplicate rows or duplicate customer IDs were found.
4. DataFrame memory usage was reduced by approximately 89.5%.
5. The churn distribution is imbalanced:
   - No churn: 5,174 customers (73.46%)
   - Churn: 1,869 customers (26.54%)
6. tenure has a negative correlation with encoded churn.
7. Month-to-month contracts have a substantially higher churn rate than one-year and two-year contracts.
8. Contract type, tenure, and billing information are useful candidates for further investigation in predictive modeling.

## How to Run the Project

### 1. Clone the Repository

bash
git clone <your-github-repository-url>


### 2. Navigate to the Project Directory

bash
cd <your-repository-folder>


### 3. Install Dependencies

bash
pip install pandas numpy matplotlib seaborn


### 4. Run the Python Script

Run the following command from the repository root:

bash
python Part1_DataAcquisition_Cleaning_EDA/Part1_DataPrep_EDA.py


Make sure the dataset path in the script matches the location of Telco-Customer-Churn.csv.

## Future Enhancements

- Perform feature engineering and categorical encoding.
- Split the dataset into training and testing sets.
- Build a baseline classification model.
- Train models such as Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
- Evaluate models using precision, recall, F1-score, ROC-AUC, and a confusion matrix.
- Address class imbalance where appropriate.
- Tune model hyperparameters using cross-validation.
- Identify the most important churn predictors.
- Develop a simple interface or API for churn prediction.

For predictive modeling, fit imputation and other preprocessing steps using the training data only, then apply the fitted transformations to validation and test data to avoid data leakage.

## Conclusion

This project establishes the data preparation and exploratory analysis foundation for a Telco Customer Churn Prediction solution. The analysis highlights differences in churn across contract types and identifies tenure as a useful variable for further investigation.

The next phase will focus on building, evaluating, and improving machine learning classification models to identify customers at risk of churn.