# Part 2: Supervised Machine Learning – Build, Train, and Evaluate

## Project Overview

This project builds supervised machine learning models using the cleaned Telco Customer Churn dataset generated in Part 1.

The objective is to predict whether a customer is likely to leave the telecommunications service based on customer demographics, subscribed services, contract information, and billing details.

A *Logistic Regression* model was developed to classify customers into churn and non-churn categories.

The project also demonstrates proper preprocessing, categorical feature encoding, leak-free scaling, classification evaluation, decision threshold tuning, regularization, and bootstrap confidence interval estimation.

The goal is to develop a reliable classification workflow and understand how machine learning can help identify customers who are likely to churn and support customer retention strategies.

---

## Dataset

Dataset: cleaned_data.csv

Dataset Path: Part1_DataAcquisition_Cleaning_EDA/cleaned_data.csv

The cleaned dataset contains *7,043 records and 21 columns*.

The target variable is Churn, which indicates whether a customer has discontinued the service.

- No – Customer has not churned.
- Yes – Customer has churned.

The target variable was converted into binary numerical values for classification.

- 0 – No Churn
- 1 – Churn

### Target Distribution

| Churn Status | Number of Customers | Percentage |
|---|---:|---:|
| No Churn | 5,174 | 73.46% |
| Churn | 1,869 | 26.54% |
| *Total* | *7,043* | *100%* |

The dataset is imbalanced because non-churn customers represent approximately 73.46% of the records, while churn customers represent 26.54%.

Therefore, accuracy alone is not sufficient to evaluate model performance. Precision, recall, F1-score, and ROC-AUC are also used.

---

## Technologies Used

- Python 3.9
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Logistic Regression
- One-Hot Encoding
- StandardScaler
- SimpleImputer
- ColumnTransformer
- Train-Test Split
- Classification Metrics
- ROC Curve Analysis
- Decision Threshold Analysis
- Regularization
- Bootstrap Resampling

---

## Project Structure

text
Capstone_Project1/
│
├── Part1_DataAcquisition_Cleaning_EDA/
│   ├── cleaned_data.csv
│   └── ...
│
├── Part2_Supervised_MachineLearning/
│   ├── Part2_ML.py
│   ├── README.md
│   └── plots/
│       └── roc_curve.png
│
└── ...


---

# Task Summary

## Task 1: Feature Matrix and Target Definition

The cleaned dataset produced in Part 1 was loaded successfully for supervised machine learning.

### Feature Matrix (X)

The feature matrix contains customer attributes used to predict churn.

The customerID column was excluded because it is a unique customer identifier and does not provide meaningful predictive information.

The remaining features include demographic information, subscribed services, contract details, and billing information.

### Classification Label (y)

The classification target is:

Churn

The target variable was converted into binary numerical values:

python
df["Churn"].map({"No": 0, "Yes": 1})


where:

- 1 = Customer has churned.
- 0 = Customer has not churned.

This binary representation allows Logistic Regression to estimate the probability that a customer belongs to the churn class.

The dataset contains 7,043 customer records, with 5,174 non-churn customers and 1,869 churn customers.

---

## Task 2: Feature Encoding

The dataset contains both numerical and categorical variables. These features require different preprocessing methods before they can be used by the machine learning model.

A preprocessing pipeline was implemented using Scikit-learn to handle missing values, encode categorical variables, and scale numerical features.

### Numerical Features

The following numerical features were used:

- SeniorCitizen
- tenure
- MonthlyCharges
- TotalCharges

Numerical preprocessing included:

- Missing-value imputation using the median.
- Feature scaling using StandardScaler.

Median imputation was used to handle missing numerical values, while standardization ensured that numerical features were placed on comparable scales.

### Categorical Features

The following categorical features were used:

- gender
- Partner
- Dependents
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod

Categorical preprocessing included:

- Missing-value imputation using the most frequent value.
- One-Hot Encoding to convert categorical values into numerical features.
- handle_unknown="ignore" to handle previously unseen categories.
- drop="first" to reduce redundant encoded columns.

A ColumnTransformer was used to apply the appropriate transformations to numerical and categorical features.

After preprocessing and encoding, the dataset contained *30 input features*.

### Why One-Hot Encoding?

One-Hot Encoding represents categorical values as separate binary features instead of assigning arbitrary numerical values that might imply an incorrect ordering between categories.

For example, categories such as Contract types and PaymentMethod values do not have a natural numerical order.

Using One-Hot Encoding allows the model to learn relationships between these categories and customer churn without introducing false ordinal relationships.

---

## Task 3: Train-Test Split and Feature Scaling

The dataset was divided into training and testing sets using an 80:20 split.

python
train_test_split(
    test_size=0.20,
    random_state=42,
    stratify=y
)


### Dataset Split

| Dataset | Number of Samples | Percentage |
|---|---:|---:|
| Training Set | 5,634 | 80% |
| Testing Set | 1,409 | 20% |
| *Total* | *7,043* | *100%* |

A stratified split was used to preserve the approximate distribution of churn and non-churn customers in both datasets.

### Feature Scaling

StandardScaler was applied to numerical input features.

The scaler was fitted only on the training data and then used to transform both the training and testing data.

The preprocessing pipeline was also fitted only on the training data before transforming the test set.

### Why Was the Scaler Fitted Only on the Training Data?

Fitting the scaler on the entire dataset would introduce data leakage because information from the test data would influence the scaling parameters used during training.

This could lead to overly optimistic evaluation results.

Fitting the preprocessing pipeline only on the training data ensures that the model is evaluated on data that did not influence the training process.

The processed training and testing datasets contained 30 encoded features each.

---

## Task 4: Logistic Regression

A Logistic Regression model was trained using the preprocessed training data.

Logistic Regression estimates the probability that a customer will churn based on the available customer features.

The baseline model was configured using:

python
LogisticRegression(max_iter=2000)


The default classification threshold of 0.50 was used for the baseline evaluation.

### Model Performance

The baseline Logistic Regression model achieved the following results on the test dataset.

| Metric | Value |
|---|---:|
| Accuracy | 0.8055 |
| Precision | 0.6572 |
| Recall | 0.5588 |
| F1-Score | 0.6040 |
| ROC-AUC | 0.8420 |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| No Churn | 0.85 | 0.89 | 0.87 | 1,035 |
| Churn | 0.66 | 0.56 | 0.60 | 374 |

### Confusion Matrix

The confusion matrix at the default decision threshold of 0.50 was:

text
[[926 109]
 [165 209]]


The confusion matrix can also be represented as follows:

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 926 | 109 |
| Actual Churn | 165 | 209 |

The model correctly classified 926 customers who did not churn and 209 customers who churned.

However, 165 customers who actually churned were incorrectly classified as non-churn customers. These false negatives represent potential churn customers who might be missed by a customer retention strategy.

### Precision and Recall

Precision is defined as:

Precision = TP / (TP + FP)

Recall is defined as:

Recall = TP / (TP + FN)

where:

- TP = True Positives
- FP = False Positives
- FN = False Negatives

For this classification task, recall is particularly important because failing to identify customers who are likely to churn can prevent the business from taking timely retention measures.

Precision is also important because incorrectly identifying customers as likely to churn may lead to unnecessary retention campaigns.

### ROC Curve

The Receiver Operating Characteristic (ROC) curve illustrates the trade-off between the True Positive Rate and False Positive Rate across different classification thresholds.

The baseline Logistic Regression model achieved a *ROC-AUC score of 0.8420*, indicating good ability to distinguish between churn and non-churn customers.

The ROC curve was saved to:

Part2_Supervised_MachineLearning/plots/roc_curve.png

---

## Task 5: ROC Curve and Classification Threshold Analysis

The ROC-AUC metric measures the model's ability to distinguish between churn and non-churn customers across different classification thresholds.

Unlike accuracy, ROC-AUC evaluates how well the model ranks customers across a range of possible thresholds.

The baseline Logistic Regression model achieved a ROC-AUC score of *0.8420*.

This indicates that the model has good discriminative ability when distinguishing between customers who churn and customers who remain.

The ROC curve was generated and saved as an image file for visualization and further analysis.

The generated plot is available at:

plots/roc_curve.png

---

## Task 5b: Decision Threshold Sensitivity

By default, Logistic Regression classifies a customer as a churn customer when the predicted probability is at least 0.50.

However, the classification threshold can be adjusted depending on the business objective.

For customer churn prediction, identifying more customers who are likely to leave may help a business take proactive retention measures.

A lower threshold generally increases recall because more customers are classified as potential churn customers. However, this can also increase the number of false-positive predictions.

The following decision thresholds were evaluated:

### Threshold Comparison

| Threshold | Precision | Recall | F1-Score |
|---:|---:|---:|---:|
| 0.30 | 0.52 | 0.75 | 0.62 |
| 0.40 | 0.57 | 0.67 | 0.62 |
| 0.50 | 0.66 | 0.56 | 0.60 |
| 0.60 | 0.72 | 0.40 | 0.51 |
| 0.70 | 0.74 | 0.18 | 0.29 |

### Best Threshold

The best tested threshold by the unrounded F1-score was *0.40, with an F1-score of approximately **0.6158*.

The threshold of 0.30 achieved a similar rounded F1-score and a higher recall.

### Interpretation

- *Threshold 0.30:* Achieved a recall of 0.75, identifying a larger proportion of customers who churned.
- *Threshold 0.40:* Achieved a recall of 0.67 and a precision of 0.57.
- *Threshold 0.50:* Achieved a precision of 0.66 and a recall of 0.56.
- *Threshold 0.60:* Increased precision to 0.72 but reduced recall to 0.40.
- *Threshold 0.70:* Achieved precision of 0.74, but recall dropped to 0.18.

Lowering the threshold increases recall because more customers are classified as potential churn customers.

Increasing the threshold improves precision because the model requires a higher predicted probability before classifying a customer as likely to churn.

The trade-off is that improving precision often reduces recall, while improving recall can increase false-positive predictions.

Since identifying potential churn customers is important for customer retention, a lower decision threshold may be appropriate when the cost of missing a customer who will churn is high.

The final threshold should be selected according to the business objective and the relative costs of false-positive and false-negative predictions.

---

## Task 6: Logistic Regression Regularization

A second Logistic Regression model was trained using stronger regularization.

The regularized model used:

python
LogisticRegression(
    C=0.01,
    max_iter=2000
)


The parameter C controls the inverse strength of regularization.

- Smaller C → Stronger regularization.
- Larger C → Weaker regularization.

Stronger regularization penalizes large model coefficients and can help control model complexity.

### Model Comparison

| Metric | Baseline Logistic Regression (C = 1.0) | Regularized Logistic Regression (C = 0.01) |
|---|---:|---:|
| Accuracy | 0.8055 | 0.7949 |
| Precision | 0.6572 | 0.6568 |
| Recall | 0.5588 | 0.4759 |
| F1-Score | 0.6040 | 0.5519 |
| ROC-AUC | 0.8420 | 0.8380 |

### Interpretation

The baseline Logistic Regression model achieved higher accuracy, recall, F1-score, and ROC-AUC than the regularized model.

Precision was nearly identical between the two models.

Reducing C to 0.01 applied stronger regularization but did not improve the observed test-set performance in this experiment.

The comparison demonstrates that regularization does not necessarily improve every evaluation metric and that its effect should be assessed using the specific objectives of the classification task.

Bootstrap confidence interval estimation was performed to examine the uncertainty in the difference between the two models' ROC-AUC scores.

---

## Task 7: Bootstrap Confidence Interval

A bootstrap procedure was performed to estimate the uncertainty in the ROC-AUC difference between the baseline and regularized Logistic Regression models.

The analysis used *1,000 valid bootstrap samples*.

The ROC-AUC difference was defined as:

Baseline ROC-AUC - Regularized ROC-AUC

A positive difference indicates that the baseline model achieved a higher ROC-AUC, while a negative difference indicates that the regularized model achieved a higher ROC-AUC.

### Bootstrap Results

| Statistic | Value |
|---|---:|
| Number of Valid Bootstrap Samples | 1,000 |
| Mean ROC-AUC Difference | 0.0039 |
| Lower Confidence Bound | -0.0002 |
| Upper Confidence Bound | 0.0081 |

### Interpretation

The mean ROC-AUC difference was approximately *0.0039* in favor of the baseline Logistic Regression model.

The confidence interval ranged from *-0.0002 to 0.0081* and included zero.

Because the confidence interval includes zero, the analysis does not provide sufficient evidence to conclude that the baseline model has a statistically significant ROC-AUC advantage over the regularized model at the confidence level represented by the interval.

Although the baseline model achieved a slightly higher ROC-AUC on the observed test dataset, the difference is small relative to the uncertainty estimated through bootstrap resampling.

### Conclusion

Bootstrap resampling provides additional information about the uncertainty of model comparisons beyond a single test-set performance score.

The baseline model performed better on the observed test data, but the confidence interval indicates that the ROC-AUC difference is not clearly distinguishable from zero.

Further evaluation, hyperparameter tuning, and validation on additional data could help determine whether the observed difference is consistent.

---

# How to Run

### Step 1: Navigate to the Project Directory

Open a terminal or command prompt and navigate to the project directory:

bash
cd "C:\Users\Mahi\Documents\AI_Projects\Capstone_Project1"


### Step 2: Install Required Libraries

Install the required Python libraries:

bash
pip install pandas numpy matplotlib scikit-learn


### Step 3: Verify the Dataset

Ensure that the cleaned dataset generated in Part 1 exists at:

text
Part1_DataAcquisition_Cleaning_EDA/cleaned_data.csv


### Step 4: Navigate to the Part 2 Directory

bash
cd Part2_Supervised_MachineLearning


### Step 5: Run the Machine Learning Script

Execute the Python script:

bash
python Part2_ML.py


The script will:

- Load the cleaned dataset.
- Inspect the dataset shape and target distribution.
- Identify numerical and categorical features.
- Separate the feature matrix and target variable.
- Perform a stratified train-test split.
- Impute missing values.
- Encode categorical features.
- Scale numerical features without data leakage.
- Train the baseline Logistic Regression model.
- Evaluate accuracy, precision, recall, F1-score, and ROC-AUC.
- Generate the classification report and confusion matrix.
- Plot and save the ROC curve.
- Perform decision threshold sensitivity analysis.
- Train and evaluate the regularized Logistic Regression model.
- Estimate the bootstrap confidence interval for the ROC-AUC difference.

The ROC curve image is saved at:

text
plots/roc_curve.png


---

## Final Conclusion

This project implemented a complete supervised machine learning workflow for predicting customer churn using the Telco Customer Churn dataset.

The baseline Logistic Regression model achieved an accuracy of *80.55%, a precision of **0.6572, a recall of **0.5588, an F1-score of **0.6040, and a ROC-AUC of **0.8420*.

Decision threshold analysis demonstrated that lowering the threshold can improve recall and help identify more customers who are likely to churn. However, this involves a trade-off with precision and the number of false-positive predictions.

Regularization was evaluated using a Logistic Regression model with C = 0.01. The regularized model did not improve the observed test-set performance compared with the baseline model.

Bootstrap resampling was used to estimate uncertainty in the ROC-AUC difference. The confidence interval included zero, so the analysis did not establish a statistically significant difference between the models' ROC-AUC scores.

Overall, this project demonstrates data preprocessing, categorical feature encoding, train-test splitting, Logistic Regression, classification evaluation, ROC curve analysis, decision threshold tuning, regularization, and bootstrap confidence interval estimation.

The results provide a foundation for further experimentation with alternative classification algorithms, hyperparameter tuning, and business-focused evaluation of customer retention strategies.