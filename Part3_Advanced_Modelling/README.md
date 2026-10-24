# Part 3: Advanced Modelling – Tree-Based Models and Model Optimization

## Project Overview

This project is the third part of the Telco Customer Churn Prediction Capstone Project. The objective is to build, evaluate, compare, and optimize machine learning models to predict whether a telecommunications customer is likely to leave the company.

Building on the data preparation and exploratory data analysis completed in Part 1 and the baseline Logistic Regression model developed in Part 2, this part explores tree-based classification models, feature importance, feature ablation, cross-validation, hyperparameter tuning, learning curves, and model serialization.

The goal is to identify useful customer churn predictors and evaluate model performance using appropriate classification metrics, particularly ROC-AUC.

## Dataset

- *Dataset:* Telco Customer Churn Dataset
- *Source file:* Part1_DataAcquisition_Cleaning_EDA/cleaned_data.csv
- *Total records:* 7,043
- *Original columns:* 21
- *Encoded feature columns:* 30
- *Target variable:* Churn
- *Target classes:* Yes (Churn) and No (No Churn)

The customerID column was excluded from the feature matrix because it is an identifier rather than a meaningful predictive feature.

The target variable was converted into binary values:

- No → 0
- Yes → 1

### Feature Categories

*Numerical features (4):*
- SeniorCitizen
- tenure
- MonthlyCharges
- TotalCharges

*Categorical features (15):*
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

Categorical variables were converted into numerical columns using one-hot encoding with drop_first=True.

The final feature matrix contained 30 columns.

## Technologies Used

- Python 3.9
- Pandas
- Scikit-learn
- NumPy
- Joblib
- StandardScaler
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression
- Cross-validation and GridSearchCV

## Project Structure

text
Capstone_Project1/
│
├── Part1_DataAcquisition_Cleaning_EDA/
│   └── cleaned_data.csv
│
├── Part2_Supervised_MachineLearning/
│   ├── Part2_ML.py
│   └── README.md
│
└── Part3_Advanced_Modelling/
    ├── Part3_TreeBasedModels.py
    ├── best_model.pkl
    └── README.md


The saved model file best_model.pkl contains the best Random Forest pipeline identified during hyperparameter tuning.

# Task Summary

## Task 1: Baseline Decision Tree

A baseline Decision Tree Classifier was trained without limiting its maximum depth.

*Results:*

| Metric | Score |
|---|---:|
| Training Accuracy | 99.80% |
| Testing Accuracy | 74.10% |

The model achieved nearly perfect training accuracy but lower testing accuracy. This indicates overfitting: the tree learned the training data very closely but did not generalize as well to unseen customers.

## Task 2: Controlled Decision Tree

A controlled Decision Tree was trained using the following parameters:

- max_depth = 5
- min_samples_split = 20
- random_state = 42

*Results:*

| Metric | Score |
|---|---:|
| Training Accuracy | 80.10% |
| Testing Accuracy | 79.42% |

Limiting the tree depth reduced training accuracy but improved testing accuracy compared with the baseline tree.

This demonstrates how controlling model complexity can help reduce overfitting and improve generalization.

## Task 3: Gini vs Entropy Comparison

Two Decision Tree models were compared using different split criteria.

- *Gini:* Measures the impurity of the groups created by a split.
- *Entropy:* Measures the uncertainty or disorder in the groups created by a split.

Both models used max_depth = 5.

*Results:*

| Criterion | Test Accuracy | ROC-AUC |
|---|---:|---:|
| Gini | 79.42% | 0.8284 |
| Entropy | 79.21% | 0.8273 |

The Gini-based Decision Tree performed slightly better than the Entropy-based tree on both test accuracy and ROC-AUC.

The difference was small, so both criteria produced broadly similar results.

## Task 4: Random Forest

A Random Forest Classifier was trained using the following parameters:

- n_estimators = 100
- max_depth = 10
- random_state = 42

Random Forest combines multiple decision trees to improve predictive performance and reduce reliance on a single tree.

*Results:*

| Metric | Score |
|---|---:|
| Training Accuracy | 86.47% |
| Testing Accuracy | 80.70% |
| ROC-AUC | 0.8427 |

The Random Forest achieved a test accuracy of 80.70% and a ROC-AUC of 0.8427.

Its training accuracy was higher than its testing accuracy, but the gap was smaller than that of the unrestricted baseline Decision Tree.

### Top 5 Important Features

The feature importance scores from the Random Forest were:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | tenure | 0.188226 |
| 2 | TotalCharges | 0.157077 |
| 3 | MonthlyCharges | 0.121407 |
| 4 | InternetService_Fiber optic | 0.068687 |
| 5 | PaymentMethod_Electronic check | 0.061691 |

The most important features were tenure, TotalCharges, and MonthlyCharges.

These results suggest that customer relationship duration and billing-related characteristics are useful predictors for the fitted Random Forest model.

Feature importance indicates predictive usefulness within this model; it does not establish that a feature causes churn.

## Task 4a: Gradient Boosting

A Gradient Boosting Classifier was trained using the following parameters:

- n_estimators = 100
- learning_rate = 0.1
- max_depth = 3
- random_state = 42

Gradient Boosting builds trees sequentially, with each new tree helping improve the predictions of the existing ensemble.

*Results:*

| Metric | Score |
|---|---:|
| Training Accuracy | 82.71% |
| Testing Accuracy | 79.84% |
| ROC-AUC | 0.8421 |

The Gradient Boosting model achieved a ROC-AUC of 0.8421, which was close to the Random Forest result of 0.8427.

The models demonstrated similar discrimination performance on the test dataset.

## Task 4b: Feature Ablation Study

A feature ablation study was performed to examine the effect of removing five features with the lowest Random Forest feature importance scores.

*Features removed:*

- MultipleLines_No phone service
- PhoneService_Yes
- StreamingMovies_No internet service
- InternetService_No
- StreamingTV_No internet service

*Results:*

| Model | ROC-AUC |
|---|---:|
| Full Random Forest | 0.8427 |
| Reduced Random Forest | 0.8410 |

Removing the five least important features resulted in a small decrease in ROC-AUC from 0.8427 to 0.8410.

This suggests that the removed features contributed relatively little to the model's predictive performance, although their removal did not improve the result in this experiment.

## Task 5: Cross-Validated Model Comparison

Stratified 5-fold cross-validation was used to compare four classification models.

The evaluation metric was ROC-AUC.

*Results:*

| Model | Mean ROC-AUC | Standard Deviation |
|---|---:|---:|
| Logistic Regression | 0.8451 | 0.0133 |
| Decision Tree | 0.8258 | 0.0125 |
| Random Forest | 0.8409 | 0.0118 |
| Gradient Boosting | 0.8469 | 0.0111 |

Gradient Boosting achieved the highest mean cross-validated ROC-AUC of 0.8469, followed closely by Logistic Regression at 0.8451.

The controlled Decision Tree had the lowest mean ROC-AUC among the four models.

The standard deviations show the variation in ROC-AUC across the five folds. The differences in mean scores are relatively small, so they should not be interpreted as proof that one model is definitively superior.

## Task 6: Hyperparameter Tuning with GridSearchCV

GridSearchCV was used to identify a suitable Random Forest configuration.

The following hyperparameters were tested:

| Hyperparameter | Values Tested |
|---|---|
| n_estimators | 50, 100, 200 |
| max_depth | 5, 10, None |
| min_samples_leaf | 1, 5 |

Stratified 5-fold cross-validation and ROC-AUC scoring were used for model selection.

*Best parameters identified:*

python
{
    "randomforestclassifier__max_depth": 10,
    "randomforestclassifier__min_samples_leaf": 5,
    "randomforestclassifier__n_estimators": 200
}


*Results:*

| Metric | ROC-AUC |
|---|---:|
| Best Cross-Validated ROC-AUC | 0.8448 |
| Tuned Model Test ROC-AUC | 0.8432 |

The tuned Random Forest achieved a test ROC-AUC of 0.8432.

The tuned model's test ROC-AUC was close to its best cross-validated score, suggesting reasonably consistent ranking performance across the validation folds and held-out test set.

## Task 7: Manual Learning Curve

A manual learning curve was created by training the tuned Random Forest on different fractions of the training dataset.

The training fractions were 20%, 40%, 60%, 80%, and 100%.

*Results:*

| Training Fraction | Training ROC-AUC | Test ROC-AUC |
|---|---:|---:|
| 20% | 0.9424 | 0.8354 |
| 40% | 0.9333 | 0.8439 |
| 60% | 0.9262 | 0.8425 |
| 80% | 0.9186 | 0.8449 |
| 100% | 0.9119 | 0.8432 |

Training ROC-AUC decreased from 0.9424 at 20% of the training data to 0.9119 at 100%. Test ROC-AUC remained relatively stable, ranging from 0.8354 to 0.8449 across the training fractions.

These results suggest that the model’s ranking performance on the held-out test set remained consistent as the training subset size increased. However, training ROC-AUC remained higher than test ROC-AUC, indicat
ing some remaining overfitting.

For this experiment, each training subset was created by selecting the first specified number of rows from the training dataset, rather than randomly sampling rows for each fraction. Therefore, the results should be interpreted as an exploratory comparison rather than a definitive learning curve. The same held-out test set was used for every training fraction.

## Task 8: Model Serialization

The tuned Random Forest pipeline was saved using Joblib.

python
joblib.dump(best_model, "best_model.pkl")


The saved model was successfully loaded again:

python
loaded_model = joblib.load("best_model.pkl")


Two test customer records were passed to the loaded model to demonstrate prediction.

*Sample predictions:*

| Customer | Predicted Churn | Churn Probability |
|---|---|---:|
| Customer 1 | No | 0.0227 |
| Customer 2 | Yes | 0.7236 |

The first sample customer was predicted not to churn, with a churn probability of 2.27%.

The second sample customer was predicted to churn, with a churn probability of 72.36%.

These are two example predictions only and do not represent the overall model performance.

# How to Run

### 1. Install the required libraries

bash
pip install pandas scikit-learn joblib


### 2. Navigate to the project root directory

Make sure the terminal is in the Capstone_Project1 directory.

### 3. Run the Part 3 script

bash
python Part3_Advanced_Modelling/Part3_TreeBasedModels.py


### 4. Check the output

The script prints the training and testing accuracy, ROC-AUC scores, feature importance rankings, cross-validation results, hyperparameter tuning results, learning curve results, and sample predictions.

The script also saves the tuned model as best_model.pkl.

# Final Conclusion

Part 3 explored several tree-based classification models and compared them with Logistic Regression using stratified cross-validation.

The unrestricted Decision Tree showed substantial overfitting, while controlling its depth improved test accuracy. Random Forest achieved a test accuracy of 80.70% and a ROC-AUC of 0.8427. Gradient Boosting achieved a ROC-AUC of 0.8421 on the test set and the highest mean cross-validated ROC-AUC of 0.8469 among the four models compared.

Hyperparameter tuning identified a Random Forest with 200 trees, a maximum depth of 10, and a minimum leaf size of 5. This tuned model achieved a test ROC-AUC of 0.8432.

The feature importance analysis highlighted tenure, TotalCharges, MonthlyCharges, internet service type, and payment method as useful predictive features in the Random Forest model. The feature ablation experiment showed that removing the five least important features caused only a small reduction in ROC-AUC.

Finally, the tuned Random Forest pipeline was successfully saved and reloaded, demonstrating that the trained model can be reused for future customer churn predictions.

Overall, this part demonstrates the importance of comparing models, controlling overfitting, tuning hyperparameters, and evaluating predictive performance before selecting a model for a customer retention application.