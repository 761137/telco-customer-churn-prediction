# Part 4: LLM-Powered Customer Churn Explanations

## Project Overview

This project is the fourth part of the Telco Customer Churn Prediction Capstone Project. The objective is to integrate a Large Language Model (LLM) with the trained machine learning model developed in Part 3 to generate understandable explanations for customer churn predictions.

Building on the data preparation and exploratory data analysis completed in Part 1, the baseline Logistic Regression model developed in Part 2, and the advanced modelling and optimization completed in Part 3, this part focuses on LLM integration, prediction explanations, PII protection, structured JSON responses, and customer-retention recommendations.

The trained Random Forest pipeline predicts whether a telecommunications customer is likely to churn. The LLM uses the model prediction, confidence score, and customer attributes to generate a human-readable explanation and suggest a possible next step.

The implementation also includes feature-schema validation, API error handling, retry logic, JSON schema validation, and error logging.

## Dataset

- Dataset: Telco Customer Churn Dataset
- Source file: Part1_DataAcquisition_Cleaning_EDA/cleaned_data.csv
- Total records: 7,043
- Raw feature columns: 19
- Encoded feature columns: 30
- Target variable: Churn
- Target classes: Yes (Churn) and No (No Churn)
- Trained model: Part3_Advanced_Modelling/best_model.pkl
- Feature schema: Part3_Advanced_Modelling/feature_columns.pkl

The customerID column was excluded from the feature matrix because it is an identifier rather than a meaningful predictive feature.

The target variable was converted into binary values:

- No → 0
- Yes → 1

Categorical variables were converted into numerical columns using one-hot encoding with drop_first=True.

The prepared feature matrix was aligned with the saved feature schema from Part 3 to ensure that the input columns and their order matched the trained model.

## Technologies Used

- Python 3.9
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Requests
- Python-dotenv
- jsonschema
- Regular Expressions
- Logging
- Groq API
- Large Language Model (LLM)
- JSON

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
├── Part3_Advanced_Modelling/
│   ├── Part3_TreeBasedModels.py
│   ├── best_model.pkl
│   └── feature_columns.pkl
│
└── Part4_LLM_Powered_feature/
    ├── Part4_LLM_Powered_feature.py
    ├── .env
    ├── Part4_errors.log
    └── README.md


The saved model file best_model.pkl contains the trained Random Forest pipeline selected during hyperparameter tuning in Part 3.

The feature_columns.pkl file contains the saved feature schema used to prepare customer data consistently with the training process.

The .env file stores the Groq API key, while Part4_errors.log records errors encountered during execution.

# Task Summary

## Task 1: Check Required Files and Load the Model

The script checks whether the cleaned dataset, trained model, and saved feature schema exist before loading them.

The trained model and feature columns are loaded using Joblib. Validation checks confirm that the model supports prediction, the feature schema is stored as a list, and the saved columns do not match an unrelated BigMart dataset schema.

The script also compares the model's feature names with the saved feature columns when the model exposes feature names.

Results:

| Check | Result |
|---|---|
| Cleaned dataset file | Found |
| Trained model file | Loaded successfully |
| Feature schema file | Loaded successfully |
| Model type | Pipeline |
| Number of features | 30 |
| Feature schema validation | Successful |

The trained model and feature schema from Part 3 were loaded successfully and reused for customer churn prediction.

## Task 2: Load and Validate the Dataset

The cleaned Telco Customer Churn dataset was loaded using Pandas.

The script checks that the Churn target column exists, removes rows with missing target labels, and validates that the target labels contain only Yes and No.

The target variable was mapped to binary values, and the customerID column was removed from the feature data. The TotalCharges column was also converted to numeric format where possible.

Results:

| Metric | Result |
|---|---:|
| Total dataset rows | 7,043 |
| Raw Telco feature columns | 19 |
| Target variable | Churn |
| Customer identifier excluded | Yes |
| Dataset validation | Successful |

The dataset was loaded successfully and prepared for feature engineering and prediction.

## Task 3: Prepare Features to Match Part 3

Categorical features were converted into numerical columns using one-hot encoding with drop_first=True and dtype=int.

The encoded features were reindexed using the feature columns saved in Part 3. Missing columns were filled with zero during reindexing, and remaining missing numerical values were handled using column medians followed by zero as a fallback.

The script validates that the prepared feature names and their order match the saved schema and checks for unexpected BigMart feature names.

Results:

| Metric | Result |
|---|---|
| Raw feature columns | 19 |
| Prepared feature columns | 30 |
| Prepared feature matrix shape | (7043, 30) |
| Feature count matches saved model | True |
| Feature schema validation | Successful |

The prepared feature matrix matched the saved 30-feature schema.

This step ensures that customer data is presented to the trained model in the expected format. The saved model pipeline also retains its own imputation and scaling steps.

## Task 4: Generate Customer Churn Predictions

The trained model was used to predict whether each selected customer was likely to churn.

The script checks that the predicted class is either 0 or 1 and that the model supports predict_proba(). It then extracts the probability associated with the churn class.

The predicted churn label is determined as follows:

- 0 → No
- 1 → Yes

The predicted-class confidence is calculated from the model probabilities:

python
confidence = (
    churn_probability
    if churn_label == "Yes"
    else 1.0 - churn_probability
)


Three customer samples were selected using random_state=42.

Results:

| Customer Sample | Actual Churn | Predicted Churn | Churn Probability | Prediction Confidence |
|---|---|---|---:|---:|
| Customer 1 | Yes | Yes | 0.6501 | 0.6501 |
| Customer 2 | No | No | 0.0482 | 0.9518 |
| Customer 3 | No | No | 0.0151 | 0.9849 |

Customer 1 was predicted to churn, with an estimated churn probability of 65.01%.

Customer 2 was predicted not to churn, with an estimated churn probability of 4.82%.

Customer 3 was predicted not to churn, with an estimated churn probability of 1.51%.

The confidence score represents the model's probability for the predicted class. It does not guarantee that a prediction is correct.

## Task 5: PII Detection and Redaction

A PII guardrail was implemented to detect and redact supported email-address and phone-number patterns.

The script uses regular expressions to identify these patterns and replaces detected information with redaction markers. Customer identifiers are excluded from the context sent to the LLM.

Example:

Original text:

text
Contact customer at sample@example.com


PII detection result:

text
Contains PII: True


Redacted text:

text
Contact customer at [REDACTED_EMAIL]


Results:

| Check | Result |
|---|---|
| PII detection | Successful |
| Email redaction | Successful |
| Redaction marker | [REDACTED_EMAIL] |
| Customer identifiers included in LLM context | No |

The PII guardrail test completed successfully.

However, regular-expression-based detection is limited to the patterns covered by the implementation and may not identify every type of sensitive information. Additional privacy checks would be required before using the system with real customer data.

## Task 6: Integrate the Groq LLM API

The Groq Chat Completions API was integrated to generate natural-language explanations for customer churn predictions.

The API endpoint used was:

text
https://api.groq.com/openai/v1/chat/completions


The configured LLM model was:

python
LLM_MODEL = "openai/gpt-oss-20b"


The API key was loaded from the .env file using Python-dotenv.

The API request includes a system prompt and a user prompt. The system prompt instructs the LLM to explain the supplied prediction without changing the model's prediction label or confidence score.

The temperature was set to 0.2, and the request used JSON object response mode.

The LLM was instructed to provide explanations using the supplied customer attributes, avoid unsupported causal claims, avoid personal information, and recommend a practical customer-retention action.

### JSON Response Schema

The response was validated using jsonschema.

The required fields were:

| Field | Description |
|---|---|
| prediction_label | Predicted churn label: Yes or No |
| confidence_level | Confidence score between 0 and 1, or null |
| top_reason | Primary explanation for the prediction |
| second_reason | Additional explanation based on customer attributes |
| next_step | Suggested customer-retention action |

The schema restricts prediction labels to Yes and No, limits numeric confidence values to the range 0 to 1, requires all five fields, and disallows additional properties.

After a valid response is received, the script explicitly restores prediction_label and confidence_level from the machine learning prediction. This prevents the LLM from changing those two values in the final explanation.

Results:

| Check | Result |
|---|---|
| Groq API connection | Successful |
| LLM responses | Successfully generated |
| JSON response parsing | Successful |
| JSON schema validation | Successful |
| Customer explanations generated | 3 out of 3 |

The LLM successfully generated structured explanations for all three customer samples.

## Task 7: Error Handling and Logging

Error handling was implemented to improve the reliability and maintainability of the script.

Console logging was configured for INFO-level messages and above, while the file handler records ERROR-level messages in Part4_errors.log.

The script includes checks for missing required files, invalid target labels, incorrect feature schemas, unsupported model predictions, and missing probability prediction support.

The Groq API integration inclu
des request timeouts, retries for eligible failures, and special handling for selected HTTP status codes. Unexpected errors and invalid LLM responses are logged for troubleshooting.

Results:

| Feature | Result |
|---|---|
| Required file validation | Implemented |
| Model and schema validation | Implemented |
| API request timeout | 60 seconds |
| Retry handling | Implemented |
| JSON parsing and validation | Implemented |
| Error log file | Part4_errors.log |
| API key printed in console | No |
| Final execution | Successful |

The script completed successfully and displayed the final message:

text
Part 4 execution finished.


## Task 8: Generate LLM Explanations for Customer Samples

The end-to-end demonstration selected three customer records using random_state=42.

For each customer, the script generated a churn prediction, calculated the churn probability and predicted-class confidence, and sent the relevant customer attributes and model output to the LLM.

The LLM returned a structured JSON response containing the prediction label, confidence level, two explanations, and a suggested next step.

### Customer 1

Results:

| Metric | Result |
|---|---|
| Actual Churn | Yes |
| Predicted Churn | Yes |
| Churn Probability | 0.6501 |
| Prediction Confidence | 0.6501 |

The LLM identified the customer's short tenure of one month and month-to-month contract as possible indicators of higher churn risk. It also noted the absence of several additional services.

The suggested next step was to consider a personalized offer, such as a discounted 12-month plan or a relevant service bundle.

### Customer 2

Results:

| Metric | Result |
|---|---|
| Actual Churn | No |
| Predicted Churn | No |
| Churn Probability | 0.0482 |
| Prediction Confidence | 0.9518 |

The LLM highlighted the customer's tenure of 41 months and monthly charge of $25.25 as characteristics associated with a stable customer relationship.

The suggested next step was to consider a relevant service bundle or loyalty reward.

### Customer 3

Results:

| Metric | Result |
|---|---|
| Actual Churn | No |
| Predicted Churn | No |
| Churn Probability | 0.0151 |
| Prediction Confidence | 0.9849 |

The LLM highlighted the customer's tenure of 52 months, two-year contract, and monthly charge of $19.35.

The suggested next step was to explore a suitable service bundle and gather customer satisfaction feedback.

These explanations are generated from the supplied customer information and general domain knowledge. They describe plausible patterns and should not be interpreted as proof that the identified characteristics caused the predictions.

# How to Run

### 1. Install the required libraries

bash
pip install pandas numpy scikit-learn joblib requests python-dotenv jsonschema


### 2. Check the required project files

Ensure that the following files are available:

- Part1_DataAcquisition_Cleaning_EDA/cleaned_data.csv
- Part3_Advanced_Modelling/best_model.pkl
- Part3_Advanced_Modelling/feature_columns.pkl
- Part4_LLM_Powered_feature/Part4_LLM_Powered_feature.py

### 3. Configure the API key

Create a .env file inside the Part4_LLM_Powered_feature directory.

Add the following line:

dotenv
LLM_API_KEY=your_actual_api_key


Replace the placeholder with a valid Groq API key. Do not publish the actual API key or commit the .env file to version control.

### 4. Navigate to the Part 4 directory

From the project root directory, run:

bash
cd Part4_LLM_Powered_feature


### 5. Run the Part 4 script

bash
python Part4_LLM_Powered_feature.py


### 6. Check the output

The script should display the PII guardrail test, load the trained model and feature schema, prepare the dataset, and generate predictions and LLM explanations for three customer samples.

If an unexpected error occurs, check Part4_errors.log for diagnostic information.

# Final Conclusion

Part 4 successfully integrated the trained machine learning pipeline from Part 3 with a Large Language Model through the Groq API.

The implementation loaded the saved model and feature schema, validated the cleaned dataset, prepared a feature matrix containing 30 columns, and generated churn predictions with probability-derived confidence scores.

The PII guardrail test successfully detected and redacted the sample email address. The Groq API integration generated structured JSON explanations for all three customer samples, including prediction labels, confidence scores, possible reasons based on customer attributes, and suggested customer-retention actions.

JSON schema validation, API error handling, request retries, and file-based error logging were also implemented.

Overall, this part demonstrates how machine learning predictions can be combined with LLM-generated explanations to make customer churn results easier to interpret. Although the implementation completed successfully, the explanations are generated by an LLM and should be reviewed before being used for real customer-retention decisions.

Part 4 completed successfully.