import os
import json
import logging
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from jsonschema import validate, ValidationError


# ============================================================
# 1. CONFIGURATION AND FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = (
    PROJECT_DIR
    / "Part1_DataAcquisition_Cleaning_EDA"
    / "cleaned_data.csv"
)

MODEL_PATH = (
    PROJECT_DIR
    / "Part3_Advanced_Modelling"
    / "best_model.pkl"
)

FEATURES_PATH = (
    PROJECT_DIR
    / "Part3_Advanced_Modelling"
    / "feature_columns.pkl"
)

ENV_PATH = BASE_DIR / ".env"
ERROR_LOG_PATH = BASE_DIR / "Part4_errors.log"

API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = "openai/gpt-oss-20b"

TARGET_COLUMN = "Churn"
ID_COLUMNS = ["customerID"]


# Console: INFO and above. File: ERROR and above.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Avoid duplicate file handlers if this module is reloaded.
if not any(
    isinstance(handler, logging.FileHandler)
    and Path(handler.baseFilename) == ERROR_LOG_PATH.resolve()
    for handler in logger.handlers
):
    error_handler = logging.FileHandler(
        ERROR_LOG_PATH,
        mode="a",
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )
    logger.addHandler(error_handler)

load_dotenv(ENV_PATH)
LLM_API_KEY = os.getenv("LLM_API_KEY")


# ============================================================
# 2. CHECK FILES AND LOAD THE MODEL
# ============================================================

def check_required_files():
    required_files = [DATA_PATH, MODEL_PATH, FEATURES_PATH]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )


def load_telco_model():
    check_required_files()

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    if not hasattr(model, "predict"):
        raise TypeError("The saved model cannot make predictions.")

    if not isinstance(feature_columns, list):
        raise TypeError(
            "feature_columns.pkl must contain a list of feature names."
        )

    if len(feature_columns) != 30:
        logger.warning(
            "Expected 30 features from the current Part 3 setup, "
            "but loaded %d.",
            len(feature_columns)
        )

    if any(
        str(col).startswith(("Item_", "Outlet_"))
        for col in feature_columns
    ):
        raise ValueError(
            "Wrong feature schema detected. This appears to be "
            "a BigMart schema, not a Telco churn schema."
        )

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None and list(model_features) != feature_columns:
        raise ValueError(
            "The model's feature names do not match feature_columns.pkl. "
            "Rerun Part 3 to regenerate both artifacts."
        )

    logger.info("Telco model loaded: %s", MODEL_PATH)
    logger.info("Feature schema loaded: %s", FEATURES_PATH)
    logger.info("Model type: %s", type(model).__name__)
    logger.info("Number of model features: %d", len(feature_columns))

    return model, feature_columns


# ============================================================
# 3. LOAD THE CLEANED TELCO DATASET
# ============================================================

def load_telco_data():
    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()

    invalid_labels = set(df[TARGET_COLUMN].unique()) - {"Yes", "No"}
    if invalid_labels:
        raise ValueError(f"Unexpected churn labels: {invalid_labels}")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    X = X.drop(columns=ID_COLUMNS, errors="ignore")

    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(
            X["TotalCharges"],
            errors="coerce"
        )

    logger.info("Dataset rows: %d", len(X))
    logger.info("Raw Telco feature columns: %d", X.shape[1])

    return X, y


# ============================================================
# 4. PREPARE FEATURES TO MATCH PART 3
# ============================================================

def prepare_features(X, feature_columns):
    """
    Reproduce Part 3's one-hot encoding and preserve the saved
    feature names and order. The saved model pipeline performs
    its own imputation and scaling as configured in Part 3.
    """
    X = X.copy()

    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(
            X["TotalCharges"],
            errors="coerce"
        )

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    X_encoded = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")
    X_encoded = X_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Fill remaining missing values; the model pipeline's imputer
    # remains in place as an additional safeguard.
    medians = X_encoded.median(numeric_only=True)
    X_encoded = X_encoded.fillna(medians).fillna(0)

    if list(X_encoded.columns) != feature_columns:
        raise ValueError(
            "Prepared features do not match the saved feature schema."
        )

    if X_encoded.shape[1] != len(feature_columns):
        raise ValueError(
            "Prepared feature count does not match the model."
        )

    if any(
        str(col).startswith(("Item_", "Outlet_"))
        for col in X_encoded.columns
    ):
        raise ValueError(
            "Unrelated BigMart features detected. Stopping predictions."
        )

    logger.info("Prepared feature matrix shape: %s", X_encoded.shape)
    return X_encoded


# ============================================================
# 5. GENERATE MODEL PREDICTIONS
# ============================================================

def predict_churn(model, customer_features):
    numeric_prediction = int(model.predict(customer_features)[0])

    if numeric_prediction not in (0, 1):
        raise ValueError(
            f"Unexpected model prediction: {numeric_prediction}"
        )

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The loaded model does not support predict_proba()."
        )

    classes = list(model.classes_)
    if 0 not in classes or 1 not in classes:
        raise ValueError(f"Unexpected model classes: {classes}")

    probabilities = model.predict_proba(customer_features)[0]
    churn_probability = float(probabilities[classes.index(1)])

    churn_label = "Yes" if numeric_prediction == 1 else "No"
    confidence = (
        churn_probability
        if numeric_prediction == 1
        else 1.0 - churn_probability
    )

    return {
        "prediction_label": churn_label,
        "confidence_level": round(confidence, 4),
        "churn_probability": round(churn_probability, 4)
    }


# ============================================================
# 6. PII DETECTION AND REDACTION
# ============================================================

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)"
)


def contains_pii(text):
    if not isinstance(text, str):
        return False
    return bool(EMAIL_PATTERN.search(text) or PHONE_PATTERN.search(text))


def redact_pii(text):
    if not isinstance(text, str):
        return text

    text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    text = PHONE_PATTERN.sub("[REDACTED_PHONE]", text)
    return text


def safe_customer_context(customer_features):
    safe_features = {}

    for key, value in customer_features.items():
        if key in ID_COLUMNS:
            continue

        if pd.isna(value):
            value = None
        elif isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, str):
            value = redact_pii(value)

        safe_features[key] = value

    return safe_features


# ============================================================
# 7. JSON RESPONSE SCHEMA
# ============================================================

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {
            "type": "string",
            "enum": ["Yes", "No"]
        },
        "confidence_level": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 1
        },
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"}
    },
    "required": [
        "prediction_label",
        "confidence_level",
        "top_reason",
        "second_reason",
        "next_step"
    ],
    "additionalProperties": False
}


# ============================================================
# 8. GROQ API CONNECTION
# ============================================================

def call_llm(system_prompt, user_prompt, max_retries=2):
    if not LLM_API_KEY:
        logger.error(
            "LLM_API_KEY is missing. Check the .env file at: %s",
            ENV_PATH
        )
        return None

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code != 200:
                # Avoid logging request headers or the API key.
                logger.error(
                    "Groq API error %s: %s",
                    response.status_code,
                    response.text[:1000]
                )

                if response.status_code in (400, 401, 403, 404):
                    return None

                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return None

            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            result = json.loads(content)
            validate(instance=result, schema=RESPONSE_SCHEMA)
            return result

        except requests.RequestException:
            logger.exception(
                "Groq API request failed on attempt %d of %d.",
                attempt + 1,
                max_retries + 1
            )

        except (KeyError, IndexError, json.JSONDecodeError):
            logger.exception("Could not parse the Groq LLM response.")
            return None

        except ValidationError:
            logger.exception("LLM response failed JSON schema validation.")
            return None

        except Exception:
            logger.exception("Unexpected error while calling the LLM.")
            return None

        if attempt < max_retries:
            time.sleep(2 ** attempt)

    return None


# ============================================================
# 9. PROMPTS AND LLM EXPLANATIONS
# ============================================================

SYSTEM_PROMPT = """
You are a responsible Telco Customer Churn explanation assistant.

You will receive a churn prediction from a trained machine-learning
model, its confidence, and a customer's service attributes.

Rules:
1. Treat the supplied model prediction as authoritative.
2. Do not change the prediction label or confidence.
3. Explain plausible factors using only supplied customer attributes
   and general domain knowledge.
4. Do not claim an attribute caused churn without causal evidence.
   Use cautious language such as "may contribute".
5. Do not invent customer history or unsupported facts.
6. Do not include personally identifying information.
7. Recommend a practical and respectful retention action.
8. Return one valid JSON object only.
9. Use exactly these fields: prediction_label, confidence_level,
   top_reason, second_reason, next_step.
"""


def explain_prediction(customer_features, prediction):
    safe_features = safe_customer_context(customer_features)

    model_output = {
        "prediction_label": prediction["prediction_label"],
        "confidence_level": prediction["confidence_level"]
    }

    user_prompt = f"""
Explain this Telco Customer Churn model prediction.

MODEL OUTPUT:
{json.dumps(model_output, ensure_ascii=False)}

CUSTOMER FEATURES:
{json.dumps(safe_features, ensure_ascii=False)}

Return one JSON object with these fields:
{{
    "prediction_label": "Yes or No",
    "confidence_level": 0.0,
    "top_reason": "A cautious explanation grounded in supplied data",
    "second_reason": "Another plausible explanation grounded in data",
    "next_step": "A practical, respectful retention action"
}}

Copy prediction_label and confidence_level exactly from MODEL OUTPUT.
Do not return customer identifiers or personal contact information.
"""

    result = call_llm(SYSTEM_PROMPT, user_prompt)
    if result is None:
        return None

    # The LLM explains the prediction but cannot change these fields.
    result["prediction_label"] = prediction["prediction_label"]
    result["confidence_level"] = prediction["confidence_level"]
    return result


# ============================================================
# 10. END-TO-END DEMONSTRATION
# ============================================================

def run_demo(model, feature_columns, X, y):
    X_prepared = prepare_features(X, feature_columns)

    print("\nPrepared feature matrix shape:", X_prepared.shape)
    print(
        "Feature count matches saved model:",
        X_prepared.shape[1] == len(feature_columns)
    )

    sample_indices = X.sample(
        n=min(3, len(X)),
        random_state=42
    ).index

    for number, idx in enumerate(sample_indices, start=1):
        raw_customer = X.loc[idx]
        customer_df = X_prepared.loc[[idx]]

        prediction = predict_churn(model, customer_df)

        print("\n" + "=" * 65)
        print(f"CUSTOMER SAMPLE {number}")
        print("=" * 65)
        print("Actual churn label:", "Yes" if y.loc[idx] == 1 else "No")
        print("Predicted churn:", prediction["prediction_label"])
        print("Churn probability:", prediction["churn_probability"])
        print("Predicted-class confidence:", prediction["confidence_level"])

        explanation = explain_prediction(
            raw_customer.to_dict(),
            prediction
        )

        if explanation is None:
            print(
                "\nLLM explanation unavailable. Check the API configuration "
                "and Part4_errors.log."
            )
            continue

        print("\nLLM Explanation:")
        print(json.dumps(explanation, indent=4, ensure_ascii=False))


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

def main():
    print("=" * 65)
    print("TELCO CUSTOMER CHURN - PART 4")
    print("LLM-Powered Prediction Explanation")
    print("=" * 65)

    pii_test = "Contact customer at sample@example.com"

    print("\nPII Guardrail Test")
    print("Contains PII:", contains_pii(pii_test))
    print("Redacted text:", redact_pii(pii_test))

    model, feature_columns = load_telco_model()
    X, y = load_telco_data()
    run_demo(model, feature_columns, X, y)

    print("\nPart 4 execution finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Record the complete traceback to Part4_errors.log, then
        # re-raise so PowerShell also shows that execution failed.
        logger.exception("Part 4 execution failed.")
        raise