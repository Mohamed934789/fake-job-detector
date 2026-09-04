"""
feature_engineering.py
-----------------------
Reproduces the exact feature-engineering logic used in the notebook's
EX3 experiment (the one that fed the winning EX4 Stacking model).

This module MUST be importable both when the model is trained
(train_model.py) and when it is loaded again for inference
(app.py), because the fitted pipeline stores a reference to
`JobPostingFeatureEngineer` inside the pickled object.
"""

import re
import string

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ----------------------------------------------------------------------
# Columns expected on the RAW input (same names as the original Kaggle
# "fake_job_postings.csv" dataset)
# ----------------------------------------------------------------------
TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]

CATEGORICAL_COLUMNS_RAW = [
    "department",
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "location",
]

BINARY_RAW_COLUMNS = ["telecommuting", "has_company_logo", "has_questions"]

# Final numeric feature set used by the model (matches notebook EX3/EX4)
NUMERIC_FEATURES = [
    "title_len",
    "description_len",
    "has_salary",
    "has_location",
    "has_company_profile",
    "telecommuting",
    "has_company_logo",
    "has_questions",
]

TEXT_FEATURE = "text_context"


def clean_text(text):
    """Same cleaning function used in the notebook (cell 9)."""
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


class JobPostingFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Turns a raw job-posting DataFrame into the feature set consumed by the
    Stacking model:
      - text_context  (title + description + requirements + benefits, cleaned)
      - title_len, description_len
      - has_salary, has_location, has_company_profile
      - telecommuting, has_company_logo, has_questions
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Make sure every expected raw column exists, even if the caller
        # (e.g. the Streamlit form) didn't provide it.
        for col in TEXT_COLUMNS + CATEGORICAL_COLUMNS_RAW:
            if col not in df.columns:
                df[col] = np.nan
        for col in BINARY_RAW_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        # --- Fill missing values with "Unknown" (as in cell 6) ---
        for col in CATEGORICAL_COLUMNS_RAW:
            df[col] = df[col].fillna("Unknown").astype(str)
            df[col] = df[col].replace("", "Unknown")
        for col in TEXT_COLUMNS:
            df[col] = df[col].fillna("Unknown").astype(str)
            df[col] = df[col].replace("", "Unknown")

        # --- Availability flags computed BEFORE text cleaning ---
        # (these compare against the capitalised "Unknown" placeholder,
        # exactly like the notebook)
        df["has_salary"] = (df["salary_range"] != "Unknown").astype(int)
        df["has_location"] = (df["location"] != "Unknown").astype(int)

        # --- Clean text columns (lowercases "Unknown" -> "unknown") ---
        for col in TEXT_COLUMNS:
            df[col] = df[col].apply(clean_text)

        df["has_company_profile"] = (df["company_profile"] != "unknown").astype(int)

        # --- Binary flags already present in the raw dataset ---
        for col in BINARY_RAW_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # --- Length features ---
        df["title_len"] = df["title"].apply(len)
        df["description_len"] = df["description"].apply(len)

        # --- Combined text feature (EX3) ---
        df[TEXT_FEATURE] = (
            df["title"] + " " + df["description"] + " " + df["requirements"] + " " + df["benefits"]
        )

        output_cols = [TEXT_FEATURE] + NUMERIC_FEATURES
        return df[output_cols]
