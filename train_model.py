"""
train_model.py
---------------
Memory-optimized version of the Winning Stacking Ensemble.

Base learners:
    - Logistic Regression + RandomOverSampler
    - Linear SVC + RandomOverSampler
    - Random Forest + RandomOverSampler
    - XGBoost + RandomOverSampler

Designed to run on a machine with limited RAM (e.g. 16 GB)
using Python 3.14.

Output:
    fake_job_stacking_model.joblib
"""

import time

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from xgboost import XGBClassifier

from feature_engineering import (
    JobPostingFeatureEngineer,
    NUMERIC_FEATURES,
    TEXT_FEATURE,
)


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "fake_job_postings.csv"
MODEL_OUT = "fake_job_stacking_model.joblib"

RANDOM_STATE = 42


# ============================================================
# Preprocessor
# ============================================================

def build_preprocessor():
    """
    TF-IDF + StandardScaler preprocessing.

    float32 is used to reduce memory consumption.
    """

    return ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=1,
                    stop_words="english",
                    dtype=np.float32,
                ),
                TEXT_FEATURE,
            ),
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ]
    )


# ============================================================
# Stacking Model
# ============================================================

def build_stacking_model():

    # --------------------------------------------------------
    # Logistic Regression + Random Over Sampling
    # --------------------------------------------------------

    ros_logreg = ImbPipeline(
        [
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "sampler",
                RandomOverSampler(
                    random_state=RANDOM_STATE
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=100,
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Linear SVC + Random Over Sampling
    # --------------------------------------------------------

    ros_svc = ImbPipeline(
        [
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "sampler",
                RandomOverSampler(
                    random_state=RANDOM_STATE
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=10,
                    max_iter=10000,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Random Forest + Random Over Sampling
    #
    # Changed from SMOTE -> ROS to reduce memory usage.
    # --------------------------------------------------------

    ros_rf = ImbPipeline(
        [
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "sampler",
                RandomOverSampler(
                    random_state=RANDOM_STATE
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=70,
                    max_depth=20,
                    min_samples_split=5,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # XGBoost + Random Over Sampling
    #
    # Changed from SMOTE -> ROS to reduce memory usage.
    # --------------------------------------------------------

    ros_xgb = ImbPipeline(
        [
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "sampler",
                RandomOverSampler(
                    random_state=RANDOM_STATE
                ),
            ),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    tree_method="hist",
                    max_bin=64,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Stacking Ensemble
    # --------------------------------------------------------

    stack = StackingClassifier(
        estimators=[
            ("lr", ros_logreg),
            ("svc", ros_svc),
            ("rf", ros_rf),
            ("xgb", ros_xgb),
        ],
        final_estimator=LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        cv=5,
        n_jobs=1,
    )

    return stack


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Fake Job Postings Detection")
    print("Memory-Optimized Stacking Ensemble")
    print("=" * 60)

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = df["fraudulent"]

    print("\nClass distribution:")
    print(y.value_counts())

    # --------------------------------------------------------
    # Train / Test Split
    # --------------------------------------------------------

    print("\nSplitting dataset...")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training samples:", len(X_train_raw))
    print("Testing samples :", len(X_test_raw))

    # --------------------------------------------------------
    # Full Pipeline
    # --------------------------------------------------------

    full_pipeline = Pipeline(
        [
            (
                "features",
                JobPostingFeatureEngineer(),
            ),
            (
                "stack",
                build_stacking_model(),
            ),
        ]
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Training Stacking Ensemble...")
    print("This may take several minutes.")
    print("=" * 60)

    start = time.time()

    full_pipeline.fit(
        X_train_raw,
        y_train,
    )

    training_time = time.time() - start

    print(
        f"\nTraining completed in "
        f"{training_time:.1f} seconds."
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    y_pred = full_pipeline.predict(
        X_test_raw
    )

    y_prob = full_pipeline.predict_proba(
        X_test_raw
    )[:, 1]

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob,
    )

    print("\n" + "=" * 60)
    print("TEST SET PERFORMANCE")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"ROC-AUC  : {roc_auc:.3f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Save Model
    # --------------------------------------------------------

    print("\nSaving trained model...")

    joblib.dump(
        full_pipeline,
        MODEL_OUT,
        compress=3,
    )

    print(
        f"\nModel saved successfully:"
        f"\n{MODEL_OUT}"
    )

    print("\nYou can now run:")
    print("streamlit run app.py")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()