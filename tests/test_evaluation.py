import pandas as pd
import pytest
from evaluation import prepare_split
from feature_engineering import JobPostingFeatureEngineer
from train_model import build_stacking_model

def sample():
    return pd.DataFrame([{"job_id": i, "title": f"role {i}", "description": f"posting {i}", "fraudulent": i % 2} for i in range(40)])

def test_duplicate_groups_never_cross_split():
    frame = sample(); duplicate = frame.copy(); duplicate["job_id"] += 100
    train, test, info = prepare_split(pd.concat([frame, duplicate], ignore_index=True))
    assert set(train["description"]).isdisjoint(test["description"])
    assert info["duplicates_removed"] == 40 and info["overlapping_postings"] == 0

def test_conflicting_labels_rejected():
    frame = sample(); duplicate = frame.iloc[[0]].copy(); duplicate["fraudulent"] = 1
    with pytest.raises(ValueError, match="conflicting"): prepare_split(pd.concat([frame, duplicate], ignore_index=True))

def test_features_and_fold_local_preprocessing():
    features = JobPostingFeatureEngineer().fit_transform(sample().drop(columns="fraudulent"))
    assert "fraudulent" not in features and "job_id" not in features
    model = build_stacking_model()
    assert all("preprocessor" in pipeline.named_steps and "sampler" in pipeline.named_steps for _, pipeline in model.estimators)


def test_duplicate_dataframe_indices_are_safe():
    train, test, info = prepare_split(pd.concat([sample(), sample()]))
    assert info["duplicates_removed"] == 40
    assert set(train["description"]).isdisjoint(test["description"])
