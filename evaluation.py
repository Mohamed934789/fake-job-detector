"""Duplicate-safe raw-data split; IDs are excluded and ambiguous labels rejected."""
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_split(frame, test_size=.2, random_state=42):
    frame = frame.reset_index(drop=True)
    if "fraudulent" not in frame or frame["fraudulent"].isna().any():
        raise ValueError("A complete fraudulent label is required")
    if not frame["fraudulent"].isin([0, 1]).all():
        raise ValueError("Labels must be binary 0/1")
    features = [c for c in frame.columns if c not in {"job_id", "fraudulent"}]
    # Normalize missing values and exact whitespace before grouping repeated postings.
    normalized = frame[features].fillna("").astype(str).apply(lambda column: column.str.strip())
    identity = pd.util.hash_pandas_object(normalized, index=False)
    labels = frame.groupby(identity)["fraudulent"].nunique()
    if (labels > 1).any():
        raise ValueError("Identical postings have conflicting labels; review these records first")
    keep = ~identity.duplicated()
    clean = frame.loc[keep].copy()
    groups = identity.loc[keep]
    train, test = train_test_split(clean, test_size=test_size, random_state=random_state, stratify=clean["fraudulent"])
    train_groups, test_groups = set(groups.loc[train.index]), set(groups.loc[test.index])
    if train_groups & test_groups: raise AssertionError("Duplicate leakage")
    fingerprint = lambda indices: hashlib.sha256("\n".join(sorted(str(groups.loc[i]) for i in indices)).encode()).hexdigest()
    return train, test, {"input_rows": len(frame), "unique_rows": len(clean), "duplicates_removed": int((~keep).sum()),
        "train_rows": len(train), "test_rows": len(test), "overlapping_postings": 0,
        "train_split_sha256": fingerprint(train.index), "test_split_sha256": fingerprint(test.index), "random_state": random_state}
