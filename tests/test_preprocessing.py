import numpy as np

import pandas as pd
import pytest

from src.data import (
    load_config,
    make_synthetic_baf_data,
    make_synthetic_fraud_data,
    prepare_dataset,
    split_dataframe,
    split_integrity_summary,
)


def test_temporal_split_and_shapes():
    config = load_config("configs/ieee_cis.yaml")
    frame = make_synthetic_fraud_data(1000, seed=7)
    prepared = prepare_dataset(frame, config)
    assert prepared.train_frame["TransactionDT"].max() <= prepared.validation_frame["TransactionDT"].min()
    assert prepared.validation_frame["TransactionDT"].max() <= prepared.test_frame["TransactionDT"].min()
    assert prepared.X_train.shape[0] == len(prepared.y_train)
    assert prepared.X_train.shape[1] == prepared.X_validation.shape[1] == prepared.X_test.shape[1]
    assert np.isfinite(prepared.X_train).all()
    assert "TransactionDT" not in prepared.feature_names


def test_preprocessor_is_fitted_on_train_only():
    config = load_config("configs/ieee_cis.yaml")
    frame = make_synthetic_fraud_data(1000, seed=9)
    prepared = prepare_dataset(frame, config)
    numeric_pipeline = prepared.preprocessor.transformer.named_transformers_["numeric"]
    scaler = numeric_pipeline.named_steps["scaler"]
    numeric = prepared.preprocessor.numeric_features
    imputer = numeric_pipeline.named_steps["imputer"]
    train_imputed = imputer.transform(prepared.train_frame[numeric])
    assert np.allclose(scaler.mean_, train_imputed.mean(axis=0))


def test_baf_temporal_groups_are_disjoint_and_time_is_not_a_feature():
    config = load_config("configs/baf.yaml")
    frame = make_synthetic_baf_data(8000, seed=17)
    prepared = prepare_dataset(frame, config)
    summary = split_integrity_summary(
        prepared.train_frame, prepared.validation_frame, prepared.test_frame, "month"
    )
    assert summary["group_disjoint"].all()
    assert prepared.train_frame["month"].unique().tolist() == [0, 1, 2, 3, 4]
    assert prepared.validation_frame["month"].unique().tolist() == [5]
    assert prepared.test_frame["month"].unique().tolist() == [6, 7]
    assert "month" not in prepared.feature_names


def test_temporal_group_split_rejects_unassigned_groups():
    frame = pd.DataFrame({"month": [0, 1, 2, 3], "target": [0, 1, 0, 1]})
    split_config = {
        "strategy": "temporal_group",
        "train_size": 0.5,
        "validation_size": 0.25,
        "test_size": 0.25,
        "train_groups": [0],
        "validation_groups": [1],
        "test_groups": [2],
    }
    with pytest.raises(ValueError, match="unassigned"):
        split_dataframe(frame, "target", split_config, "month")


def test_split_summary_bounds_high_cardinality_time_metadata():
    frame = pd.DataFrame({"time": np.arange(600), "target": np.arange(600) % 2})
    summary = split_integrity_summary(frame.iloc[:300], frame.iloc[300:450], frame.iloc[450:], "time")
    assert summary["time_group_count"].tolist() == [300, 150, 150]
    assert summary["time_groups"].tolist() == [[], [], []]
    assert summary["overlap_groups"].tolist() == [[], [], []]
    assert summary["group_disjoint"].all()
