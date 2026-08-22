from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from src.data import load_config, load_fraud_dataframe


def test_ieee_loader_records_identity_join_coverage(tmp_path):
    transactions = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [10, 20, 30, 40],
            "TransactionAmt": [1.0, 2.0, 3.0, 4.0],
            "isFraud": [0, 0, 1, 0],
        }
    )
    identities = pd.DataFrame({"TransactionID": [2, 4], "DeviceType": ["mobile", "desktop"]})
    transactions.to_csv(tmp_path / "train_transaction.csv", index=False)
    identities.to_csv(tmp_path / "train_identity.csv", index=False)

    config = deepcopy(load_config("configs/ieee_cis.yaml"))
    frame = load_fraud_dataframe(config, data_root=tmp_path)

    assert frame.attrs["identity_join_coverage"] == pytest.approx(0.5)
    assert frame.attrs["benchmark_type"] == "observational_competition_benchmark"
    assert len(frame) == len(transactions)


def test_ieee_loader_rejects_duplicate_identity_keys(tmp_path):
    pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [10, 20],
            "TransactionAmt": [1.0, 2.0],
            "isFraud": [0, 1],
        }
    ).to_csv(tmp_path / "train_transaction.csv", index=False)
    pd.DataFrame(
        {"TransactionID": [1, 1], "DeviceType": ["mobile", "desktop"]}
    ).to_csv(tmp_path / "train_identity.csv", index=False)

    config = deepcopy(load_config("configs/ieee_cis.yaml"))
    with pytest.raises(ValueError, match="must be unique"):
        load_fraud_dataframe(config, data_root=tmp_path)


def test_baf_loader_labels_the_published_benchmark_source(tmp_path):
    pd.DataFrame(
        {"month": [0, 5, 6], "income": [0.1, -1.0, 0.9], "fraud_bool": [0, 1, 0]}
    ).to_csv(tmp_path / "Base.csv", index=False)

    config = deepcopy(load_config("configs/baf.yaml"))
    frame = load_fraud_dataframe(config, data_root=tmp_path)

    assert frame.attrs["benchmark_type"] == "privacy_preserving_synthetic_benchmark"
    assert int(frame.eq(-1).to_numpy().sum()) == 1
