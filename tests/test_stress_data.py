from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.stress_testing import (
    AMLNET_LEAKAGE_DENYLIST,
    AMLNET_V1_MD5,
    TRANSXION_V2_SHA256,
    file_checksum,
    load_amlnet_dataset,
    load_stress_dataset,
    load_transxion_dataset,
    make_synthetic_amlnet_test_fixture,
    make_synthetic_transxion_test_fixture,
    reject_git_lfs_pointer,
    resolve_stress_data_file,
    temporal_stress_split,
)
from src.stress_testing.data import _add_amlnet_history


def _transxion_config(path: Path, *, join_profiles: bool = True) -> dict:
    return {
        "dataset": {
            "name": "transxion_v2",
            "data_file": path.name,
            "join_profiles": join_profiles,
            "profile_files": ["person.csv", "merchant.csv"],
            "checksum": {
                "algorithm": "sha256",
                "value": file_checksum(path, "sha256"),
            },
            "verify_checksum": True,
        },
        "split": {"train_size": 0.60, "validation_size": 0.20, "test_size": 0.20},
    }


def _amlnet_config(path: Path) -> dict:
    return {
        "dataset": {
            "name": "amlnet_v1.0",
            "data_file": path.name,
            "target_column": "isMoneyLaundering",
            "checksum": {"algorithm": "md5", "value": file_checksum(path, "md5")},
            "verify_checksum": True,
        },
        "split": {"train_size": 0.60, "validation_size": 0.20, "test_size": 0.20},
    }


def test_published_checksum_constants_are_exact():
    assert TRANSXION_V2_SHA256 == (
        "d6c345f07a8d8e26123dba5fe4f6572ef198e04fb94f9b53facb3fef6d197a35"
    )
    assert AMLNET_V1_MD5 == "7668fc7d74c787e07546ce85c6f790b9"


def test_recursive_exact_file_discovery_and_ambiguity(tmp_path):
    nested = tmp_path / "kaggle-dataset" / "version-1"
    nested.mkdir(parents=True)
    expected = nested / "tx.csv"
    expected.write_text("a\n1\n", encoding="utf-8")
    assert resolve_stress_data_file("tx.csv", tmp_path) == expected.resolve()

    second = tmp_path / "another-version" / "tx.csv"
    second.parent.mkdir()
    second.write_text("a\n2\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Ambiguous exact filename"):
        resolve_stress_data_file("tx.csv", tmp_path)


def test_git_lfs_pointer_is_rejected_before_csv_parsing(tmp_path):
    pointer = tmp_path / "tx.csv"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:d6c345f07a8d8e26123dba5fe4f6572ef198e04fb94f9b53facb3fef6d197a35\n"
        "size 245159741\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Git LFS pointer"):
        reject_git_lfs_pointer(pointer)
    with pytest.raises(ValueError, match="Git LFS pointer"):
        load_transxion_dataset(
            {"dataset": {"name": "transxion_v2", "data_file": "tx.csv"}},
            tmp_path,
        )


def test_transxion_loads_canonical_schema_profiles_and_causal_features(tmp_path):
    paths = make_synthetic_transxion_test_fixture(tmp_path)
    bundle = load_stress_dataset(_transxion_config(paths["transactions"]), tmp_path)
    frame = bundle.frame

    canonical = {
        "event_time",
        "sender_bank_id",
        "sender_account_id",
        "receiver_bank_id",
        "receiver_account_id",
        "amount_received",
        "amount_paid",
        "label",
    }
    semantic = {
        "event_hour",
        "event_day_of_week",
        "is_weekend",
        "amount_paid_log",
        "amount_received_log",
        "amount_relative_difference",
        "same_currency",
        "cross_bank",
    }
    causal = {
        "sender_prior_transaction_count",
        "sender_prior_total_amount_paid",
        "sender_seconds_since_previous",
        "receiver_prior_transaction_count",
        "receiver_prior_total_amount_received",
        "receiver_seconds_since_previous",
        "pair_prior_transaction_count",
    }
    assert canonical | semantic | causal <= set(frame.columns)
    assert frame["event_time"].is_monotonic_increasing
    assert set(frame["label"].unique()) == {0, 1}
    assert bundle.manifest["data_quality"]["profile_join_coverage"] == {
        "sender": 1.0,
        "receiver": 1.0,
    }
    assert any(column.startswith("sender_profile_") for column in frame)
    assert any(column.startswith("receiver_profile_") for column in frame)
    assert len(frame) == bundle.manifest["population"]["rows"]
    assert "label" not in bundle.feature_columns
    assert causal <= set(bundle.feature_columns)
    assert bundle.manifest["source_files"][0]["verified"] is True
    assert bundle.manifest["source_reference"]["canonical_dataset_name"] == "TransXion"
    assert bundle.manifest["source_reference"]["paper_revision"] == "v2"
    assert bundle.manifest["source_reference"]["dataset_v2_release_claimed"] is False

    first_sender_rows = frame.groupby(
        ["sender_bank_id", "sender_account_id"], sort=False, dropna=False
    ).head(1)
    assert (first_sender_rows["sender_prior_transaction_count"] == 0).all()
    assert (first_sender_rows["sender_prior_total_amount_paid"] == 0.0).all()


def test_transxion_accepts_the_official_duplicate_account_header(tmp_path):
    paths = make_synthetic_transxion_test_fixture(tmp_path)
    raw = pd.read_csv(paths["transactions"])
    raw.columns = [
        "Account" if column in {"From Account", "To Account"} else column
        for column in raw.columns
    ]
    raw.to_csv(paths["transactions"], index=False)

    observed = pd.read_csv(paths["transactions"], nrows=1).columns.tolist()
    assert "Account" in observed and "Account.1" in observed
    bundle = load_transxion_dataset(
        _transxion_config(paths["transactions"], join_profiles=False),
        tmp_path,
    )
    assert {
        "sender_account_id",
        "receiver_account_id",
    } <= set(bundle.frame.columns)
    assert bundle.frame["sender_account_id"].notna().all()
    assert bundle.frame["receiver_account_id"].notna().all()


def test_transxion_history_features_do_not_use_labels(tmp_path):
    paths = make_synthetic_transxion_test_fixture(tmp_path / "original")
    original = load_transxion_dataset(
        _transxion_config(paths["transactions"], join_profiles=False),
        paths["transactions"].parent,
    ).frame

    changed_root = tmp_path / "changed"
    changed_paths = make_synthetic_transxion_test_fixture(changed_root)
    changed_raw = pd.read_csv(changed_paths["transactions"])
    changed_raw["Is Laundering"] = 1 - changed_raw["Is Laundering"]
    changed_raw.to_csv(changed_paths["transactions"], index=False)
    changed = load_transxion_dataset(
        _transxion_config(changed_paths["transactions"], join_profiles=False),
        changed_root,
    ).frame

    history = [column for column in original if "prior_" in column or "seconds_since" in column]
    pd.testing.assert_frame_equal(original[history], changed[history])


def test_history_features_exclude_other_rows_at_the_same_timestamp():
    frame = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
                utc=True,
            ),
            "sender_account_id": ["sender", "sender", "sender"],
            "receiver_account_id": ["receiver", "receiver", "receiver"],
            "amount": [10.0, 20.0, 30.0],
        }
    )

    history = _add_amlnet_history(frame)

    assert history["sender_prior_transaction_count"].tolist() == [0, 0, 2]
    assert history["sender_prior_total_amount"].tolist() == [0.0, 0.0, 30.0]
    assert history["pair_prior_transaction_count"].tolist() == [0, 0, 2]
    assert history["sender_seconds_since_previous"].iloc[:2].isna().all()
    assert history["sender_seconds_since_previous"].iloc[2] == pytest.approx(3600.0)


def test_transxion_rejects_duplicate_composite_profile_key(tmp_path):
    paths = make_synthetic_transxion_test_fixture(tmp_path)
    person = pd.read_csv(paths["person"])
    person = pd.concat([person, person.iloc[[0]]], ignore_index=True)
    person.to_csv(paths["person"], index=False)

    with pytest.raises(ValueError, match=r"unique on \(bank, bank_account_number\)"):
        load_transxion_dataset(_transxion_config(paths["transactions"]), tmp_path)


def test_checksum_mismatch_is_a_hard_failure(tmp_path):
    paths = make_synthetic_transxion_test_fixture(tmp_path)
    config = _transxion_config(paths["transactions"], join_profiles=False)
    config["dataset"]["checksum"]["value"] = "0" * 64
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_transxion_dataset(config, tmp_path)


def test_amlnet_metadata_parse_sort_and_leakage_contract(tmp_path):
    path = make_synthetic_amlnet_test_fixture(tmp_path)
    raw = pd.read_csv(path)
    # Reverse source rows to demonstrate timestamp + original-order stable sort.
    raw.iloc[::-1].to_csv(path, index=False)
    bundle = load_amlnet_dataset(_amlnet_config(path), tmp_path)
    frame = bundle.frame

    assert frame["event_time"].notna().all()
    assert frame["event_time"].is_monotonic_increasing
    assert bundle.manifest["data_quality"]["timestamp_parse_coverage"] == 1.0
    assert bundle.manifest["source_reference"]["doi"] == "10.5281/zenodo.16736515"
    assert bundle.manifest["source_reference"]["canonical_dataset_name"] == "AMLNet"
    assert bundle.manifest["source_reference"]["license"] == "CC BY-NC 4.0"
    assert {
        "event_hour",
        "event_day_of_week",
        "is_weekend",
        "amount_log",
        "amount_to_sender_balance_fraction",
        "sender_prior_transaction_count",
        "receiver_prior_transaction_count",
        "pair_prior_transaction_count",
    } <= set(bundle.feature_columns)
    assert np.isfinite(frame["amount_to_sender_balance_fraction"]).all()
    assert "balance_depletion_fraction" not in frame
    assert set(AMLNET_LEAKAGE_DENYLIST).isdisjoint(bundle.feature_columns)
    assert {
        "label",
        "auxiliary_fraud_label",
        "money_laundering_typology",
        "metadata",
        "fraud_probability",
        "sender_balance_after",
    }.isdisjoint(bundle.feature_columns)

    tied = frame.groupby("event_time", sort=False)["__source_order"].apply(list)
    assert all(values == sorted(values) for values in tied)


def test_amlnet_requires_one_hundred_percent_metadata_timestamp_coverage(tmp_path):
    path = make_synthetic_amlnet_test_fixture(tmp_path)
    raw = pd.read_csv(path)
    raw.loc[17, "metadata"] = "{'timestamp': 'not-a-datetime'}"
    raw.to_csv(path, index=False)

    with pytest.raises(ValueError, match="parse coverage must be 100%"):
        load_amlnet_dataset(_amlnet_config(path), tmp_path)


@pytest.mark.parametrize("dataset", ["transxion", "amlnet"])
def test_quick_sampling_occurs_after_full_temporal_processing_and_is_marked(tmp_path, dataset):
    if dataset == "transxion":
        paths = make_synthetic_transxion_test_fixture(tmp_path)
        config = _transxion_config(paths["transactions"], join_profiles=False)
    else:
        path = make_synthetic_amlnet_test_fixture(tmp_path)
        config = _amlnet_config(path)

    full = load_stress_dataset(config, tmp_path)
    quick = load_stress_dataset(config, tmp_path, quick_rows=80)
    assert len(quick.frame) == 80
    assert quick.frame["event_time"].min() == full.frame["event_time"].min()
    assert quick.frame["event_time"].max() == full.frame["event_time"].max()
    assert quick.manifest["population"]["rows"] == len(full.frame)
    assert quick.manifest["quick_sample"] == {
        "enabled": True,
        "requested_rows": 80,
        "sampling": "deterministic_even_spacing_after_full_temporal_processing",
        "full_time_span_retained": True,
        "not_for_thesis_claims": True,
    }
    assert quick.frame.attrs["quick_sample"]["not_for_thesis_claims"] is True


@pytest.mark.parametrize("dataset", ["transxion", "amlnet"])
def test_timestamp_safe_split_and_four_independent_validation_partitions(tmp_path, dataset):
    if dataset == "transxion":
        paths = make_synthetic_transxion_test_fixture(tmp_path)
        config = _transxion_config(paths["transactions"], join_profiles=False)
    else:
        path = make_synthetic_amlnet_test_fixture(tmp_path)
        config = _amlnet_config(path)
    bundle = load_stress_dataset(config, tmp_path)
    split = temporal_stress_split(bundle.frame, config)

    assert len(split.train) + len(split.validation) + len(split.test) == len(bundle.frame)
    assert (
        len(split.calibration_fit)
        + len(split.calibration_select)
        + len(split.rule_audit)
        + len(split.policy_select)
        == len(split.validation)
    )
    assert split.train["event_time"].max() < split.validation["event_time"].min()
    assert split.validation["event_time"].max() < split.test["event_time"].min()
    assert split.calibration_fit["event_time"].max() < split.calibration_select["event_time"].min()
    assert split.calibration_select["event_time"].max() < split.rule_audit["event_time"].min()
    assert split.rule_audit["event_time"].max() < split.policy_select["event_time"].min()
    for partition in (
        split.train,
        split.test,
        split.calibration_fit,
        split.calibration_select,
        split.rule_audit,
        split.policy_select,
    ):
        assert set(partition["label"].unique()) == {0, 1}
    assert split.manifest["both_classes_asserted"] is True


def test_temporal_split_rejects_a_partition_without_both_classes(tmp_path):
    path = make_synthetic_amlnet_test_fixture(tmp_path)
    config = _amlnet_config(path)
    frame = load_amlnet_dataset(config, tmp_path).frame.copy()
    frame["label"] = 0
    with pytest.raises(ValueError, match="must contain both target classes"):
        temporal_stress_split(frame, config)


def test_synthetic_fixtures_are_deterministic(tmp_path):
    first_tx = make_synthetic_transxion_test_fixture(tmp_path / "tx-a", seed=2026)
    second_tx = make_synthetic_transxion_test_fixture(tmp_path / "tx-b", seed=2026)
    assert file_checksum(first_tx["transactions"], "sha256") == file_checksum(
        second_tx["transactions"], "sha256"
    )

    first_aml = make_synthetic_amlnet_test_fixture(tmp_path / "aml-a", seed=2026)
    second_aml = make_synthetic_amlnet_test_fixture(tmp_path / "aml-b", seed=2026)
    assert file_checksum(first_aml, "md5") == file_checksum(second_aml, "md5")


def test_canonical_schema_is_enforced(tmp_path):
    paths = make_synthetic_transxion_test_fixture(tmp_path)
    raw = pd.read_csv(paths["transactions"])
    raw = raw.drop(columns="To Account")
    raw.to_csv(paths["transactions"], index=False)
    config = _transxion_config(paths["transactions"], join_profiles=False)
    with pytest.raises(KeyError, match="Required canonical columns"):
        load_transxion_dataset(config, tmp_path)
