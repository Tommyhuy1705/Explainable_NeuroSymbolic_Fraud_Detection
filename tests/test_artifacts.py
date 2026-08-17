import numpy as np

from src.artifacts import (
    assert_frozen_alignment,
    load_frozen_reference_artifact,
    stable_config_hash,
    write_frozen_reference_artifact,
)


def test_frozen_artifact_round_trip(tmp_path):
    config = {"dataset": {"name": "demo"}, "split": {"strategy": "temporal"}}
    y_validation = np.array([0, 1, 0, 1])
    y_test = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.2, 0.7])
    write_frozen_reference_artifact(
        tmp_path,
        manifest={"dataset_name": "demo", "config_sha256": stable_config_hash(config)},
        y_validation=y_validation,
        y_test=y_test,
        validation_probability=probabilities,
        test_probability=probabilities,
        validation_raw_probability=probabilities,
        test_raw_probability=probabilities,
    )
    artifact = load_frozen_reference_artifact(
        "demo", expected_config=config, search_roots=[tmp_path]
    )
    assert_frozen_alignment(artifact, y_validation, y_test)
    assert np.allclose(artifact["test_probability"], probabilities)
