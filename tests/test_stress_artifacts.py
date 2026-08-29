from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.artifacts import sha256_file
from src.stress_testing.artifacts import (
    stress_pipeline_fingerprint,
    stress_source_fingerprint,
    validate_stress_lineage,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_stress_json_is_strict_and_nonfinite_metrics_become_null(tmp_path):
    destination = write_json(
        tmp_path / "metrics.json",
        {"defined": 0.5, "undefined": float("nan"), "infinite": float("inf")},
    )
    raw = destination.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    assert json.loads(raw) == {"defined": 0.5, "undefined": None, "infinite": None}


def test_stress_lineage_detects_output_tampering(tmp_path):
    output = tmp_path / "coverage_results.csv"
    output.write_text("coverage,precision\n0.1,0.5\n", encoding="utf-8")
    lineage = {
        "lineage_schema_version": 1,
        "notebook_id": "fixture",
        "outputs": {output.name: sha256_file(output)},
    }
    lineage_path = write_json(tmp_path / "stress_lineage.json", lineage)
    validated = validate_stress_lineage(lineage_path)
    assert validated["notebook_id"] == "fixture"

    output.write_text("coverage,precision\n0.1,0.9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_stress_lineage(lineage_path)


def test_source_fingerprint_is_config_independent_but_pipeline_fingerprint_is_not():
    source = stress_source_fingerprint(PROJECT_ROOT)
    assert len(source) == 64
    transxion = stress_pipeline_fingerprint(
        PROJECT_ROOT, PROJECT_ROOT / "configs/stress/transxion_v2.yaml"
    )
    amlnet = stress_pipeline_fingerprint(
        PROJECT_ROOT, PROJECT_ROOT / "configs/stress/amlnet_v1.yaml"
    )
    assert transxion != amlnet
