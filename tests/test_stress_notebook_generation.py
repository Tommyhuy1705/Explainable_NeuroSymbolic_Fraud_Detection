from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat as nbf

from scripts.generate_stress_test_notebooks import build_all_stress_notebooks


ROOT = Path(__file__).resolve().parents[1]
LEGACY_NOTEBOOK_HASHES = {
    "01_Data_Exploration.ipynb": "e4b6761e41cc26a6b66eff0195ebb3724be26d3a82c09dfe3364a0c7e85e2ceb",
    "02_IEEE_CIS_Model_Benchmarks.ipynb": "dd0f3822a9808c5259ab8011ecc3350d2a740abc6be9bd9e567db9179f16c703",
    "03_BAF_Model_Benchmarks.ipynb": "602b27ef78e6df382b570e23a31206443d153e7512d3d43771783339d12e0931",
    "04_IEEE_CIS_LTN_Rule_Analysis.ipynb": "2a204a40c4da78e397b3d266354e63d29ef32314d3402179d889506f15d188a4",
    "05_IEEE_CIS_Rule_Explanation_Evaluation.ipynb": "1ee11d9466740a2f6c5164f4914e369d53d0798c6c50b391a46dfd70220d50a8",
    "06_IEEE_CIS_Rule_Ablation.ipynb": "b9071e23ef67f59a5f67db41818ffe13c4f85b0930c3cd612cdb91a9017f16dc",
    "07_BAF_LTN_Generalization.ipynb": "b018157856fe4a1fe42dd71b78e17c9720592739fa3fa6ab4f2b23139b0d03e5",
    "08_Cross_Dataset_Result_Synthesis.ipynb": "dba79011c5e853c87ca58adacc651a422c6a5a6b069c27048b3308d6b00abec0",
}


def _source_signature(notebook):
    return [(cell.cell_type, cell.source) for cell in notebook.cells]


def test_stress_notebook_sources_match_generator():
    for filename, expected in build_all_stress_notebooks().items():
        actual = nbf.read(ROOT / "notebooks" / filename, as_version=4)
        assert _source_signature(actual) == _source_signature(expected), filename


def test_stress_notebook_generation_is_byte_deterministic():
    first = build_all_stress_notebooks()
    second = build_all_stress_notebooks()
    assert first.keys() == second.keys()
    for filename in first:
        assert nbf.writes(first[filename]) == nbf.writes(second[filename]), filename


def test_stress_notebooks_validate_and_all_code_compiles():
    for filename, notebook in build_all_stress_notebooks().items():
        nbf.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{filename}:cell-{index}", "exec")


def test_dataset_notebooks_encode_proposal_guardrails():
    notebooks = build_all_stress_notebooks()
    for filename in ("09_TransXion_v2_Stress_Test.ipynb", "10_AMLNet_v1_0_Stress_Test.ipynb"):
        source = "\n".join(cell.source for cell in notebooks[filename].cells)
        for required in (
            "run_stress_test",
            "THESIS_STRESS_TEST_FIXTURE",
            "checksum_match",
            "locked_policies.json",
            "coverage_results.csv",
            "matched_risk_results.csv",
            "paired_bootstrap.csv",
            "negative_controls.csv",
            "ablation_results.csv",
            "ablation_rule_selection.csv",
            "ablation_rule_pool_provenance.json",
            "predictor_explanation_sensitivity.csv",
            "contrastive_meta_coefficients.csv",
            "predictor_contrastive_sensitivity.csv",
            "claim_eligible",
        ):
            assert required in source
        assert "—" not in source


def test_setup_validates_declared_ranges_without_replacing_torch():
    notebooks = build_all_stress_notebooks()
    for filename, notebook in notebooks.items():
        setup = notebook.cells[1].source
        assert 'PROJECT_ROOT / "requirements.txt"' in setup, filename
        assert "Requirement(requirement_text)" in setup, filename
        assert "installed_version not in requirement.specifier" in setup, filename
        assert 'package_name == "torch"' in setup, filename
        assert 'getattr(torch_module, "__version__"' in setup, filename
        assert "STRICT_RUNTIME_REQUIREMENTS = not (QUICK_RUN and USE_TEST_FIXTURE)" in setup, filename
        assert 'package_name != "torch" and status != "ok"' in setup, filename
        assert '"--upgrade"' in setup, filename
        assert '"--no-deps"' in setup, filename
        assert "will not install, upgrade, downgrade, or replace Torch" in setup, filename
        assert "RUNTIME_PACKAGE_VERSIONS" in setup, filename
        assert "RUNTIME_REQUIREMENTS_VALID" in setup, filename


def test_synthesis_only_consumes_lineage_validated_outputs():
    source = "\n".join(
        cell.source for cell in build_all_stress_notebooks()["11_Stress_Test_Synthesis.ipynb"].cells
    )
    assert "validate_stress_lineage" in source
    assert "run_stress_test(" not in source
    assert "model_rule_or_policy_reselection" in source
    assert "stress_outcomes.csv" in source
    assert "residual_evidence.csv" in source
    assert "ablation_results.csv" in source
    assert "core_equal_count_score_only_comparison" in source
    assert "shuffled_control_comparison_valid" in source
    assert "predictor_explanation_sensitivity.csv" in source
    assert "contrastive_meta_provenance.json" in source
    assert "contrastive_meta_available" in source
    assert "synthesis_lineage.json" in source
    assert "upstream_manifest_sha256" in source
    assert "CORE_ENVIRONMENT_PACKAGES" in source
    assert "upstream_core_environments" in source
    assert "serialized_environments" in source
    assert "Notebook 09/10 artifacts must use matching Python and core package versions" in source
    assert '"scikit-learn"' in source
    assert "matched_upstream_core_environment" in source
    assert "synthesis_runtime_core_environment" in source
    assert "Every Notebook 09/10 artifact must record a non-null Git commit" in source
    assert "Notebook 09/10 artifacts must use one non-null executable source fingerprint" in source
    assert "Notebook 11 source commit must match" not in source
    assert "primary_stress_test_summary.csv" in source
    assert "stress_test_summary.png" in source
    assert "score_only_requested_budget_precision" in source
    assert "supported_alert_rate" in source
    assert "requested_explanation_coverage" in source
    assert "all_primary_abstained" in source
    assert "Saturation không được xác nhận" in source


def test_generator_does_not_change_executed_notebooks_01_to_08():
    for filename, expected in LEGACY_NOTEBOOK_HASHES.items():
        actual = hashlib.sha256((ROOT / "notebooks" / filename).read_bytes()).hexdigest()
        assert actual == expected, filename
