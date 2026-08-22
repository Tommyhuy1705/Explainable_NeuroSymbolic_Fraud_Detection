from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from scripts.generate_notebooks import (
    _same_canonical_content,
    _preserve_execution_when_source_is_unchanged,
    build_all_notebooks,
)


ROOT = Path(__file__).resolve().parents[1]


def _source_signature(notebook):
    return [(cell.cell_type, cell.source) for cell in notebook.cells]


def test_committed_notebook_sources_match_generator():
    for filename, expected in build_all_notebooks().items():
        actual = nbf.read(ROOT / "notebooks" / filename, as_version=4)
        assert _source_signature(actual) == _source_signature(expected), filename


def test_every_generated_code_cell_compiles_and_notebook_validates():
    for filename, notebook in build_all_notebooks().items():
        nbf.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{filename}:cell-{index}", "exec")


def test_frozen_preflight_is_generated_for_every_downstream_notebook():
    notebooks = build_all_notebooks()
    for filename in (
        "05_IEEE_CIS_Rule_Explanation_Evaluation.ipynb",
        "06_IEEE_CIS_Rule_Ablation.ipynb",
        "07_BAF_LTN_Generalization.ipynb",
    ):
        source = "\n".join(cell.source for cell in notebooks[filename].cells)
        assert "preflight_manifests" in source
        assert "for root_value in INPUT_ROOTS" in source
        assert "Full thesis evaluation cannot consume a synthetic-fallback frozen artifact" in source


def test_synthesis_uses_manifest_selection_and_post_hoc_claim_boundary():
    source = "\n".join(
        cell.source for cell in build_all_notebooks()["08_Cross_Dataset_Result_Synthesis.ipynb"].cells
    )
    assert 'ieee_artifact["manifest"]["model_key"]' in source
    assert 'baf_artifact["manifest"]["model_key"]' in source
    assert "sort_values([\"dataset\", \"raw_pr_auc_mean\"]" not in source
    assert "post-hoc" in source
    assert "frozen_manifest_sha256" in source


def test_executed_outputs_are_preserved_only_for_identical_source(tmp_path):
    existing = nbf.v4.new_notebook(
        cells=[nbf.v4.new_code_cell("value = 1", execution_count=1, outputs=[nbf.v4.new_output("execute_result", data={"text/plain": "1"}, execution_count=1)])]
    )
    path = tmp_path / "same.ipynb"
    nbf.write(existing, path)

    same = nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell("value = 1")])
    assert _preserve_execution_when_source_is_unchanged(same, path)
    assert same.cells[0].execution_count == 1
    assert same.cells[0].outputs

    changed = nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell("value = 2")])
    assert not _preserve_execution_when_source_is_unchanged(changed, path)
    assert changed.cells[0].execution_count is None
    assert not changed.cells[0].outputs


def test_canonical_comparison_ignores_execution_and_cell_ids():
    generated = nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell("value = 1")])
    executed = nbf.v4.new_notebook(
        cells=[
            nbf.v4.new_code_cell(
                "value = 1",
                execution_count=4,
                outputs=[
                    nbf.v4.new_output(
                        "execute_result",
                        data={"text/plain": "1"},
                        execution_count=4,
                    )
                ],
            )
        ]
    )
    generated.cells[0]["id"] = "generated-id"
    executed.cells[0]["id"] = "executed-id"
    executed.cells[0].metadata["tags"] = ["papermill-derived"]

    assert _same_canonical_content(generated, executed)
    executed.cells[0].source = "value = 2"
    assert not _same_canonical_content(generated, executed)
