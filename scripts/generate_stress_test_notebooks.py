"""Generate proposal-aligned stress-test notebooks with ``nbformat``.

Notebook 09 and 10 each execute one complete, independent dataset protocol.
Notebook 11 consumes only their checksummed artifacts.  Keeping this generator
separate from ``generate_notebooks.py`` preserves the executed Notebook 01-08
source lineage.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


SETUP = r'''
from pathlib import Path
from importlib import metadata as importlib_metadata
import importlib
import importlib.util
import json
import os
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

KAGGLE = Path("/kaggle").exists()
REPO_URL = "https://github.com/Tommyhuy1705/Explainable_NeuroSymbolic_Fraud_Detection.git"
BRANCH = "main"
KAGGLE_PROJECT_DIR = Path("/kaggle/working/Explainable_NeuroSymbolic_Fraud_Detection")

def sync_project() -> Path:
    if KAGGLE:
        if not KAGGLE_PROJECT_DIR.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(KAGGLE_PROJECT_DIR)],
                check=True,
            )
        elif not (KAGGLE_PROJECT_DIR / ".git").is_dir():
            raise RuntimeError(f"Expected a Git clone at {KAGGLE_PROJECT_DIR}")
        else:
            subprocess.run(
                ["git", "-C", str(KAGGLE_PROJECT_DIR), "pull", "--ff-only", "origin", BRANCH],
                check=True,
            )
        return KAGGLE_PROJECT_DIR.resolve()
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Run this notebook inside the project or on Kaggle with Internet enabled")

PROJECT_ROOT = sync_project()
while str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# A Kaggle kernel may survive a previous run. Never execute a stale src module.
for module_name in tuple(sys.modules):
    if module_name == "src" or module_name.startswith("src."):
        del sys.modules[module_name]

QUICK_RUN = os.getenv("THESIS_STRESS_QUICK_RUN", "0") == "1"
USE_TEST_FIXTURE = os.getenv("THESIS_STRESS_TEST_FIXTURE", "0") == "1"
if USE_TEST_FIXTURE and not QUICK_RUN:
    raise ValueError("THESIS_STRESS_TEST_FIXTURE=1 requires THESIS_STRESS_QUICK_RUN=1")
STRICT_RUNTIME_REQUIREMENTS = not (QUICK_RUN and USE_TEST_FIXTURE)

OUTPUT_BASE = (
    Path("/kaggle/working/thesis_stress_outputs")
    if KAGGLE else PROJECT_ROOT / "results/stress/notebooks"
)
DATA_ROOTS = [Path("/kaggle/input")] if KAGGLE else [PROJECT_ROOT / "data/raw"]

RUNTIME_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "pyyaml": "yaml",
    "joblib": "joblib",
    "torch": "torch",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
}

declared_requirements = {}
for raw_line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
    requirement_text = raw_line.split("#", maxsplit=1)[0].strip()
    if not requirement_text:
        continue
    requirement = Requirement(requirement_text)
    normalized_name = canonicalize_name(requirement.name)
    if normalized_name in RUNTIME_IMPORTS:
        declared_requirements[normalized_name] = requirement
missing_declarations = sorted(set(RUNTIME_IMPORTS).difference(declared_requirements))
if missing_declarations:
    raise RuntimeError(
        "requirements.txt has no declared range for runtime packages: "
        + ", ".join(missing_declarations)
    )

def runtime_package_status(package_name):
    requirement = declared_requirements[package_name]
    import_name = RUNTIME_IMPORTS[package_name]
    if importlib.util.find_spec(import_name) is None:
        return "missing", None, requirement
    try:
        installed_version = importlib_metadata.version(requirement.name)
    except importlib_metadata.PackageNotFoundError:
        installed_version = None
    # Vendor-managed Torch images can expose a fully usable import while the
    # distribution metadata has no Version field. Torch is never repaired in
    # this notebook, so reading its runtime version cannot create a stale
    # import after pip installation.
    if not installed_version and package_name == "torch":
        torch_module = importlib.import_module(import_name)
        installed_version = str(getattr(torch_module, "__version__", "")).strip() or None
    if not installed_version:
        return "metadata_missing", None, requirement
    if installed_version not in requirement.specifier:
        return "out_of_range", installed_version, requirement
    return "ok", installed_version, requirement

runtime_status = {
    package_name: runtime_package_status(package_name)
    for package_name in RUNTIME_IMPORTS
}
torch_status, torch_version, torch_requirement = runtime_status["torch"]
if torch_status != "ok" and STRICT_RUNTIME_REQUIREMENTS:
    observed = torch_version if torch_version is not None else torch_status
    raise ImportError(
        f"PyTorch environment is incompatible: observed {observed!r}, required "
        f"{torch_requirement}. Select a Kaggle image whose preinstalled Torch/CUDA wheel "
        "satisfies this range; this notebook will not install, upgrade, downgrade, or replace Torch."
    )

repair_requirements = [
    str(requirement)
    for package_name, (status, _, requirement) in runtime_status.items()
    if package_name != "torch" and status != "ok"
]
if repair_requirements and KAGGLE and STRICT_RUNTIME_REQUIREMENTS:
    # --no-deps ensures this repair cannot replace Kaggle's preinstalled
    # Torch/CUDA wheel indirectly through dependency resolution.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            "--no-deps",
            *repair_requirements,
        ],
        check=True,
    )
    importlib.invalidate_caches()
    runtime_status = {
        package_name: runtime_package_status(package_name)
        for package_name in RUNTIME_IMPORTS
    }

invalid_packages = {
    package_name: {
        "status": status,
        "installed_version": installed_version,
        "required": str(requirement),
    }
    for package_name, (status, installed_version, requirement) in runtime_status.items()
    if status != "ok"
}
RUNTIME_REQUIREMENTS_VALID = not invalid_packages
if invalid_packages and STRICT_RUNTIME_REQUIREMENTS:
    raise ImportError(
        "Runtime packages do not satisfy requirements.txt: "
        + json.dumps(invalid_packages, sort_keys=True)
    )
RUNTIME_PACKAGE_VERSIONS = {
    declared_requirements[package_name].name: installed_version
    for package_name, (_, installed_version, _) in runtime_status.items()
}
if invalid_packages and not STRICT_RUNTIME_REQUIREMENTS:
    print(
        "Engineering fixture warning: runtime packages are outside the declared "
        "ranges. This run is claim-ineligible by construction: "
        + json.dumps(invalid_packages, sort_keys=True)
    )

try:
    GIT_COMMIT = subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
except (OSError, subprocess.CalledProcessError):
    GIT_COMMIT = None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_colwidth", 160)
print({
    "project_root": str(PROJECT_ROOT),
    "git_commit": GIT_COMMIT,
    "quick_run": QUICK_RUN,
    "test_fixture": USE_TEST_FIXTURE,
    "claim_eligible_mode": not QUICK_RUN and not USE_TEST_FIXTURE,
    "strict_runtime_requirements": STRICT_RUNTIME_REQUIREMENTS,
    "runtime_requirements_valid": RUNTIME_REQUIREMENTS_VALID,
    "runtime_package_versions": RUNTIME_PACKAGE_VERSIONS,
})
'''


def _notebook(cells: list, name: str):
    for index, cell in enumerate(cells):
        identity = f"{name}\0{index}\0{cell.cell_type}\0{cell.source}".encode("utf-8")
        cell["id"] = hashlib.sha256(identity).hexdigest()[:16]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "stress_test_notebook": name,
        }
    )
    return notebook


def build_dataset_notebook(
    *,
    notebook_id: str,
    title: str,
    dataset_key: str,
    config_relative: str,
    source_filename: str,
    source_note: str,
    fixture_function: str,
) -> nbf.NotebookNode:
    output_slug = notebook_id.lower()
    cells = [
        md(
            f'''
            # {title}

            Notebook này thực hiện stress test độc lập cho **{dataset_key}** theo proposal khóa luận.
            Dataset này không thay thế hai benchmark chính IEEE-CIS và BAF. Vai trò của nó là kiểm tra
            giới hạn của lớp giải thích chọn lọc trong một bối cảnh AML bổ sung.

            Giao thức bắt buộc: time split 60/20/20; predictor, calibrator và threshold được khóa trước
            explanation; rule audit và policy selection dùng các phần validation riêng; test chỉ được
            đánh giá sau khi policy đã khóa. Fixed coverage là 5%, 10%, 25% và 50%, trong đó 10% và 25%
            là kết quả chính. Mọi phương pháp phải so với predictor-score-only ở cùng ngân sách.

            {source_note}

            Kết quả quick/fixture chỉ kiểm tra code và không được dùng làm bằng chứng khóa luận.
            '''
        ),
        code(SETUP),
        md(
            '''
            ## 1. Chế độ chạy và accelerator

            Full mode là mặc định. Trên Kaggle nên chọn GPU T4. Bộ giải quyết device sẽ kiểm tra CUDA
            compute capability; nếu wheel không tương thích (thường gặp với P100/sm_60), neural model
            chuyển về CPU và ghi rõ fallback trong provenance thay vì phát sinh lỗi kernel khó chẩn đoán.
            '''
        ),
        code(
            f'''
            from src.stress_testing.artifacts import load_yaml_config
            from src.stress_testing.predictors import resolve_safe_torch_device

            CONFIG_PATH = PROJECT_ROOT / {config_relative!r}
            CONFIG = load_yaml_config(CONFIG_PATH)
            OUTPUT_DIR = OUTPUT_BASE / {output_slug!r}
            DEVICE = resolve_safe_torch_device("cuda")
            print({{
                "config": str(CONFIG_PATH),
                "output_dir": str(OUTPUT_DIR),
                "requested_device": DEVICE.requested,
                "selected_device": DEVICE.selected,
                "device_reason": DEVICE.reason,
                "dataset_role": CONFIG["project"]["claim_scope"],
                "coverage_budgets": CONFIG["evaluation"]["coverages"],
            }})
            '''
        ),
        md(
            '''
            ## 2. Input discovery, identity and checksum preflight

            Full run phải tìm đúng một file có tên chính xác trong các input roots. Loader sẽ từ chối
            Git-LFS pointer, nhiều file trùng tên, schema sai, target không nhị phân và checksum sai.
            Checksum giúp định danh release; filename hoặc số dòng riêng lẻ không đủ làm provenance.
            '''
        ),
        code(
            f'''
            from src.stress_testing.data import (
                {fixture_function}, file_checksum, resolve_stress_data_file
            )

            if USE_TEST_FIXTURE:
                # Use the exact directory/row contract consumed by
                # run_stress_test so preflight and execution identify one file.
                FIXTURE_ROOT = OUTPUT_DIR / ".stress_test_fixture"
                fixture_rows = int(CONFIG.get("testing", {{}}).get("fixture_rows", 800))
                fixture_result = {fixture_function}(
                    FIXTURE_ROOT,
                    n_rows=fixture_rows,
                    seed=int(CONFIG["project"].get("seed", 42)),
                )
                DATA_ROOTS = [FIXTURE_ROOT]

            source_path = resolve_stress_data_file({source_filename!r}, DATA_ROOTS)
            checksum_config = CONFIG["dataset"]["checksum"]
            observed_checksum = file_checksum(source_path, checksum_config["algorithm"])
            preflight = {{
                "source_path": str(source_path),
                "bytes": source_path.stat().st_size,
                "checksum_algorithm": checksum_config["algorithm"],
                "observed_checksum": observed_checksum,
                "expected_checksum": checksum_config["value"],
                "checksum_match": observed_checksum.lower() == checksum_config["value"].lower(),
                "verification_required": not USE_TEST_FIXTURE,
            }}
            display(preflight)
            if not USE_TEST_FIXTURE and not preflight["checksum_match"]:
                raise ValueError("The attached input is not the proposal-pinned dataset release")
            '''
        ),
        md(
            '''
            ## 3. Execute the locked stress-test state machine

            `run_stress_test` thực hiện tuần tự các bước sau:

            1. Chuẩn hóa schema, dựng đặc trưng lịch sử chỉ từ các giao dịch quá khứ và kiểm tra leakage denylist.
            2. Split thời gian 60/20/20; validation được chia tiếp cho calibration, model selection, rule audit và policy selection.
            3. Train XGBoost, LightGBM và TabularResNetV2 với xử lý class imbalance.
            4. Chọn calibrator, F2 threshold và reference predictor chỉ bằng validation rồi đóng băng chúng.
            5. Fit Tier A/B candidates trên train; giữ Tier C làm baseline tách biệt; audit attribution, TP-FP, lift, stability và redundancy chỉ trong frozen-predictor alert region của validation.
            6. Fit contrastive meta-scorer từ audited rule truth và frozen calibrated risk chỉ trên TP/FP alerts của `rule_audit`; không fit lại trên `policy_select` hoặc test.
            7. Chọn individual method hoặc pre-registered guarded ensemble, hoặc abstain, ở từng coverage bằng score-only và stability guardrails.
            8. Sau khi policy đã khóa mới tính test metrics, matched-risk, residual TP-FP và paired block bootstrap.
            9. Chạy negative controls; phân loại kết quả dương, âm, abstention hoặc saturation; ghi checksum/lineage.
            '''
        ),
        code(
            '''
            from src.stress_testing.experiment import run_stress_test

            run_result = run_stress_test(
                CONFIG_PATH,
                data_roots=DATA_ROOTS,
                output_dir=OUTPUT_DIR,
                quick_run=QUICK_RUN,
                use_test_fixture=USE_TEST_FIXTURE,
                device=DEVICE.selected,
            )
            print(run_result)
            '''
        ),
        md("## 4. Data quality, provenance and temporal integrity"),
        code(
            '''
            def read_json(name):
                return json.loads((OUTPUT_DIR / name).read_text(encoding="utf-8"))

            data_manifest = read_json("data_manifest.json")
            split_integrity = pd.read_csv(OUTPUT_DIR / "split_integrity.csv")
            display(pd.DataFrame(data_manifest["source_files"]))
            display(pd.DataFrame(data_manifest["data_quality"].items(), columns=["check", "value"]))
            display(split_integrity)

            assert data_manifest["data_quality"]["timestamp_parse_coverage"] == 1.0
            assert data_manifest["data_quality"]["target_missing_count"] == 0
            assert split_integrity["both_classes"].all()
            assert split_integrity["chronologically_disjoint"].all()
            '''
        ),
        md(
            '''
            ## 5. Predictive benchmark and frozen reference predictor

            PR-AUC là metric chọn predictor chính do class imbalance. ROC-AUC, F2, precision, recall,
            Recall@1% FPR, Brier, ECE và NLL được báo cùng prevalence/null baseline. Việc model có metric
            cao không tự động chứng minh explanation có incremental value.
            '''
        ),
        code(
            '''
            predictive = pd.read_csv(OUTPUT_DIR / "predictive_metrics.csv")
            calibration = pd.read_csv(OUTPUT_DIR / "calibration_comparison.csv")
            predictor_manifest = read_json("predictor_manifest.json")
            display(predictive.sort_values(["selection_split_pr_auc", "family"], ascending=[False, True]))
            display(calibration)
            display({key: predictor_manifest.get(key) for key in [
                "reference_family", "reference_seed", "threshold", "calibration_method", "frozen_before_explanation"
            ]})
            assert predictor_manifest["frozen_before_explanation"] is True
            '''
        ),
        md("## 6. Candidate registry, validation-only audit and deduplication"),
        code(
            '''
            registry = pd.read_csv(OUTPUT_DIR / "rule_registry.csv")
            audit = pd.read_csv(OUTPUT_DIR / "rule_audit.csv")
            redundancy = pd.read_csv(OUTPUT_DIR / "rule_redundancy.csv")
            counterfactual = pd.read_csv(OUTPUT_DIR / "counterfactual_diagnostics.csv")
            candidate_provenance = read_json("candidate_generation_provenance.json")
            surrogate_rules = pd.read_csv(OUTPUT_DIR / "surrogate_rules.csv")
            surrogate_provenance = read_json("surrogate_provenance.json")
            ablation_validation = pd.read_csv(OUTPUT_DIR / "ablation_validation_results.csv")
            ablation_stability = pd.read_csv(OUTPUT_DIR / "ablation_validation_stability.csv")
            ablation_rule_selection = pd.read_csv(OUTPUT_DIR / "ablation_rule_selection.csv")
            ablation_rule_pool_provenance = read_json("ablation_rule_pool_provenance.json")
            ablation = pd.read_csv(OUTPUT_DIR / "ablation_results.csv")
            predictor_sensitivity = pd.read_csv(
                OUTPUT_DIR / "predictor_explanation_sensitivity.csv"
            )
            predictor_attribution = pd.read_csv(
                OUTPUT_DIR / "predictor_attribution_sensitivity.csv"
            )
            contrastive_coefficients = pd.read_csv(
                OUTPUT_DIR / "contrastive_meta_coefficients.csv"
            )
            contrastive_provenance = read_json("contrastive_meta_provenance.json")
            predictor_contrastive = pd.read_csv(
                OUTPUT_DIR / "predictor_contrastive_sensitivity.csv"
            )
            predictor_contrastive_provenance = read_json(
                "predictor_contrastive_sensitivity_provenance.json"
            )
            display(registry.groupby(["tier", "source"], dropna=False).size().rename("candidate_count"))
            display(counterfactual)
            display(candidate_provenance)
            display(surrogate_rules)
            display(surrogate_provenance)
            display(audit.sort_values(["selected", "audit_alert_score"], ascending=[False, False]))
            display(redundancy.head(30))
            display(ablation_validation[ablation_validation["coverage"].round(2).isin([0.10, 0.25])])
            display(ablation_stability[ablation_stability["coverage"].round(2).isin([0.10, 0.25])])
            display(ablation_rule_selection)
            display(ablation_rule_pool_provenance)
            display(ablation[ablation["coverage_budget"].round(2).isin([0.10, 0.25])])
            display(predictor_sensitivity[
                predictor_sensitivity["coverage_budget"].round(2).isin([0.10, 0.25])
            ])
            display(predictor_attribution.head(30))
            display(contrastive_provenance)
            display(contrastive_coefficients)
            display(predictor_contrastive.head(30))
            display(predictor_contrastive_provenance)
            '''
        ),
        md(
            '''
            ## 7. Locked selective policies and fixed-coverage results

            Một coverage result chỉ được diễn giải là lợi ích bổ sung nếu nó vượt predictor-score-only
            ở đúng cùng số lượng alert thực tế được chọn. Nếu rule support trên test không đủ requested
            budget, VASRE được phép underfill và comparator chính cũng bị khóa về realized count; bảng vẫn
            giữ requested-budget score-only như diagnostic riêng. Nếu validation guardrail không đạt, hệ
            thống phải abstain; đây là kết quả hợp lệ, không phải lỗi pipeline.
            '''
        ),
        code(
            '''
            policies = pd.DataFrame(read_json("locked_policies.json"))
            candidates = pd.read_csv(OUTPUT_DIR / "policy_candidates_validation.csv")
            coverage = pd.read_csv(OUTPUT_DIR / "coverage_results.csv")
            display(policies)
            display(coverage)

            plot_frame = coverage.copy()
            fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
            ax.plot(plot_frame["coverage_budget"] * 100, plot_frame["selected_alert_precision"], marker="o", label="VASRE")
            ax.plot(plot_frame["coverage_budget"] * 100, plot_frame["score_only_precision"], marker="s", label="Predictor score only")
            ax.axhline(plot_frame["all_alert_precision"].iloc[0], color="grey", linestyle=":", label="All alerts")
            ax.set(xlabel="Explanation coverage among alerts (%)", ylabel="Fraud precision", title="Fixed-budget alert triage")
            ax.legend(loc="best")
            plt.show()
            '''
        ),
        md("## 8. Matched-risk, residual evidence and paired uncertainty"),
        code(
            '''
            matched = pd.read_csv(OUTPUT_DIR / "matched_risk_results.csv")
            residual = pd.read_csv(OUTPUT_DIR / "residual_evidence.csv")
            bootstrap = pd.read_csv(OUTPUT_DIR / "paired_bootstrap.csv")
            outcomes = pd.read_csv(OUTPUT_DIR / "stress_outcomes.csv")
            display(matched)
            display(residual)
            display(bootstrap)
            display(outcomes)

            primary = coverage[coverage["coverage_budget"].round(2).isin([0.10, 0.25])]
            if len(primary):
                fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
                sns.barplot(data=primary, x="coverage_budget", y="delta_vs_score_only", ax=axes[0], color="#4c72b0")
                axes[0].axhline(0, color="black", linewidth=1)
                axes[0].set(title="Incremental precision vs score only", xlabel="Coverage", ylabel="Precision difference")
                primary_outcomes = outcomes[outcomes["coverage_budget"].round(2).isin([0.10, 0.25])]
                sns.barplot(data=primary_outcomes, x="coverage_budget", y="risk_conditioned_tp_fp_gap", ax=axes[1], color="#55a868")
                axes[1].axhline(0, color="black", linewidth=1)
                axes[1].set(title="Risk-conditioned residual TP-FP evidence", xlabel="Coverage", ylabel="Evidence gap")
                plt.show()
            '''
        ),
        md("## 9. Stability, negative controls, saturation and claim boundary"),
        code(
            '''
            stability = pd.read_csv(OUTPUT_DIR / "validation_stability.csv")
            rule_stability = pd.read_csv(OUTPUT_DIR / "rule_audit_stability.csv")
            controls = pd.read_csv(OUTPUT_DIR / "negative_controls.csv")
            summary = read_json("stress_summary.json")
            manifest = read_json("stress_test_manifest.json")
            lineage = read_json("stress_lineage.json")
            display(stability)
            display(rule_stability)
            display(controls)
            display(summary)
            display({
                "claim_eligible": manifest["claim_eligible"],
                "claim_blockers": manifest["claim_blockers"],
                "stress_pipeline_fingerprint": manifest["stress_pipeline_fingerprint"],
                "lineage_outputs": len(lineage["outputs"]),
            })

            if QUICK_RUN or USE_TEST_FIXTURE:
                assert manifest["claim_eligible"] is False
            '''
        ),
        md(
            '''
            ## 10. Diễn giải kết quả đúng phạm vi

            - Kết quả dương chỉ được xác nhận khi paired bootstrap, matched-risk, residual evidence,
              validation stability và negative controls cùng hỗ trợ; label-shuffle và weight-shuffle
              validation controls là hai cổng bắt buộc. Một CI dương riêng lẻ là chưa đủ.
            - Domain Tier A only, exact counterfactual-only và all Tier B tự deduplicate/select/weight từ
              complete pre-dedup audit pool; chúng không chỉ lọc các rule đã thắng trong primary A/B pool.
            - Kết quả âm hoặc abstention cho thấy rule evidence chưa tạo incremental value trong protocol này.
            - Rule audit, TP-FP metrics, weighting và deduplication chỉ dùng frozen-predictor alerts; TN/FN ngoài alert region chỉ xuất hiện trong population diagnostics.
            - Counterfactual-only là đúng tập rule có source `train_model_counterfactual`; `train_derived_tier_b_only` rộng hơn và được báo riêng.
            - Tier C CART và signed native-attribution top-k là baseline chẩn đoán, không được nhập vào primary A/B policy.
            - Contrastive meta-scorer dùng audited rule truth cùng frozen calibrated risk và chỉ fit trên TP/FP alerts của `rule_audit`; fallback thiếu mẫu/lớp được báo rõ và không được âm thầm fit trên test.
            - Guarded ensemble là trung bình của các component đã khóa trong YAML, sau đó vẫn phải qua score-only và resampling-stability guardrails trên `policy_select`.
            - Saturation chỉ được xác nhận cho AMLNet full/checksummed run khi đủ positive, matched pairs,
              bootstrap replicates và toàn bộ lineage/attribution guardrails; TransXion không mang nhãn này.
            - Matched-risk và residual TP-FP làm giảm một số confounding theo risk score nhưng không chứng minh nhân quả.
            - Stress test không chứng minh production generalization hoặc hiệu quả với chuyên viên AML thực tế.

            Hãy lưu một Kaggle version hoàn tất để bảo toàn toàn bộ files trong `OUTPUT_DIR`. Notebook 11
            chỉ tổng hợp version outputs này và không được chọn lại model, rule hoặc policy.
            '''
        ),
    ]
    return _notebook(cells, notebook_id)


def build_synthesis_notebook() -> nbf.NotebookNode:
    cells = [
        md(
            '''
            # 11 - TransXion and AMLNet Stress-Test Synthesis

            Notebook này tổng hợp hậu nghiệm hai stress test đã khóa. Nó không train model, không audit
            lại rule và không chọn lại policy. Hãy Add Input hai Kaggle version outputs của Notebook 09
            và 10. Mọi artifact phải qua checksum lineage trước khi được đưa vào bảng khóa luận.
            '''
        ),
        code(SETUP),
        md("## 1. Discover exactly one valid artifact for each stress dataset"),
        code(
            '''
            from src.stress_testing.artifacts import find_unique_stress_manifest, validate_stress_lineage

            SEARCH_ROOTS = [Path("/kaggle/input")] if KAGGLE else [OUTPUT_BASE]
            manifest_paths = {
                "transxion_v2": find_unique_stress_manifest("transxion_v2", SEARCH_ROOTS),
                "amlnet_v1_0": find_unique_stress_manifest("amlnet_v1_0", SEARCH_ROOTS),
            }
            manifests = {}
            for dataset, manifest_path in manifest_paths.items():
                lineage_path = manifest_path.with_name("stress_lineage.json")
                validate_stress_lineage(lineage_path, expected_dataset_manifest=manifest_path)
                manifests[dataset] = json.loads(manifest_path.read_text(encoding="utf-8"))
                print(dataset, manifest_path, "lineage=valid")
            '''
        ),
        md("## 2. Claim eligibility and source identity gate"),
        code(
            '''
            CORE_ENVIRONMENT_PACKAGES = (
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "xgboost",
                "lightgbm",
                "torch",
            )

            def manifest_core_environment(payload):
                environment = payload.get("environment")
                if not isinstance(environment, dict):
                    raise ValueError("Stress manifest is missing its captured environment")
                python_runtime = str(environment.get("python") or "").strip()
                packages = environment.get("packages")
                if not python_runtime or not isinstance(packages, dict):
                    raise ValueError(
                        "Stress manifest must record Python and core package versions"
                    )
                signature = {"python": python_runtime.split()[0]}
                for package_name in CORE_ENVIRONMENT_PACKAGES:
                    version = packages.get(package_name)
                    if version is None or not str(version).strip():
                        raise ValueError(
                            f"Stress manifest is missing version metadata for {package_name}"
                        )
                    signature[package_name] = str(version).strip()
                return signature

            upstream_core_environments = {
                dataset: manifest_core_environment(payload)
                for dataset, payload in manifests.items()
            }
            serialized_environments = {
                json.dumps(environment, sort_keys=True)
                for environment in upstream_core_environments.values()
            }
            if len(serialized_environments) != 1:
                raise ValueError(
                    "Notebook 09/10 artifacts must use matching Python and core package versions: "
                    + json.dumps(upstream_core_environments, sort_keys=True)
                )
            MATCHED_UPSTREAM_CORE_ENVIRONMENT = next(
                iter(upstream_core_environments.values())
            )
            CURRENT_CORE_ENVIRONMENT = {
                "python": sys.version.split()[0],
                **{
                    package_name: RUNTIME_PACKAGE_VERSIONS[package_name]
                    for package_name in CORE_ENVIRONMENT_PACKAGES
                },
            }

            eligibility = pd.DataFrame([
                {
                    "dataset": dataset,
                    "claim_eligible": payload["claim_eligible"],
                    "claim_blockers": ";".join(payload["claim_blockers"]),
                    "git_commit": payload["git_commit"],
                    "pipeline_fingerprint": payload["stress_pipeline_fingerprint"],
                    "source_fingerprint": payload["stress_source_fingerprint"],
                    "core_environment": json.dumps(
                        upstream_core_environments[dataset], sort_keys=True
                    ),
                    "source_checksum_verified": all(
                        record.get("verified", False)
                        for record in payload["data_manifest"]["source_files"]
                        if record.get("role", "transactions") != "profile"
                    ),
                }
                for dataset, payload in manifests.items()
            ])
            display(eligibility)
            if not eligibility["claim_eligible"].all() and not (QUICK_RUN and USE_TEST_FIXTURE):
                raise ValueError("At least one stress artifact is not eligible for thesis claims")
            if not eligibility["claim_eligible"].all():
                display(Markdown(
                    "**Engineering smoke only:** upstream fixture/quick artifacts are intentionally "
                    "ineligible and this synthesis must not be used in the thesis results."
                ))
            commits = set(eligibility["git_commit"].dropna().astype(str))
            source_fingerprints = set(eligibility["source_fingerprint"].dropna().astype(str))
            if len(commits) != 1 or len(source_fingerprints) != 1:
                raise ValueError(
                    "Stress artifacts must use the same non-null Git commit and executable source fingerprint"
                )
            if GIT_COMMIT not in commits:
                raise ValueError(
                    "Notebook 11 source commit must match the upstream Notebook 09/10 artifacts"
                )
            '''
        ),
        md("## 3. Read locked outputs without re-selection"),
        code(
            '''
            def read_table(dataset, filename):
                return pd.read_csv(manifest_paths[dataset].parent / filename).assign(dataset=dataset)

            def read_json_artifact(dataset, filename):
                return json.loads(
                    (manifest_paths[dataset].parent / filename).read_text(encoding="utf-8")
                )

            coverage = pd.concat(
                [read_table(dataset, "coverage_results.csv") for dataset in manifest_paths],
                ignore_index=True,
            )
            predictive = pd.concat(
                [read_table(dataset, "predictive_metrics.csv") for dataset in manifest_paths],
                ignore_index=True,
            )
            matched = pd.concat(
                [read_table(dataset, "matched_risk_results.csv") for dataset in manifest_paths],
                ignore_index=True,
            )
            bootstrap = pd.concat(
                [read_table(dataset, "paired_bootstrap.csv") for dataset in manifest_paths],
                ignore_index=True,
            )
            controls = pd.concat(
                [read_table(dataset, "negative_controls.csv") for dataset in manifest_paths],
                ignore_index=True,
            )
            outcomes = pd.concat(
                [read_table(dataset, "stress_outcomes.csv") for dataset in manifest_paths],
                ignore_index=True,
            )
            residual = pd.concat(
                [read_table(dataset, "residual_evidence.csv") for dataset in manifest_paths],
                ignore_index=True,
            )
            ablations = pd.concat(
                [read_table(dataset, "ablation_results.csv") for dataset in manifest_paths],
                ignore_index=True,
            )
            predictor_sensitivity = pd.concat(
                [read_table(dataset, "predictor_explanation_sensitivity.csv") for dataset in manifest_paths],
                ignore_index=True,
            )
            contrastive_status = pd.DataFrame([
                {
                    "dataset": dataset,
                    "contrastive_meta_available": bool(
                        read_json_artifact(dataset, "contrastive_meta_provenance.json")["available"]
                    ),
                    "contrastive_meta_fallback_reason": read_json_artifact(
                        dataset, "contrastive_meta_provenance.json"
                    ).get("fallback_reason"),
                    "contrastive_meta_fit_partition": read_json_artifact(
                        dataset, "contrastive_meta_provenance.json"
                    )["fit_partition"],
                }
                for dataset in manifest_paths
            ])
            display(predictive)
            display(contrastive_status)
            '''
        ),
        md("## 4. Primary fixed-coverage comparison"),
        code(
            '''
            primary = coverage[coverage["coverage_budget"].round(2).isin([0.10, 0.25])].copy()
            display(primary.sort_values(["coverage_budget", "dataset"]))

            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
            sns.barplot(data=primary, x="coverage_budget", y="delta_vs_score_only", hue="dataset", ax=axes[0])
            axes[0].axhline(0, color="black", linewidth=1)
            axes[0].set(title="Incremental precision at primary coverages", xlabel="Coverage", ylabel="VASRE - score-only precision")
            sns.barplot(data=primary, x="coverage_budget", y="selected_alert_precision", hue="dataset", ax=axes[1])
            axes[1].set(title="Selected-alert precision", xlabel="Coverage", ylabel="Precision")
            plt.show()
            '''
        ),
        md("## 5. Uncertainty, matched-risk and negative-control evidence"),
        code(
            '''
            display(bootstrap[bootstrap["coverage"].round(2).isin([0.10, 0.25])])
            display(matched[matched["coverage_budget"].round(2).isin([0.10, 0.25])])
            display(controls)
            display(outcomes[outcomes["coverage_budget"].round(2).isin([0.10, 0.25])])
            display(ablations[ablations["coverage_budget"].round(2).isin([0.10, 0.25])])
            display(predictor_sensitivity[
                predictor_sensitivity["coverage_budget"].round(2).isin([0.10, 0.25])
            ])

            bootstrap_primary = bootstrap[
                bootstrap["coverage"].round(2).isin([0.10, 0.25])
            ][[
                "dataset", "coverage", "ci_low", "ci_high", "valid_iterations",
                "probability_vasre_better", "outcome"
            ]].rename(columns={"outcome": "bootstrap_outcome"})
            matched_primary = matched[
                matched["coverage_budget"].round(2).isin([0.10, 0.25])
            ][[
                "dataset", "coverage_budget", "matched_pairs", "match_rate",
                "mean_absolute_risk_gap", "matched_precision_difference",
                "matched_ci_low", "matched_ci_high"
            ]]
            outcome_columns = [
                "dataset", "coverage_budget", "outcome", "risk_conditioned_tp_fp_gap",
                "positive_incremental_evidence_confirmed", "saturation_confirmed",
                "saturation_scope_eligible", "saturation_sample_adequate",
                "core_equal_count_score_only_comparison",
                "shuffled_control_comparison_valid",
                "shuffled_controls_below_locked_primary",
                "shuffled_controls_no_material_gain"
            ]
            outcome_primary = outcomes[
                outcomes["coverage_budget"].round(2).isin([0.10, 0.25])
            ][[column for column in outcome_columns if column in outcomes]].rename(
                columns={"outcome": "stress_outcome"}
            )
            synthesis = primary.merge(
                bootstrap_primary,
                left_on=["dataset", "coverage_budget"],
                right_on=["dataset", "coverage"],
                how="left",
                validate="one_to_one",
            ).merge(
                matched_primary,
                on=["dataset", "coverage_budget"],
                how="left",
                validate="one_to_one",
            ).merge(
                outcome_primary,
                on=["dataset", "coverage_budget"],
                how="left",
                validate="one_to_one",
            ).merge(
                contrastive_status,
                on="dataset",
                how="left",
                validate="many_to_one",
            )
            display(synthesis)
            '''
        ),
        md("## 6. Export a checksum-ready synthesis table"),
        code(
            '''
            from datetime import datetime, timezone
            import hashlib

            def sha256_path(path):
                digest = hashlib.sha256()
                with Path(path).open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()

            SYNTHESIS_DIR = OUTPUT_BASE / "11_stress_test_synthesis"
            SYNTHESIS_DIR.mkdir(parents=True, exist_ok=True)
            synthesis_path = SYNTHESIS_DIR / "primary_stress_test_synthesis.csv"
            eligibility_path = SYNTHESIS_DIR / "stress_input_eligibility.csv"
            synthesis.to_csv(synthesis_path, index=False)
            eligibility.to_csv(eligibility_path, index=False)
            upstream = {
                dataset: {
                    "stress_test_manifest": str(manifest_path),
                    "stress_test_manifest_sha256": sha256_path(manifest_path),
                    "stress_lineage_sha256": sha256_path(manifest_path.with_name("stress_lineage.json")),
                    "git_commit": manifests[dataset]["git_commit"],
                    "pipeline_fingerprint": manifests[dataset]["stress_pipeline_fingerprint"],
                    "source_fingerprint": manifests[dataset]["stress_source_fingerprint"],
                    "core_environment": upstream_core_environments[dataset],
                }
                for dataset, manifest_path in manifest_paths.items()
            }
            synthesis_note = {
                "schema_version": 1,
                "analysis_type": "post_hoc_locked_artifact_synthesis",
                "datasets": sorted(manifest_paths),
                "primary_coverages": [0.10, 0.25],
                "model_rule_or_policy_reselection": False,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": GIT_COMMIT,
                "source_fingerprint": eligibility["source_fingerprint"].iloc[0],
                "upstream_core_environment_match_required": True,
                "matched_upstream_core_environment": MATCHED_UPSTREAM_CORE_ENVIRONMENT,
                "synthesis_runtime_core_environment": CURRENT_CORE_ENVIRONMENT,
                "upstream": upstream,
                "outputs": {
                    synthesis_path.name: sha256_path(synthesis_path),
                    eligibility_path.name: sha256_path(eligibility_path),
                },
                "claim_boundary": "Stress evidence does not establish causality or production generalization.",
            }
            synthesis_manifest_path = SYNTHESIS_DIR / "synthesis_manifest.json"
            synthesis_manifest_path.write_text(
                json.dumps(synthesis_note, indent=2), encoding="utf-8"
            )
            synthesis_lineage = {
                "schema_version": 1,
                "manifest": synthesis_manifest_path.name,
                "manifest_sha256": sha256_path(synthesis_manifest_path),
                "outputs": synthesis_note["outputs"],
                "upstream_manifest_sha256": {
                    dataset: values["stress_test_manifest_sha256"]
                    for dataset, values in upstream.items()
                },
                "matched_upstream_core_environment": MATCHED_UPSTREAM_CORE_ENVIRONMENT,
                "synthesis_runtime_core_environment": CURRENT_CORE_ENVIRONMENT,
            }
            synthesis_lineage_path = SYNTHESIS_DIR / "synthesis_lineage.json"
            synthesis_lineage_path.write_text(
                json.dumps(synthesis_lineage, indent=2), encoding="utf-8"
            )
            assert sha256_path(synthesis_manifest_path) == synthesis_lineage["manifest_sha256"]
            print({
                "output_dir": str(SYNTHESIS_DIR),
                "rows": len(synthesis),
                "manifest_sha256": synthesis_lineage["manifest_sha256"],
            })
            '''
        ),
        md(
            '''
            ## 7. Thesis interpretation

            Báo cáo từng stress dataset theo đúng vai trò riêng thay vì lấy trung bình gộp. TransXion
            kiểm tra robustness trong một AML benchmark khó hơn; AMLNet kiểm tra saturation/giới hạn.
            Nếu một dataset abstain hoặc không có CI dương, giữ nguyên kết quả âm. Không dùng dataset
            còn lại để che khuất failure và không gọi hai stress test này là external validation thực tế.
            '''
        ),
    ]
    return _notebook(cells, "11_stress_test_synthesis")


def build_all_stress_notebooks() -> dict[str, nbf.NotebookNode]:
    return {
        "09_TransXion_v2_Stress_Test.ipynb": build_dataset_notebook(
            notebook_id="09_transxion_v2_stress_test",
            title="09 - TransXion v2 Stress Test",
            dataset_key="transxion_v2",
            config_relative="configs/stress/transxion_v2.yaml",
            source_filename="tx.csv",
            source_note=(
                "Trong tên notebook, `v2` là revision 2 của paper được proposal dẫn chiếu; "
                "official repository không công bố một dataset-v2 tag riêng. Full run khóa vào `tx.csv`, "
                "repository revision và SHA-256 ghi trong config."
            ),
            fixture_function="make_synthetic_transxion_test_fixture",
        ),
        "10_AMLNet_v1_0_Stress_Test.ipynb": build_dataset_notebook(
            notebook_id="10_amlnet_v1_0_stress_test",
            title="10 - AMLNet v1.0 Saturation Stress Test",
            dataset_key="amlnet_v1_0",
            config_relative="configs/stress/amlnet_v1.yaml",
            source_filename="AMLNet_August 2025.csv",
            source_note=(
                "Full run khóa vào Zenodo record v1.0, exact filename và MD5 ghi trong config. "
                "Timestamp được parse an toàn từ metadata rồi stable-sort; row order và `step` không được dùng làm thời gian."
            ),
            fixture_function="make_synthetic_amlnet_test_fixture",
        ),
        "11_Stress_Test_Synthesis.ipynb": build_synthesis_notebook(),
    }


def _source_signature(notebook: nbf.NotebookNode) -> list[tuple[str, str]]:
    return [(cell.cell_type, cell.source) for cell in notebook.cells]


def write_all(*, check: bool = False) -> list[Path]:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    changed: list[Path] = []
    for filename, notebook in build_all_stress_notebooks().items():
        nbf.validate(notebook)
        destination = NOTEBOOK_DIR / filename
        if check:
            if not destination.is_file():
                raise SystemExit(f"Missing generated notebook: {destination}")
            actual = nbf.read(destination, as_version=4)
            if _source_signature(actual) != _source_signature(notebook):
                raise SystemExit(f"Notebook source is out of sync: {destination}")
            continue
        nbf.write(notebook, destination)
        changed.append(destination)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if committed notebook source is stale")
    args = parser.parse_args()
    paths = write_all(check=args.check)
    if args.check:
        print("Stress-test notebook source is synchronized.")
    else:
        for path in paths:
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
