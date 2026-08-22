"""Generate the eight reproducible thesis notebooks with nbformat."""

from __future__ import annotations

import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


# Preserve Notebook 01's successfully executed setup source byte-for-byte. It
# only consumes direct dataset inputs and is not an upstream dependency of the
# lineage-gated synthesis notebook.
EDA_SETUP = r'''
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

KAGGLE = Path("/kaggle").exists()
REPO_URL = "https://github.com/Tommyhuy1705/Explainable_NeuroSymbolic_Fraud_Detection.git"
KAGGLE_PROJECT_DIR = Path("/kaggle/working/Explainable_NeuroSymbolic_Fraud_Detection")

if KAGGLE:
    os.environ.setdefault("THESIS_QUICK_RUN", "0")
    os.environ.setdefault("THESIS_SYNTHETIC_FALLBACK", "0")

def find_project_root() -> Path | None:
    direct_candidates = [KAGGLE_PROJECT_DIR, Path.cwd(), *Path.cwd().parents]
    for candidate in direct_candidates:
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate
    for base in (Path("/kaggle/working"), Path("/kaggle/input")):
        if base.exists():
            matches = sorted(path.parent for path in base.glob("**/configs") if path.is_dir())
            for candidate in matches:
                if (candidate / "src").is_dir():
                    return candidate
    return None

PROJECT_ROOT = find_project_root()
if PROJECT_ROOT is None and KAGGLE:
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", "main", REPO_URL, str(KAGGLE_PROJECT_DIR)],
        check=True,
    )
    PROJECT_ROOT = find_project_root()
if PROJECT_ROOT is None:
    raise FileNotFoundError("Project root with src/ and configs/ was not found")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AUDIT_SOURCE_FILES = (
    "scripts/generate_notebooks.py",
    "src/artifacts.py",
    "src/data/dataset.py",
    "src/data/preprocessing.py",
    "src/experiment.py",
    "src/explanation/explanation_metrics.py",
    "src/explanation/rule_explainer.py",
    "src/logic/fraud_rules.py",
    "src/logic/knowledge_base.py",
    "src/logic/predicates.py",
    "src/logic/tensor_logic.py",
)

def audit_pipeline_fingerprint(config_path: Path) -> str:
    paths = [PROJECT_ROOT / relative for relative in AUDIT_SOURCE_FILES]
    paths.append(Path(config_path))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Files required for the audit-pipeline fingerprint are missing: {missing}")
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(PROJECT_ROOT).as_posix()):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

QUICK_RUN = os.getenv("THESIS_QUICK_RUN", "0") == "1"
ALLOW_SYNTHETIC_FALLBACK = os.getenv("THESIS_SYNTHETIC_FALLBACK", "0") == "1"
OUTPUT_BASE = Path("/kaggle/working/thesis_outputs") if KAGGLE else PROJECT_ROOT / "results/runs/notebooks"
INPUT_ROOTS = [OUTPUT_BASE, PROJECT_ROOT / "results/runs/notebooks"]
if Path("/kaggle/input").exists():
    INPUT_ROOTS.append(Path("/kaggle/input"))

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
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 120)
print({
    "project_root": str(PROJECT_ROOT),
    "git_commit": GIT_COMMIT,
    "quick_run": QUICK_RUN,
    "synthetic_fallback": ALLOW_SYNTHETIC_FALLBACK,
    "kaggle": KAGGLE,
})
'''


# Notebook 04-08 may attach upstream notebook outputs that themselves contain
# an old repository clone. Their setup must never discover executable source
# under /kaggle/input.
SETUP = r'''
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

KAGGLE = Path("/kaggle").exists()
REPO_URL = "https://github.com/Tommyhuy1705/Explainable_NeuroSymbolic_Fraud_Detection.git"
KAGGLE_PROJECT_DIR = Path("/kaggle/working/Explainable_NeuroSymbolic_Fraud_Detection")

if KAGGLE:
    os.environ.setdefault("THESIS_QUICK_RUN", "0")
    os.environ.setdefault("THESIS_SYNTHETIC_FALLBACK", "0")

def sync_kaggle_project() -> Path:
    """Use one current working clone; never import source bundled in an input artifact."""
    if not KAGGLE_PROJECT_DIR.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", "main", REPO_URL, str(KAGGLE_PROJECT_DIR)],
            check=True,
        )
    else:
        if not (KAGGLE_PROJECT_DIR / ".git").is_dir():
            raise RuntimeError(
                f"Kaggle project path exists but is not a Git clone: {KAGGLE_PROJECT_DIR}"
            )
        subprocess.run(
            ["git", "-C", str(KAGGLE_PROJECT_DIR), "pull", "--ff-only", "origin", "main"],
            check=True,
        )
    return KAGGLE_PROJECT_DIR.resolve()

def find_project_root() -> Path | None:
    direct_candidates = [Path.cwd(), *Path.cwd().parents]
    for candidate in direct_candidates:
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate.resolve()
    return None

PROJECT_ROOT = sync_kaggle_project() if KAGGLE else find_project_root()
if PROJECT_ROOT is None:
    raise FileNotFoundError("Project root with src/ and configs/ was not found")

project_root_string = str(PROJECT_ROOT)
while project_root_string in sys.path:
    sys.path.remove(project_root_string)
sys.path.insert(0, project_root_string)

# Run All can reuse a live Kaggle kernel. Remove previously imported project
# modules so an updated working clone cannot be shadowed by stale objects.
for module_name in tuple(sys.modules):
    if module_name == "src" or module_name.startswith("src."):
        del sys.modules[module_name]

AUDIT_SOURCE_FILES = (
    "scripts/generate_notebooks.py",
    "src/artifacts.py",
    "src/data/dataset.py",
    "src/data/preprocessing.py",
    "src/experiment.py",
    "src/explanation/explanation_metrics.py",
    "src/explanation/rule_explainer.py",
    "src/logic/fraud_rules.py",
    "src/logic/knowledge_base.py",
    "src/logic/predicates.py",
    "src/logic/tensor_logic.py",
)

def audit_pipeline_fingerprint(config_path: Path) -> str:
    paths = [PROJECT_ROOT / relative for relative in AUDIT_SOURCE_FILES]
    paths.append(Path(config_path))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Files required for the audit-pipeline fingerprint are missing: {missing}")
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(PROJECT_ROOT).as_posix()):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

QUICK_RUN = os.getenv("THESIS_QUICK_RUN", "0") == "1"
ALLOW_SYNTHETIC_FALLBACK = os.getenv("THESIS_SYNTHETIC_FALLBACK", "0") == "1"
OUTPUT_BASE = Path("/kaggle/working/thesis_outputs") if KAGGLE else PROJECT_ROOT / "results/runs/notebooks"
INPUT_ROOTS = [OUTPUT_BASE, PROJECT_ROOT / "results/runs/notebooks"]
if Path("/kaggle/input").exists():
    INPUT_ROOTS.append(Path("/kaggle/input"))

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
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 120)
print({
    "project_root": str(PROJECT_ROOT),
    "project_source_policy": "working_clone_main" if KAGGLE else "local_project_root",
    "git_commit": GIT_COMMIT,
    "quick_run": QUICK_RUN,
    "synthetic_fallback": ALLOW_SYNTHETIC_FALLBACK,
    "kaggle": KAGGLE,
})
'''


# Keep the setup source of already executed Notebooks 02/03 byte-for-byte stable.
# Their full Kaggle outputs are frozen evidence; changing even an execution-neutral
# setup line would make the displayed output no longer correspond to the source.
BENCHMARK_SETUP = r'''
from pathlib import Path
import json
import os
import subprocess
import sys

KAGGLE = Path("/kaggle").exists()
REPO_URL = "https://github.com/Tommyhuy1705/Explainable_NeuroSymbolic_Fraud_Detection.git"
KAGGLE_PROJECT_DIR = Path("/kaggle/working/Explainable_NeuroSymbolic_Fraud_Detection")

if KAGGLE:
    os.environ["THESIS_QUICK_RUN"] = "0"
    os.environ["THESIS_SYNTHETIC_FALLBACK"] = "0"
    if not KAGGLE_PROJECT_DIR.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", "main", REPO_URL, str(KAGGLE_PROJECT_DIR)],
            check=True,
        )

def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate
    for base in (Path("/kaggle/working"), Path("/kaggle/input")):
        if base.exists():
            matches = [path for path in base.glob("**/configs") if path.is_dir()]
            if matches:
                return matches[0].parent
    raise FileNotFoundError("Project root with src/ and configs/ was not found")

PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

QUICK_RUN = os.getenv("THESIS_QUICK_RUN", "0") == "1"
ALLOW_SYNTHETIC_FALLBACK = os.getenv("THESIS_SYNTHETIC_FALLBACK", "0") == "1"
OUTPUT_BASE = Path("/kaggle/working/thesis_outputs") if KAGGLE else PROJECT_ROOT / "results/runs/notebooks"
INPUT_ROOTS = [OUTPUT_BASE, PROJECT_ROOT / "results/runs/notebooks"]
if Path("/kaggle/input").exists():
    INPUT_ROOTS.append(Path("/kaggle/input"))

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
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 120)
print({
    "project_root": str(PROJECT_ROOT),
    "git_commit": GIT_COMMIT,
    "quick_run": QUICK_RUN,
    "synthetic_fallback": ALLOW_SYNTHETIC_FALLBACK,
    "kaggle": KAGGLE,
})
'''


def frozen_artifact_preflight(dataset_name: str) -> str:
    """Return a local/Kaggle-safe cell that makes frozen inputs visible before loading."""
    return f'''
    preflight_manifests = []
    preflight_artifacts = []
    for root_value in INPUT_ROOTS:
        root = Path(root_value)
        if not root.exists():
            continue
        manifest_paths = [root] if root.is_file() and root.name == "frozen_reference_manifest.json" else list(root.glob("**/frozen_reference_manifest.json"))
        for manifest_path in manifest_paths:
            try:
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(manifest_payload.get("dataset_name", "")).lower() != "{dataset_name}".lower():
                continue
            preflight_manifests.append(manifest_path.resolve())
            artifact_path = manifest_path.parent / str(manifest_payload.get("artifact_file", ""))
            if artifact_path.exists():
                preflight_artifacts.append(artifact_path.resolve())

    preflight_table = pd.DataFrame({{
        "manifest": [str(path) for path in sorted(set(preflight_manifests))],
    }})
    display(preflight_table)
    print("Frozen artifacts:")
    for path in sorted(set(preflight_artifacts)):
        print(path)
    if not preflight_manifests or not preflight_artifacts:
        raise FileNotFoundError(
            "No complete {dataset_name} frozen artifact was found below INPUT_ROOTS. "
            "On Kaggle, attach the corresponding benchmark notebook output; locally, place it below results/runs/notebooks."
        )
    '''


def notebook(title: str, cells: list, accelerator: str = "none") -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(cells=[md(f"# {title}"), *cells])
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "kaggle": {
            "accelerator": accelerator,
            "dataSources": [],
            "isInternetEnabled": True,
            "isGpuEnabled": accelerator != "none",
        },
    }
    return nb


def exploration_section(
    label: str,
    config_name: str,
    synthetic_factory: str,
    slug: str,
    sentinel_value: int | float | None = None,
) -> list:
    sentinel_literal = repr(sentinel_value)
    return [
        md(f"## {label}"),
        code(f'''
        config = load_config(PROJECT_ROOT / "configs/{config_name}")
        try:
            frame = load_fraud_dataframe(config, max_rows=max_rows)
            data_source = "official_benchmark_file"
        except FileNotFoundError:
            if not ALLOW_SYNTHETIC_FALLBACK:
                raise
            frame = {synthetic_factory}(max_rows or 6000, seed=config["project"]["seed"])
            data_source = "synthetic_fallback"
        target = config["dataset"]["target_column"]
        time_column = config["dataset"]["time_column"]
        prepared = prepare_dataset(frame, config)
        integrity = split_integrity_summary(
            prepared.train_frame, prepared.validation_frame, prepared.test_frame, time_column
        )
        integrity["fraud_rate"] = [prepared.y_train.mean(), prepared.y_validation.mean(), prepared.y_test.mean()]
        if config["split"]["strategy"] == "temporal_group":
            assert integrity["group_disjoint"].all(), "Temporal groups overlap across BAF splits"
        assert time_column not in prepared.feature_names, "Raw split time must not enter the predictor"
        native_missing = frame.isna().mean().sort_values(ascending=False).rename("missing_fraction")
        configured_sentinel = {sentinel_literal}
        if configured_sentinel is None:
            sentinel_missing = pd.Series(dtype=float, name="sentinel_fraction")
            sentinel_missing_cells = 0
        else:
            sentinel_mask = frame.eq(configured_sentinel)
            sentinel_missing = sentinel_mask.mean().sort_values(ascending=False).rename("sentinel_fraction")
            sentinel_missing_cells = int(sentinel_mask.to_numpy().sum())
        identity_join_coverage = frame.attrs.get("identity_join_coverage")
        summary = pd.DataFrame({{
            "dataset": [config["dataset"]["name"]],
            "benchmark_type": [frame.attrs.get("benchmark_type", "deterministic_smoke_data")],
            "rows": [len(frame)],
            "columns": [frame.shape[1]],
            "fraud_count": [int(frame[target].sum())],
            "fraud_rate": [float(frame[target].mean())],
            "duplicate_rows": [int(frame.duplicated().sum())],
            "native_missing_cells": [int(frame.isna().to_numpy().sum())],
            "sentinel_missing_cells": [sentinel_missing_cells],
            "identity_join_coverage": [identity_join_coverage],
            "data_source": [data_source],
            "selected_features": [len(prepared.feature_names)],
        }})
        dataset_summaries.append(summary)
        display(
            summary,
            integrity,
            native_missing.rename("native_nan_fraction").to_frame().head(20),
        )
        if configured_sentinel is not None:
            display(sentinel_missing.rename(f"sentinel_{{configured_sentinel}}_fraction").to_frame().head(20))
        integrity.to_csv(output_dir / "{slug}_split_summary.csv", index=False)
        native_missing.to_frame().to_csv(output_dir / "{slug}_missingness.csv")
        sentinel_missing.to_frame().to_csv(output_dir / "{slug}_sentinel_missingness.csv")
        '''),
        code(f'''
        top_native_missing = native_missing.loc[native_missing > 0].head(20).sort_values()
        top_sentinel_missing = sentinel_missing.loc[sentinel_missing > 0].head(20).sort_values()
        if not top_native_missing.empty:
            missing_for_plot = top_native_missing
            missing_title = "Top native NaN columns"
        elif not top_sentinel_missing.empty:
            missing_for_plot = top_sentinel_missing
            missing_title = f"Top configured missing sentinels (value={{configured_sentinel}})"
        else:
            missing_for_plot = pd.Series(dtype=float)
            missing_title = "Missingness audit"

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(17, 8),
            gridspec_kw={{"width_ratios": [1.0, 1.45]}},
        )
        integrity.plot.bar(
            x="split", y="fraud_rate", ax=axes[0], legend=False, color="#C44E52", rot=0
        )
        axes[0].set_title("{label.split('. ', 1)[-1]} fraud rate by split")
        axes[0].set_ylabel("Fraud rate")
        axes[0].set_xlabel("Split")
        axes[0].margins(x=0.12)

        if missing_for_plot.empty:
            axes[1].set_title(missing_title)
            axes[1].text(
                0.5,
                0.5,
                "No native NaN or configured sentinel values detected",
                ha="center",
                va="center",
                fontsize=12,
                color="#4C72B0",
                transform=axes[1].transAxes,
            )
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            for spine in axes[1].spines.values():
                spine.set_visible(False)
        else:
            missing_for_plot.plot.barh(ax=axes[1], legend=False, color="#4C72B0")
            axes[1].set_title(missing_title)
            axes[1].set_xlabel("Missing fraction")
            axes[1].set_ylabel("Feature")
            axes[1].tick_params(axis="y", labelsize=10, pad=7)
            axes[1].margins(y=0.03)

        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.90, wspace=0.62)
        fig.savefig(output_dir / "{slug}_data_quality.png", dpi=160, bbox_inches="tight")
        plt.show()
        '''),
        code(f'''
        temporal_frame = frame[[time_column, target]].dropna(subset=[time_column]).copy()
        if config["split"]["strategy"] == "temporal_group":
            temporal_fraud = (
                temporal_frame.groupby(time_column, as_index=False)[target]
                .agg(rows="size", fraud_count="sum", fraud_rate="mean")
                .rename(columns={{time_column: "period"}})
            )
            period_label = str(time_column)
        else:
            temporal_frame = temporal_frame.sort_values(time_column, kind="mergesort").reset_index(drop=True)
            temporal_frame["period"] = pd.qcut(
                np.arange(len(temporal_frame)), q=min(12, len(temporal_frame)), labels=False, duplicates="drop"
            ) + 1
            temporal_fraud = temporal_frame.groupby("period", as_index=False)[target].agg(
                rows="size", fraud_count="sum", fraud_rate="mean"
            )
            period_label = "Chronological quantile bin"
        display(temporal_fraud.round(5))
        temporal_fraud.to_csv(output_dir / "{slug}_temporal_fraud_rate.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(temporal_fraud["period"], temporal_fraud["fraud_rate"], marker="o", color="#4C72B0")
        ax.set_title("{label.split('. ', 1)[-1]} fraud rate over ordered time periods")
        ax.set_xlabel(period_label)
        ax.set_ylabel("Fraud rate")
        ax.margins(x=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / "{slug}_temporal_fraud_rate.png", dpi=160, bbox_inches="tight")
        plt.show()
        '''),
    ]


def build_exploration() -> nbf.NotebookNode:
    cells = [
        code(EDA_SETUP),
        md("""
        ## Thiết lập

        Notebook kiểm tra dữ liệu, class imbalance, missingness và ranh giới thời gian trước khi huấn luyện.
        BAF phải có các nhóm tháng không giao nhau; raw time chỉ dùng để chia tập, không làm predictor feature.
        """),
        code("""
        from src.data import (
            load_config, load_fraud_dataframe, make_synthetic_baf_data,
            make_synthetic_fraud_data, prepare_dataset, split_integrity_summary,
        )

        max_rows = 12000 if QUICK_RUN else None
        output_dir = OUTPUT_BASE / "01_data_exploration"
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_summaries = []
        """),
    ]
    cells += exploration_section("I. IEEE-CIS", "ieee_cis.yaml", "make_synthetic_fraud_data", "ieee_cis")
    cells += exploration_section(
        "II. BAF", "baf.yaml", "make_synthetic_baf_data", "baf", sentinel_value=-1
    )
    cells += [
        md("## III. So sánh tổng quan"),
        code("""
        combined_summary = pd.concat(dataset_summaries, ignore_index=True)
        display(combined_summary)
        combined_summary.to_csv(output_dir / "dataset_summary.csv", index=False)
        ax = combined_summary.plot.bar(
            x="dataset",
            y="fraud_rate",
            legend=False,
            color=["#4C72B0", "#55A868"],
            figsize=(8, 5),
            rot=0,
        )
        ax.set_title("Fraud prevalence across datasets")
        ax.set_ylabel("Fraud rate")
        ax.set_xlabel("Dataset")
        ax.margins(x=0.15)
        plt.tight_layout()
        plt.savefig(output_dir / "dataset_fraud_rate_comparison.png", dpi=160, bbox_inches="tight")
        plt.show()
        """),
        md("## Takeaways"),
        code("""
        display(Markdown(
            "- Review every displayed split before model training.\\n"
            "- For BAF, the required final split is months **0-4 / 5 / 6-7** with no overlap.\\n"
            "- Native NaN and dataset-specific `-1` missing sentinels are audited separately.\\n"
            "- `official_benchmark_file` means the published file was loaded; BAF itself remains a privacy-preserving synthetic benchmark."
        ))
        """),
    ]
    return notebook("01 - Multi-Dataset Exploration and Split Audit", cells)


def build_benchmark(number: str, dataset_name: str, label: str) -> nbf.NotebookNode:
    return notebook(
        f"{number} - {label} Predictive Model Benchmarks",
        [
            code(BENCHMARK_SETUP),
            md("""
            ## Thiết lập

            PR-AUC là metric xếp hạng chính. Mỗi full run dùng ba seeds. Calibration được fit trên nửa đầu
            validation, chọn trên nửa sau validation; decision threshold cũng chỉ chọn trên nửa sau validation.
            Reference predictor được chọn bằng mean validation raw PR-AUC, không dùng test.
            """),
            code(f'''
            from src.experiment import run_repeated_predictive_benchmarks

            output_dir = OUTPUT_BASE / "{number}_{dataset_name}_model_benchmarks"
            result = run_repeated_predictive_benchmarks(
                PROJECT_ROOT / "configs/{dataset_name}.yaml",
                output_dir=output_dir,
                model_names=("mlp", "tabular_resnet", "tree"),
                quick_run=QUICK_RUN,
                synthetic_fallback=ALLOW_SYNTHETIC_FALLBACK,
            )
            summary = result["summary"]
            print({{
                "data_sources": result["data_sources"],
                "seeds": result["seeds"],
                "reference_model_key": result["reference_model_key"],
                "reference_seed": result["reference_seed"],
                "frozen_manifest": str(result["frozen_manifest_path"]),
            }})
            display(summary.round(4))
            '''),
            md("## Data and calibration"),
            code("""
            display(result["data_summary"])
            selected_calibration = result["calibration_comparison"].query("selected").copy()
            display(selected_calibration.round(5))
            display(result["predictive_bootstrap"].round(5))
            """),
            md("## Results"),
            code("""
            test_metrics = summary.query("split == 'test'").copy()
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].bar(test_metrics["model"], test_metrics["raw_pr_auc_mean"],
                        yerr=test_metrics["raw_pr_auc_std"].fillna(0), color="#4C72B0", capsize=4)
            axes[0].set_title("Test raw PR-AUC mean ± SD")
            axes[0].tick_params(axis="x", rotation=20)
            axes[1].bar(test_metrics["model"], test_metrics["fbeta_mean"],
                        yerr=test_metrics["fbeta_std"].fillna(0), color="#55A868", capsize=4)
            axes[1].set_title("Test calibrated-threshold F2 mean ± SD")
            axes[1].tick_params(axis="x", rotation=20)
            plt.tight_layout()
            fig.savefig(output_dir / "predictive_model_comparison.png", dpi=160, bbox_inches="tight")
            plt.show()
            """),
            code("""
            history_rows = []
            for seed, model_histories in result["histories"].items():
                for model_name, history in model_histories.items():
                    history_frame = pd.DataFrame(history)
                    best_index = history_frame["validation_pr_auc"].idxmax()
                    history_rows.append({
                        "seed": seed, "model": model_name, "epochs_run": len(history_frame),
                        "best_epoch": int(history_frame.loc[best_index, "epoch"]),
                        "best_validation_pr_auc": history_frame.loc[best_index, "validation_pr_auc"],
                    })
            display(pd.DataFrame(history_rows).round(5))
            """),
            md("## Takeaways"),
            code("""
            selected_key = result["reference_model_key"]
            validation_best = summary.query("split == 'validation' and model_key == @selected_key").iloc[0]
            test_reference = summary.query("split == 'test' and model_key == @selected_key").iloc[0]
            display(Markdown(
                f"- Reference model selected on validation: **{validation_best['model']}**.\\n"
                f"- Mean validation raw PR-AUC: **{validation_best['raw_pr_auc_mean']:.4f}**.\\n"
                f"- Locked test raw PR-AUC: **{test_reference['raw_pr_auc_mean']:.4f} ± {test_reference['raw_pr_auc_std']:.4f}**.\\n"
                f"- Frozen reference seed: **{result['reference_seed']}**.\\n"
                "- Test metrics report the locked protocol; they are not used for model, calibration, or threshold selection."
            ))
            """),
        ],
        accelerator="nvidiaTeslaT4",
    )


def build_rule_analysis() -> nbf.NotebookNode:
    return notebook(
        "04 - IEEE-CIS LTN-Style Fraud Rule Analysis",
        [
            code(SETUP),
            md("""
            ## Thiết lập

            Rules và quantile thresholds chỉ fit trên train. Notebook đánh giá rule truth values,
            class-balanced knowledge-base satisfaction và một diagnostic fuzzy predicate khả vi.
            Explicit numeric thresholds dùng softness theo đơn vị gốc; quantile/category-risk predicates
            dùng relative softness. `transaction_hour` chỉ là cyclic phase suy ra từ TransactionDT,
            không được diễn giải như giờ địa phương đã biết. Đây là LTN-style explanation layer,
            không phải end-to-end LTN predictor.
            """),
            code("""
            from src.artifacts import sha256_file, stable_config_hash
            from src.data import load_config, prepare_dataset
            from src.experiment import load_experiment_data
            from src.explanation import rule_quality_table
            from src.logic import FraudKnowledgeBase, FraudRuleEngine

            config_path = PROJECT_ROOT / "configs/ieee_cis.yaml"
            config = load_config(config_path)
            output_dir = OUTPUT_BASE / "04_ieee_cis_ltn_rule_analysis"
            output_dir.mkdir(parents=True, exist_ok=True)
            frame, data_source = load_experiment_data(
                config, max_rows=12000 if QUICK_RUN else None,
                synthetic_fallback=ALLOW_SYNTHETIC_FALLBACK,
                synthetic_rows=12000 if QUICK_RUN else 6000,
            )
            if not QUICK_RUN and str(data_source).lower() == "synthetic":
                raise ValueError("Full thesis rule analysis cannot consume synthetic-fallback data")
            prepared = prepare_dataset(frame, config)
            target = config["dataset"]["target_column"]
            engine = FraudRuleEngine(config["logic"]["rules"]).fit(prepared.train_frame, target)
            knowledge_base = FraudKnowledgeBase(engine)
            print({"data_source": data_source, "active_rules": len(engine.rules), "skipped": engine.skipped_rules})
            fitted_thresholds = engine.fitted_thresholds()
            rationale_rows = []
            for definition in config["logic"]["rules"]:
                operators = [condition["operator"] for condition in definition["conditions"]]
                has_category_risk = "category_risk" in operators
                has_quantile = any(operator.endswith("_quantile") for operator in operators)
                if has_category_risk and has_quantile:
                    knowledge_type = "data-informed and train-fitted fuzzy hypothesis"
                    threshold_source = (
                        "train-label-smoothed category risk with configured cutoff plus training-split quantile"
                    )
                elif has_category_risk:
                    knowledge_type = "data-informed fuzzy hypothesis"
                    threshold_source = "train-label-smoothed category risk with configured cutoff"
                elif has_quantile:
                    knowledge_type = "domain hypothesis with train-fitted thresholds"
                    threshold_source = "training-split quantile"
                else:
                    knowledge_type = "configured domain hypothesis"
                    threshold_source = "explicit configured value"
                limitation = (
                    "TransactionDT-derived cyclic phase; the unknown clock origin prevents a local-hour claim."
                    if definition["name"] == "unusual_transaction_hour"
                    else "Association-based audit evidence; activation does not establish causality or model faithfulness."
                )
                rationale_rows.append({
                    "rule": definition["name"],
                    "description": definition.get("description", ""),
                    "knowledge_type": knowledge_type,
                    "threshold_source": threshold_source,
                    "limitation": limitation,
                })
            rule_rationale = pd.DataFrame(rationale_rows)
            display(fitted_thresholds, rule_rationale)
            rule_rationale.to_csv(output_dir / "ieee_rule_rationale.csv", index=False)
            """),
            md("## Differentiable fuzzy predicate diagnostic"),
            code("""
            import torch
            from src.logic import SoftThresholdPredicate, TensorLogic

            amount = torch.tensor(prepared.train_frame["TransactionAmt"].to_numpy(float), dtype=torch.float32)
            center = torch.nanmedian(amount)
            scale = torch.nan_to_num(amount.std(), nan=1.0).clamp_min(1e-6)
            values = torch.nan_to_num((amount - center) / scale)
            labels = torch.tensor(prepared.y_train, dtype=torch.float32)
            predicate = SoftThresholdPredicate(float(torch.quantile(values, 0.90)), temperature=0.5, learnable=True)
            optimizer = torch.optim.Adam(predicate.parameters(), lr=0.03)
            initial_threshold = float(predicate.threshold.detach())
            history = []
            for _ in range(30 if QUICK_RUN else 100):
                optimizer.zero_grad()
                evidence = predicate(values)
                satisfaction = 0.5 * (
                    TensorLogic.forall(evidence[labels == 1]) +
                    TensorLogic.forall(1.0 - evidence[labels == 0])
                )
                loss = 1.0 - satisfaction + 0.01 * (predicate.threshold - initial_threshold).pow(2)
                loss.backward()
                optimizer.step()
                history.append(float(satisfaction.detach()))
            tensor_demo = pd.DataFrame([{
                "initial_threshold": initial_threshold,
                "learned_threshold": float(predicate.threshold.detach()),
                "initial_satisfaction": history[0],
                "final_satisfaction": history[-1],
            }])
            display(tensor_demo.round(5))
            """),
            md("## Rule and knowledge-base results"),
            code("""
            split_frames = {
                "train": (prepared.train_frame, prepared.y_train),
                "validation": (prepared.validation_frame, prepared.y_validation),
                "test": (prepared.test_frame, prepared.y_test),
            }
            activation = float(config["logic"]["activation_threshold"])
            quality_frames = []
            satisfaction_rows = []
            for split, (split_frame, labels) in split_frames.items():
                truth = engine.evaluate(split_frame)
                quality_frames.append(rule_quality_table(truth, labels, activation).assign(split=split))
                satisfaction_rows.append({"split": split, **knowledge_base.satisfaction_breakdown(split_frame, target)})
            quality = pd.concat(quality_frames, ignore_index=True)
            satisfaction = pd.DataFrame(satisfaction_rows)
            display(quality.round(4), satisfaction.round(4))
            quality.to_csv(output_dir / "ieee_rule_quality.csv", index=False)
            satisfaction.to_csv(output_dir / "ieee_knowledge_base_satisfaction.csv", index=False)
            fitted_thresholds.to_csv(output_dir / "ieee_fitted_rule_thresholds.csv", index=False)
            tensor_demo.to_csv(output_dir / "ieee_tensor_predicate_diagnostic.csv", index=False)
            """),
            code("""
            validation_quality = quality.query("split == 'validation'")
            test_quality = quality.query("split == 'test'")
            stability = validation_quality.merge(test_quality, on="rule", suffixes=("_validation", "_test"))
            stability["coverage_delta"] = stability["coverage_test"] - stability["coverage_validation"]
            stability["lift_delta"] = stability["lift_test"] - stability["lift_validation"]
            display(stability[["rule", "coverage_delta", "lift_delta"]].round(4))
            stability.to_csv(output_dir / "ieee_rule_stability.csv", index=False)
            rule_output_files = [
                "ieee_rule_rationale.csv",
                "ieee_rule_quality.csv",
                "ieee_knowledge_base_satisfaction.csv",
                "ieee_fitted_rule_thresholds.csv",
                "ieee_tensor_predicate_diagnostic.csv",
                "ieee_rule_stability.csv",
            ]
            rule_lineage = {
                "notebook_id": "04_IEEE_CIS_LTN_Rule_Analysis",
                "git_commit": GIT_COMMIT,
                "dataset_name": config["dataset"]["name"],
                "data_source": data_source,
                "quick_run": QUICK_RUN,
                "config_sha256": stable_config_hash(config),
                "audit_source_sha256": audit_pipeline_fingerprint(config_path),
                "output_files": rule_output_files,
                "output_sha256": {
                    name: sha256_file(output_dir / name) for name in rule_output_files
                },
            }
            lineage_path = output_dir / "upstream_lineage.json"
            lineage_path.write_text(json.dumps(rule_lineage, indent=2), encoding="utf-8")
            print({"upstream_lineage": str(lineage_path)})
            """),
            md("## Takeaways"),
            code("""
            eligible_rules = test_quality.dropna(subset=["lift"]).query("active_count > 0")
            strongest = eligible_rules.sort_values("lift", ascending=False).iloc[0]
            test_satisfaction = satisfaction.query("split == 'test'").iloc[0]
            display(Markdown(
                f"- Highest observed test lift: **{strongest['rule']} = {strongest['lift']:.3f}** "
                f"with **{int(strongest['active_count'])}** active rows; lift is never interpreted without this denominator.\\n"
                f"- Test balanced knowledge-base satisfaction: **{test_satisfaction['balanced_satisfaction']:.3f}**.\\n"
                "- Rule lift and satisfaction measure limited association/logic agreement, not causality or predictor faithfulness."
            ))
            """),
        ],
    )


FROZEN_LINEAGE_HELPER = '''
def write_upstream_lineage(destination, notebook_id, output_files, config_path):
    output_files = list(output_files)
    lineage = {
        "notebook_id": notebook_id,
        "git_commit": GIT_COMMIT,
        "dataset_name": artifact["manifest"]["dataset_name"],
        "data_source": data_source,
        "frozen_data_source": artifact["manifest"].get("data_source"),
        "quick_run": QUICK_RUN,
        "reference_model_key": artifact["manifest"]["model_key"],
        "reference_seed": artifact["manifest"]["reference_seed"],
        "config_sha256": artifact["manifest"]["config_sha256"],
        "audit_source_sha256": audit_pipeline_fingerprint(config_path),
        "frozen_manifest_sha256": sha256_file(artifact["manifest_path"]),
        "frozen_artifact_sha256": sha256_file(artifact["artifact_path"]),
        "output_files": output_files,
        "output_sha256": {
            name: sha256_file(Path(destination) / name) for name in output_files
        },
    }
    lineage_path = Path(destination) / "upstream_lineage.json"
    lineage_path.write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    return lineage_path
'''


FROZEN_IEEE_SETUP = '''
from src.artifacts import assert_frozen_alignment, load_frozen_reference_artifact, sha256_file
from src.data import load_config, prepare_dataset
from src.experiment import load_experiment_data

config = load_config(PROJECT_ROOT / "configs/ieee_cis.yaml")
frame, data_source = load_experiment_data(
    config, max_rows=12000 if QUICK_RUN else None,
    synthetic_fallback=ALLOW_SYNTHETIC_FALLBACK,
    synthetic_rows=12000 if QUICK_RUN else 6000,
)
prepared = prepare_dataset(frame, config)
artifact = load_frozen_reference_artifact(
    "ieee_cis", expected_config=config,
    search_roots=[OUTPUT_BASE / "02_ieee_cis_model_benchmarks", *INPUT_ROOTS],
)
assert_frozen_alignment(artifact, prepared.y_validation, prepared.y_test)
if bool(artifact["manifest"]["quick_run"]) != QUICK_RUN:
    raise ValueError("Notebook mode and frozen artifact quick_run flag do not match")
if int(artifact["manifest"]["reference_seed"]) != int(config["evaluation"]["reference_seed"]):
    raise ValueError("Frozen artifact reference seed does not match the locked dataset protocol")
if not QUICK_RUN and str(artifact["manifest"].get("data_source", "")).lower() == "synthetic":
    raise ValueError("Full thesis evaluation cannot consume a synthetic-fallback frozen artifact")
probabilities = artifact["test_probability"]
threshold = float(artifact["manifest"]["threshold"])
print({
    "data_source": data_source,
    "reference_model": artifact["manifest"]["model"],
    "reference_seed": artifact["manifest"]["reference_seed"],
    "calibration_method": artifact["manifest"]["calibration_method"],
    "threshold": threshold,
    "artifact": str(artifact["artifact_path"]),
})
''' + FROZEN_LINEAGE_HELPER


FROZEN_BAF_SETUP = '''
from src.artifacts import assert_frozen_alignment, load_frozen_reference_artifact, sha256_file
from src.data import load_config, prepare_dataset
from src.experiment import load_experiment_data

config = load_config(PROJECT_ROOT / "configs/baf.yaml")
frame, data_source = load_experiment_data(
    config, max_rows=12000 if QUICK_RUN else None,
    synthetic_fallback=ALLOW_SYNTHETIC_FALLBACK,
    synthetic_rows=12000 if QUICK_RUN else 6000,
)
prepared = prepare_dataset(frame, config)
artifact = load_frozen_reference_artifact(
    "baf", expected_config=config,
    search_roots=[OUTPUT_BASE / "03_baf_model_benchmarks", *INPUT_ROOTS],
)
assert_frozen_alignment(artifact, prepared.y_validation, prepared.y_test)
if bool(artifact["manifest"]["quick_run"]) != QUICK_RUN:
    raise ValueError("Notebook mode and frozen artifact quick_run flag do not match")
if int(artifact["manifest"]["reference_seed"]) != int(config["evaluation"]["reference_seed"]):
    raise ValueError("Frozen artifact reference seed does not match the locked dataset protocol")
if not QUICK_RUN and str(artifact["manifest"].get("data_source", "")).lower() == "synthetic":
    raise ValueError("Full thesis evaluation cannot consume a synthetic-fallback frozen artifact")
probabilities = artifact["test_probability"]
threshold = float(artifact["manifest"]["threshold"])
print({
    "data_source": data_source,
    "benchmark_type": frame.attrs.get("benchmark_type", "privacy_preserving_synthetic_benchmark"),
    "reference_model": artifact["manifest"]["model"],
    "reference_seed": artifact["manifest"]["reference_seed"],
    "calibration_method": artifact["manifest"]["calibration_method"],
    "threshold": threshold,
    "artifact": str(artifact["artifact_path"]),
})
''' + FROZEN_LINEAGE_HELPER


def build_explanation() -> nbf.NotebookNode:
    return notebook(
        "05 - IEEE-CIS Rule Explanation Evaluation",
        [
            code(SETUP),
            md("""
            ## Thiết lập

            Notebook chỉ đọc frozen IEEE-CIS predictor từ Notebook 02. Predictor probabilities,
            calibration và threshold không được train/chọn lại trong bước explanation.
            """),
            code(frozen_artifact_preflight("ieee_cis")),
            code(FROZEN_IEEE_SETUP),
            code("""
            from src.explanation import (
                RuleExplainer, bootstrap_explanation_precision_gain,
                explanation_quality_metrics, rule_quality_table,
            )
            from src.logic import FraudRuleEngine

            output_dir = OUTPUT_BASE / "05_ieee_cis_rule_explanation_evaluation"
            output_dir.mkdir(parents=True, exist_ok=True)
            target = config["dataset"]["target_column"]
            engine = FraudRuleEngine(config["logic"]["rules"]).fit(prepared.train_frame, target)
            explainer = RuleExplainer(engine, config["logic"]["activation_threshold"], config["logic"]["top_k_rules"])
            explanations = explainer.explain(prepared.test_frame, probabilities, threshold)
            explanations["y_true"] = prepared.y_test
            display(explanations.head())
            """),
            md("## Results"),
            code("""
            truth = engine.evaluate(prepared.test_frame)
            rule_quality = rule_quality_table(truth, prepared.y_test, config["logic"]["activation_threshold"])
            quality = explanation_quality_metrics(explanations, prepared.y_test, probabilities, threshold)
            quality.update(bootstrap_explanation_precision_gain(
                explanations, prepared.y_test, probabilities, threshold,
                n_bootstrap=config["evaluation"]["bootstrap_iterations"], seed=config["project"]["seed"],
            ))
            quality_frame = pd.DataFrame([quality])
            display(rule_quality.round(4), quality_frame.round(4))

            predicted_alert = probabilities >= threshold
            cases = explanations.copy()
            cases["top_rule_name"] = truth.idxmax(axis=1)
            cases["top_rule_strength"] = truth.max(axis=1)
            cases["case_type"] = np.select(
                [predicted_alert & (prepared.y_test == 1), predicted_alert & (prepared.y_test == 0),
                 (~predicted_alert) & (prepared.y_test == 1)],
                ["true_positive", "false_positive", "false_negative"], default="true_negative",
            )
            cases["explanation_status"] = np.where(cases["explained"], "explained", "unexplained")
            cases["case_group"] = cases["case_type"] + "_" + cases["explanation_status"]
            requested_groups = [
                f"{case_type}_{status}"
                for case_type in ("true_positive", "false_positive", "false_negative", "true_negative")
                for status in ("explained", "unexplained")
            ]
            selected_groups = []
            for group_name in requested_groups:
                group = cases.loc[cases["case_group"] == group_name]
                if not group.empty:
                    selected_groups.append(group.sort_values("predicted_probability", ascending=False).head(2))
            selected_cases = (
                pd.concat(selected_groups).copy()
                if selected_groups
                else cases.head(0).copy()
            )

            fitted_rules = {rule.name: rule for rule in engine.rules}
            def serializable_value(value):
                if pd.isna(value):
                    return None
                return value.item() if hasattr(value, "item") else value

            def audit_evidence(row_index, case_row):
                active_names = list(case_row["rule_names"])
                audit_names = active_names or [case_row["top_rule_name"]]
                strengths = dict(zip(active_names, case_row["rule_strengths"]))
                details = []
                for rule_name in audit_names:
                    rule = fitted_rules[rule_name]
                    details.append({
                        "rule": rule_name,
                        "activated": rule_name in active_names,
                        "truth_strength": float(strengths.get(rule_name, truth.loc[row_index, rule_name])),
                        "conditions": [
                            {
                                "feature": condition.feature,
                                "operator": condition.operator,
                                "raw_value": serializable_value(prepared.test_frame.loc[row_index, condition.feature]),
                                "fitted_threshold": condition.threshold,
                                "fitted_missing_value": condition.numeric_fill_value,
                                "softness": condition.softness,
                                "softness_mode": condition.softness_mode,
                            }
                            for condition in rule.conditions
                        ],
                    })
                return details

            selected_cases["model_decision_threshold"] = threshold
            selected_cases["rule_activation_threshold"] = float(config["logic"]["activation_threshold"])
            selected_cases["audit_evidence"] = [
                audit_evidence(row_index, row) for row_index, row in selected_cases.iterrows()
            ]
            selected_cases = selected_cases.reset_index(names="row_index")
            display(selected_cases[[
                "row_index", "case_group", "y_true", "predicted_probability",
                "model_decision_threshold", "predicted_alert", "explained",
                "top_rule_name", "top_rule_strength", "rule_names", "audit_evidence",
            ]])
            quality_frame.to_csv(output_dir / "ieee_explanation_quality.csv", index=False)
            rule_quality.to_csv(output_dir / "ieee_test_rule_quality.csv", index=False)
            selected_cases.to_json(
                output_dir / "ieee_explanation_cases.json", orient="records", indent=2, force_ascii=False
            )
            lineage_path = write_upstream_lineage(
                output_dir,
                "05_IEEE_CIS_Rule_Explanation_Evaluation",
                ["ieee_explanation_quality.csv", "ieee_test_rule_quality.csv", "ieee_explanation_cases.json"],
                PROJECT_ROOT / "configs/ieee_cis.yaml",
            )
            print({"upstream_lineage": str(lineage_path)})
            """),
            md("## Takeaways"),
            code("""
            display(Markdown(
                f"- Alert explanation coverage: **{quality['explanation_coverage_alerts']:.3f}**.\\n"
                f"- Explained-alert precision gain: **{quality['explained_alert_precision_gain']:.3f}** "
                f"(95% CI [{quality['precision_gain_ci_low']:.3f}, {quality['precision_gain_ci_high']:.3f}]).\\n"
                f"- Unsupported-alert rate: **{quality['unsupported_alert_rate']:.3f}** "
                f"({int(quality['unsupported_alert_count'])}/{int(quality['predicted_alert_count'])} alerts).\\n"
                "- Overall consistency is secondary because non-alert rows dominate this imbalanced task. "
                "These are selective rule-evidence diagnostics, not causal or model-faithful explanations."
            ))
            """),
        ],
    )


def build_ablation() -> nbf.NotebookNode:
    return notebook(
        "06 - IEEE-CIS Logical Rule Ablation",
        [
            code(SETUP),
            md("""
            ## Thiết lập

            Ablation giữ nguyên frozen IEEE predictor, calibrated probabilities và threshold.
            Chỉ rule subset thay đổi giữa các điều kiện. Đây là post-hoc diagnostic trên locked test,
            không phải bước chọn một final rule set mới.
            """),
            code(frozen_artifact_preflight("ieee_cis")),
            code(FROZEN_IEEE_SETUP),
            code("""
            from src.logic import FraudRuleEngine

            output_dir = OUTPUT_BASE / "06_ieee_cis_rule_ablation"
            output_dir.mkdir(parents=True, exist_ok=True)
            target = config["dataset"]["target_column"]
            engine = FraudRuleEngine(config["logic"]["rules"]).fit(prepared.train_frame, target)
            truth = engine.evaluate(prepared.test_frame)
            predicted_alert = probabilities >= threshold
            activation = float(config["logic"]["activation_threshold"])

            def score_subset(name, columns, activation_threshold=activation):
                rule_score = truth[columns].max(axis=1).to_numpy(float) if columns else np.zeros(len(truth))
                explained = rule_score >= activation_threshold
                explained_alert = explained & predicted_alert
                evidence_without_alert = explained & ~predicted_alert
                alert_count = int(predicted_alert.sum())
                non_alert_count = int((~predicted_alert).sum())
                explained_count = int(explained.sum())
                explained_alert_count = int(explained_alert.sum())
                explained_alert_fraud_count = int(prepared.y_test[explained_alert].sum())
                base_precision = float(prepared.y_test[predicted_alert].mean()) if alert_count else np.nan
                explained_precision = (
                    float(prepared.y_test[explained_alert].mean()) if explained_alert_count else np.nan
                )
                alert_support_rate = explained_alert_count / alert_count if alert_count else np.nan
                non_alert_no_evidence_rate = (
                    float((~explained & ~predicted_alert).sum()) / non_alert_count
                    if non_alert_count else np.nan
                )
                return {
                    "ablation": name, "activation_threshold": activation_threshold, "rule_count": len(columns),
                    "analysis_role": "post_hoc_locked_test_diagnostic",
                    "test_rows": len(truth),
                    "predicted_alert_count": alert_count,
                    "explained_count": explained_count,
                    "explained_alert_count": explained_alert_count,
                    "explained_alert_fraud_count": explained_alert_fraud_count,
                    "coverage_all": float(explained.mean()),
                    "coverage_alerts": alert_support_rate,
                    "unsupported_alert_rate": 1.0 - alert_support_rate if alert_count else np.nan,
                    "rule_evidence_without_alert_rate": (
                        float(evidence_without_alert.sum()) / non_alert_count if non_alert_count else np.nan
                    ),
                    "explained_alert_precision": explained_precision,
                    "all_alert_precision": base_precision,
                    "precision_gain": (
                        explained_precision - base_precision
                        if np.isfinite(explained_precision) and np.isfinite(base_precision) else np.nan
                    ),
                    "prediction_rule_consistency": float((predicted_alert == explained).mean()),
                    "balanced_prediction_rule_consistency": (
                        0.5 * (alert_support_rate + non_alert_no_evidence_rate)
                        if np.isfinite(alert_support_rate) and np.isfinite(non_alert_no_evidence_rate)
                        else np.nan
                    ),
                }
            """),
            md("## Results"),
            code("""
            all_rules = truth.columns.tolist()
            rows = [score_subset("full_rule_set", all_rules), score_subset("no_rules", [])]
            rows += [score_subset(f"only:{rule}", [rule]) for rule in all_rules]
            rows += [score_subset(f"without:{rule}", [item for item in all_rules if item != rule]) for rule in all_rules]
            rows += [score_subset("full_rule_set_sensitivity", all_rules, value) for value in (0.50, 0.60, 0.70, 0.80)]
            ablation = pd.DataFrame(rows)
            display(ablation.round(4))
            ablation.to_csv(output_dir / "ieee_rule_ablation.csv", index=False)
            lineage_path = write_upstream_lineage(
                output_dir,
                "06_IEEE_CIS_Rule_Ablation",
                ["ieee_rule_ablation.csv"],
                PROJECT_ROOT / "configs/ieee_cis.yaml",
            )
            print({"upstream_lineage": str(lineage_path)})
            """),
            md("## Takeaways"),
            code("""
            full = ablation.query("ablation == 'full_rule_set'").iloc[0]
            display(Markdown(
                f"- Full-set alert coverage: **{full['coverage_alerts']:.3f}**.\\n"
                f"- Full-set precision gain: **{full['precision_gain']:.3f}**.\\n"
                f"- Full-set explained-alert denominator: **{int(full['explained_alert_count'])}** alerts.\\n"
                "- Leave-one-out and sensitivity rows are post-hoc locked-test diagnostics. "
                "No condition is selected as a new final rule set from these results."
            ))
            """),
        ],
    )


def build_baf_generalization() -> nbf.NotebookNode:
    return notebook(
        "07 - BAF Cross-Dataset Rule and Explanation Replication",
        [
            code(SETUP),
            md("""
            ## Thiết lập

            Notebook chỉ đọc frozen BAF reference predictor từ Notebook 03. Rules fit trên train months 0-4,
            được kiểm tra trên validation month 5 và locked test months 6-7.
            Đây là replication/portability evaluation với BAF-specific predictor và rule base, không phải
            việc chuyển nguyên model hoặc rules từ IEEE-CIS. BAF là privacy-preserving synthetic benchmark.
            """),
            code(frozen_artifact_preflight("baf")),
            code(FROZEN_BAF_SETUP),
            md("## Rule and explanation results"),
            code("""
            from src.explanation import (
                RuleExplainer, bootstrap_explanation_precision_gain,
                explanation_quality_metrics, rule_quality_table,
            )
            from src.logic import FraudKnowledgeBase, FraudRuleEngine

            output_dir = OUTPUT_BASE / "07_baf_ltn_generalization"
            output_dir.mkdir(parents=True, exist_ok=True)
            target = config["dataset"]["target_column"]
            engine = FraudRuleEngine(config["logic"]["rules"]).fit(prepared.train_frame, target)
            knowledge_base = FraudKnowledgeBase(engine)
            fitted_thresholds = engine.fitted_thresholds()
            rationale_rows = []
            for definition in config["logic"]["rules"]:
                operators = [condition["operator"] for condition in definition["conditions"]]
                rationale_rows.append({
                    "rule": definition["name"],
                    "description": definition.get("description", ""),
                    "knowledge_type": (
                        "data-informed fuzzy hypothesis" if "category_risk" in operators
                        else "domain hypothesis with train-fitted thresholds"
                        if any(operator.endswith("_quantile") for operator in operators)
                        else "configured domain hypothesis"
                    ),
                    "threshold_source": (
                        "training-split quantile plus any explicit configured condition"
                        if any(operator.endswith("_quantile") for operator in operators)
                        else "explicit configured value"
                    ),
                    "limitation": (
                        "BAF-specific association; this rule is not transferred from IEEE-CIS and is not causal."
                    ),
                })
            rule_rationale = pd.DataFrame(rationale_rows)
            validation_truth = engine.evaluate(prepared.validation_frame)
            test_truth = engine.evaluate(prepared.test_frame)
            activation = float(config["logic"]["activation_threshold"])
            validation_quality = rule_quality_table(validation_truth, prepared.y_validation, activation).assign(split="validation")
            test_quality = rule_quality_table(test_truth, prepared.y_test, activation).assign(split="test")
            rule_quality = pd.concat([validation_quality, test_quality], ignore_index=True)
            satisfaction = pd.DataFrame([
                {"split": "train", **knowledge_base.satisfaction_breakdown(prepared.train_frame, target)},
                {"split": "validation", **knowledge_base.satisfaction_breakdown(prepared.validation_frame, target)},
                {"split": "test", **knowledge_base.satisfaction_breakdown(prepared.test_frame, target)},
            ])
            explainer = RuleExplainer(engine, activation, config["logic"]["top_k_rules"])
            explanations = explainer.explain(prepared.test_frame, probabilities, threshold)
            quality = explanation_quality_metrics(explanations, prepared.y_test, probabilities, threshold)
            quality.update(bootstrap_explanation_precision_gain(
                explanations, prepared.y_test, probabilities, threshold,
                n_bootstrap=config["evaluation"]["bootstrap_iterations"], seed=config["project"]["seed"],
            ))
            explanation_quality = pd.DataFrame([quality])
            display(
                fitted_thresholds,
                rule_rationale,
                rule_quality.round(4),
                satisfaction.round(4),
                explanation_quality.round(4),
            )
            rule_quality.to_csv(output_dir / "baf_rule_quality.csv", index=False)
            satisfaction.to_csv(output_dir / "baf_knowledge_base_satisfaction.csv", index=False)
            explanation_quality.to_csv(output_dir / "baf_explanation_quality.csv", index=False)
            fitted_thresholds.to_csv(output_dir / "baf_fitted_rule_thresholds.csv", index=False)
            rule_rationale.to_csv(output_dir / "baf_rule_rationale.csv", index=False)
            """),
            code("""
            stability = validation_quality.merge(test_quality, on="rule", suffixes=("_validation", "_test"))
            stability["coverage_delta"] = stability["coverage_test"] - stability["coverage_validation"]
            stability["lift_delta"] = stability["lift_test"] - stability["lift_validation"]
            display(stability[["rule", "coverage_delta", "lift_delta"]].round(4))
            stability.to_csv(output_dir / "baf_rule_stability.csv", index=False)
            lineage_path = write_upstream_lineage(
                output_dir,
                "07_BAF_Cross_Dataset_Rule_and_Explanation_Replication",
                [
                    "baf_rule_quality.csv", "baf_knowledge_base_satisfaction.csv",
                    "baf_explanation_quality.csv", "baf_fitted_rule_thresholds.csv",
                    "baf_rule_rationale.csv", "baf_rule_stability.csv",
                ],
                PROJECT_ROOT / "configs/baf.yaml",
            )
            print({"upstream_lineage": str(lineage_path)})
            """),
            md("## Takeaways"),
            code("""
            eligible_rules = test_quality.dropna(subset=["lift"]).query("active_count > 0")
            strongest_rule = eligible_rules.sort_values("lift", ascending=False).iloc[0]
            display(Markdown(
                f"- Frozen predictor: **{artifact['manifest']['model']}**, seed **{artifact['manifest']['reference_seed']}**.\\n"
                f"- Highest observed BAF test lift: **{strongest_rule['rule']} = {strongest_rule['lift']:.3f}** "
                f"with **{int(strongest_rule['active_count'])}** active rows.\\n"
                f"- Alert explanation coverage: **{quality['explanation_coverage_alerts']:.3f}**; "
                f"unsupported-alert rate: **{quality['unsupported_alert_rate']:.3f}**.\\n"
                "- This is BAF-specific framework replication. It does not establish model/rule transfer, causality, or production generalization."
            ))
            """),
        ],
    )


def build_synthesis() -> nbf.NotebookNode:
    return notebook(
        "08 - Thesis Evidence Synthesis Across IEEE-CIS and BAF",
        [
            code(SETUP),
            md("""
            ## Thiết lập

            Notebook tổng hợp CSV/JSON từ các notebook trước, không train model và không thay đổi threshold.
            Attach outputs mới nhất của Notebook 02-07 bằng Add Input khi chạy trên Kaggle. Reference model
            luôn được đọc từ frozen manifest đã chọn bằng validation; test không được dùng để chọn lại model,
            calibration, threshold hoặc rule subset.
            """),
            code("""
            from src.artifacts import (
                find_result_file, load_frozen_reference_artifact, sha256_file, stable_config_hash,
            )
            from src.data import load_config
            from src.evaluation import expected_calibration_error
            from sklearn.calibration import calibration_curve
            from sklearn.metrics import (
                average_precision_score, brier_score_loss, confusion_matrix,
                log_loss, precision_recall_curve,
            )

            ieee_config_path = PROJECT_ROOT / "configs/ieee_cis.yaml"
            baf_config_path = PROJECT_ROOT / "configs/baf.yaml"
            ieee_config = load_config(ieee_config_path)
            baf_config = load_config(baf_config_path)
            ieee_artifact = load_frozen_reference_artifact("ieee_cis", expected_config=ieee_config, search_roots=INPUT_ROOTS)
            baf_artifact = load_frozen_reference_artifact("baf", expected_config=baf_config, search_roots=INPUT_ROOTS)
            artifacts = {"IEEE-CIS": ieee_artifact, "BAF": baf_artifact}
            configs = {"IEEE-CIS": ieee_config, "BAF": baf_config}
            config_paths = {"IEEE-CIS": ieee_config_path, "BAF": baf_config_path}
            for dataset, frozen in artifacts.items():
                if bool(frozen["manifest"]["quick_run"]) != QUICK_RUN:
                    raise ValueError(f"{dataset} frozen artifact mode does not match Notebook 08")
                if int(frozen["manifest"]["reference_seed"]) != int(configs[dataset]["evaluation"]["reference_seed"]):
                    raise ValueError(f"{dataset} frozen artifact reference seed violates the locked protocol")
                if not QUICK_RUN and str(frozen["manifest"].get("data_source", "")).lower() == "synthetic":
                    raise ValueError(f"{dataset} uses a synthetic-fallback artifact and cannot support final thesis claims")

            ieee_benchmark_root = ieee_artifact["manifest_path"].parent
            baf_benchmark_root = baf_artifact["manifest_path"].parent
            output_dir = OUTPUT_BASE / "08_cross_dataset_result_synthesis"
            output_dir.mkdir(parents=True, exist_ok=True)

            required = {
                "ieee_predictive": ieee_benchmark_root / "predictive_metrics_summary.csv",
                "baf_predictive": baf_benchmark_root / "predictive_metrics_summary.csv",
                "ieee_bootstrap": ieee_benchmark_root / "paired_bootstrap_model_differences.csv",
                "baf_bootstrap": baf_benchmark_root / "paired_bootstrap_model_differences.csv",
                "ieee_rules": find_result_file("ieee_rule_quality.csv", INPUT_ROOTS, "04_ieee_cis_ltn_rule_analysis"),
                "ieee_satisfaction": find_result_file("ieee_knowledge_base_satisfaction.csv", INPUT_ROOTS, "04_ieee_cis_ltn_rule_analysis"),
                "ieee_stability": find_result_file("ieee_rule_stability.csv", INPUT_ROOTS, "04_ieee_cis_ltn_rule_analysis"),
                "ieee_thresholds": find_result_file("ieee_fitted_rule_thresholds.csv", INPUT_ROOTS, "04_ieee_cis_ltn_rule_analysis"),
                "ieee_rationale": find_result_file("ieee_rule_rationale.csv", INPUT_ROOTS, "04_ieee_cis_ltn_rule_analysis"),
                "ieee_rule_lineage": find_result_file("upstream_lineage.json", INPUT_ROOTS, "04_ieee_cis_ltn_rule_analysis"),
                "ieee_explanations": find_result_file("ieee_explanation_quality.csv", INPUT_ROOTS, "05_ieee_cis_rule_explanation_evaluation"),
                "ieee_explanation_lineage": find_result_file("upstream_lineage.json", INPUT_ROOTS, "05_ieee_cis_rule_explanation_evaluation"),
                "ieee_ablation": find_result_file("ieee_rule_ablation.csv", INPUT_ROOTS, "06_ieee_cis_rule_ablation"),
                "ieee_ablation_lineage": find_result_file("upstream_lineage.json", INPUT_ROOTS, "06_ieee_cis_rule_ablation"),
                "baf_rules": find_result_file("baf_rule_quality.csv", INPUT_ROOTS, "07_baf_ltn_generalization"),
                "baf_explanations": find_result_file("baf_explanation_quality.csv", INPUT_ROOTS, "07_baf_ltn_generalization"),
                "baf_satisfaction": find_result_file("baf_knowledge_base_satisfaction.csv", INPUT_ROOTS, "07_baf_ltn_generalization"),
                "baf_stability": find_result_file("baf_rule_stability.csv", INPUT_ROOTS, "07_baf_ltn_generalization"),
                "baf_thresholds": find_result_file("baf_fitted_rule_thresholds.csv", INPUT_ROOTS, "07_baf_ltn_generalization"),
                "baf_rationale": find_result_file("baf_rule_rationale.csv", INPUT_ROOTS, "07_baf_ltn_generalization"),
                "baf_replication_lineage": find_result_file("upstream_lineage.json", INPUT_ROOTS, "07_baf_ltn_generalization"),
            }
            missing = [str(path) for path in required.values() if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Required upstream outputs are missing: {missing}")

            manifest_items = {
                **required,
                "ieee_frozen_manifest": ieee_artifact["manifest_path"],
                "ieee_frozen_artifact": ieee_artifact["artifact_path"],
                "baf_frozen_manifest": baf_artifact["manifest_path"],
                "baf_frozen_artifact": baf_artifact["artifact_path"],
            }
            input_manifest = pd.DataFrame([
                {
                    "artifact": name,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for name, path in manifest_items.items()
            ])
            display(input_manifest)
            input_manifest.to_csv(output_dir / "input_artifact_manifest.csv", index=False)

            lineage_specs = [
                {
                    "dataset": "IEEE-CIS", "stage": "04", "path": required["ieee_rule_lineage"],
                    "notebook_id": "04_IEEE_CIS_LTN_Rule_Analysis", "frozen": None,
                },
                {
                    "dataset": "IEEE-CIS", "stage": "05", "path": required["ieee_explanation_lineage"],
                    "notebook_id": "05_IEEE_CIS_Rule_Explanation_Evaluation", "frozen": ieee_artifact,
                },
                {
                    "dataset": "IEEE-CIS", "stage": "06", "path": required["ieee_ablation_lineage"],
                    "notebook_id": "06_IEEE_CIS_Rule_Ablation", "frozen": ieee_artifact,
                },
                {
                    "dataset": "BAF", "stage": "07", "path": required["baf_replication_lineage"],
                    "notebook_id": "07_BAF_Cross_Dataset_Rule_and_Explanation_Replication", "frozen": baf_artifact,
                },
            ]
            lineage_rows = []
            for spec in lineage_specs:
                dataset = spec["dataset"]
                lineage_path = spec["path"]
                frozen = spec["frozen"]
                payload = json.loads(lineage_path.read_text(encoding="utf-8"))
                output_files = list(payload.get("output_files", []))
                output_hashes = dict(payload.get("output_sha256", {}))
                outputs_present = bool(output_files) and all(
                    (lineage_path.parent / name).is_file() for name in output_files
                )
                outputs_hash_bound = (
                    outputs_present
                    and set(output_files) == set(output_hashes)
                    and all(
                        sha256_file(lineage_path.parent / name) == output_hashes[name]
                        for name in output_files
                    )
                )
                checks = {
                    "notebook_match": payload.get("notebook_id") == spec["notebook_id"],
                    "dataset_match": str(payload.get("dataset_name", "")).lower() == str(configs[dataset]["dataset"]["name"]).lower(),
                    "config_match": payload.get("config_sha256") == stable_config_hash(configs[dataset]),
                    "source_fingerprint_match": payload.get("audit_source_sha256") == audit_pipeline_fingerprint(config_paths[dataset]),
                    "manifest_match": (
                        True if frozen is None else
                        payload.get("frozen_manifest_sha256") == sha256_file(frozen["manifest_path"])
                    ),
                    "artifact_match": (
                        True if frozen is None else
                        payload.get("frozen_artifact_sha256") == sha256_file(frozen["artifact_path"])
                    ),
                    "reference_seed_match": (
                        True if frozen is None else
                        int(payload.get("reference_seed", -1)) == int(configs[dataset]["evaluation"]["reference_seed"])
                    ),
                    "mode_match": bool(payload.get("quick_run")) == QUICK_RUN,
                    "data_source_acceptable": QUICK_RUN or str(payload.get("data_source", "")).lower() != "synthetic",
                    "declared_outputs_present": outputs_present,
                    "output_hashes_match": outputs_hash_bound,
                }
                lineage_rows.append({
                    "dataset": dataset, "stage": spec["stage"],
                    "git_commit": payload.get("git_commit"), "lineage": str(lineage_path), **checks,
                })
            lineage_audit = pd.DataFrame(lineage_rows)
            display(lineage_audit)
            audit_checks = lineage_audit.drop(columns=["dataset", "stage", "git_commit", "lineage"])
            if not audit_checks.all(axis=None):
                raise ValueError("At least one Notebook 04-07 output failed provenance or alignment checks")
            lineage_audit.to_csv(output_dir / "upstream_lineage_audit.csv", index=False)
            """),
            md("## Predictive evidence"),
            code("""
            predictive = pd.concat([
                pd.read_csv(required["ieee_predictive"]).assign(dataset="IEEE-CIS"),
                pd.read_csv(required["baf_predictive"]).assign(dataset="BAF"),
            ], ignore_index=True)
            predictive_test = predictive.query("split == 'test'").copy()
            reference_keys = {
                "IEEE-CIS": ieee_artifact["manifest"]["model_key"],
                "BAF": baf_artifact["manifest"]["model_key"],
            }
            reference_rows = pd.concat([
                predictive.loc[
                    (predictive["dataset"] == dataset)
                    & (predictive["model_key"] == model_key)
                    & (predictive["split"] == "test")
                ]
                for dataset, model_key in reference_keys.items()
            ], ignore_index=True)
            if len(reference_rows) != len(reference_keys):
                raise ValueError("Frozen reference model rows are missing or duplicated in predictive summaries")
            prevalence = {
                dataset: float(frozen["y_test"].mean()) for dataset, frozen in artifacts.items()
            }
            reference_rows["test_prevalence"] = reference_rows["dataset"].map(prevalence)
            reference_rows["pr_auc_over_prevalence"] = (
                reference_rows["raw_pr_auc_mean"] / reference_rows["test_prevalence"]
            )

            reference_split = predictive.loc[
                predictive.apply(lambda row: row["model_key"] == reference_keys[row["dataset"]], axis=1)
            ].copy()
            reference_gap = reference_split.pivot(
                index=["dataset", "model", "model_key"], columns="split", values="raw_pr_auc_mean"
            ).reset_index()
            reference_gap["validation_to_test_gap"] = reference_gap["test"] - reference_gap["validation"]

            display(predictive_test.round(4), reference_rows.round(4), reference_gap.round(4))
            predictive_test.to_csv(output_dir / "table_cross_dataset_predictive_test.csv", index=False)
            reference_rows.to_csv(output_dir / "table_frozen_reference_predictive_test.csv", index=False)
            reference_gap.to_csv(output_dir / "table_validation_to_test_gap.csv", index=False)

            bootstrap = pd.concat([
                pd.read_csv(required["ieee_bootstrap"]).assign(dataset="IEEE-CIS"),
                pd.read_csv(required["baf_bootstrap"]).assign(dataset="BAF"),
            ], ignore_index=True)
            bootstrap["uncertainty_scope"] = "paired row-level bootstrap on frozen reference-seed predictions"
            display(bootstrap.round(5))
            bootstrap.to_csv(output_dir / "table_cross_dataset_predictive_bootstrap.csv", index=False)

            confusion_rows = []
            calibration_rows = []
            for dataset, frozen in artifacts.items():
                labels = frozen["y_test"].astype(int)
                raw_probability = frozen["test_raw_probability"].astype(float)
                calibrated_probability = frozen["test_probability"].astype(float)
                decision_threshold = float(frozen["manifest"]["threshold"])
                tn, fp, fn, tp = confusion_matrix(
                    labels, calibrated_probability >= decision_threshold, labels=[0, 1]
                ).ravel()
                confusion_rows.append({
                    "dataset": dataset, "threshold": decision_threshold,
                    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                    "predicted_alert_count": int(fp + tp), "test_rows": len(labels),
                })
                for probability_type, probability in (
                    ("raw", raw_probability), ("calibrated", calibrated_probability)
                ):
                    calibration_rows.append({
                        "dataset": dataset,
                        "probability_type": probability_type,
                        "pr_auc": average_precision_score(labels, probability),
                        "brier": brier_score_loss(labels, probability),
                        "ece": expected_calibration_error(labels, probability, n_bins=15),
                        "nll": log_loss(labels, np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1]),
                    })
            confusion_table = pd.DataFrame(confusion_rows)
            calibration_table = pd.DataFrame(calibration_rows)
            display(confusion_table, calibration_table.round(5))
            confusion_table.to_csv(output_dir / "table_frozen_reference_confusion_counts.csv", index=False)
            calibration_table.to_csv(output_dir / "table_raw_vs_calibrated_probability_metrics.csv", index=False)
            """),
            code("""
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)
            model_color = "#4C72B0"
            for axis, dataset in zip(axes, ("IEEE-CIS", "BAF")):
                subset = predictive_test.loc[predictive_test["dataset"] == dataset].sort_values("raw_pr_auc_mean")
                positions = np.arange(len(subset))
                axis.bar(
                    positions, subset["raw_pr_auc_mean"],
                    yerr=subset["raw_pr_auc_std"].fillna(0),
                    color=model_color, edgecolor="#2F4B66", capsize=4,
                )
                axis.axhline(prevalence[dataset], color="#4A4A4A", linestyle="--", linewidth=1.2,
                             label=f"Test prevalence = {prevalence[dataset]:.4f}")
                axis.set_xticks(positions, subset["model"], rotation=18, ha="right")
                axis.set_title(f"{dataset} test raw PR-AUC")
                axis.set_ylabel("Raw PR-AUC (mean ± SD across 3 seeds)")
                axis.legend(loc="upper left")
                axis.margins(x=0.10)
            fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.88, wspace=0.26)
            fig.savefig(output_dir / "figure_predictive_performance_by_dataset.png", dpi=180, bbox_inches="tight")
            plt.show()

            fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.4))
            for axis, (dataset, frozen) in zip(axes, artifacts.items()):
                labels = frozen["y_test"].astype(int)
                raw_probability = frozen["test_raw_probability"].astype(float)
                calibrated_probability = frozen["test_probability"].astype(float)
                for name, probability, color, line_style in (
                    ("Raw score", raw_probability, "#4C72B0", "-"),
                    ("Calibrated probability", calibrated_probability, "#DD8452", "--"),
                ):
                    precision, recall, _ = precision_recall_curve(labels, probability)
                    axis.plot(recall, precision, color=color, linestyle=line_style, linewidth=2, label=name)
                axis.axhline(prevalence[dataset], color="#4A4A4A", linestyle=":", label="Prevalence")
                axis.set_title(f"{dataset} frozen-reference precision-recall curve")
                axis.set_xlabel("Recall")
                axis.set_ylabel("Precision")
                axis.set_xlim(0, 1)
                axis.set_ylim(0, 1)
                axis.legend(loc="upper right")
            fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.88, wspace=0.38)
            fig.savefig(output_dir / "figure_frozen_reference_pr_curves.png", dpi=180, bbox_inches="tight")
            plt.show()

            fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.4))
            for axis, (dataset, frozen) in zip(axes, artifacts.items()):
                labels = frozen["y_test"].astype(int)
                for name, probability, color, marker in (
                    ("Raw score", frozen["test_raw_probability"], "#4C72B0", "o"),
                    ("Calibrated probability", frozen["test_probability"], "#DD8452", "s"),
                ):
                    observed, predicted = calibration_curve(labels, probability, n_bins=10, strategy="quantile")
                    axis.plot(predicted, observed, color=color, marker=marker, linewidth=1.8, label=name)
                axis.plot([0, 1], [0, 1], color="#4A4A4A", linestyle="--", linewidth=1, label="Ideal")
                axis.set_title(f"{dataset} reliability diagram")
                axis.set_xlabel("Mean predicted probability")
                axis.set_ylabel("Observed fraud rate")
                axis.set_xlim(0, 1)
                axis.set_ylim(0, 1)
                axis.legend(loc="upper left")
            fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.88, wspace=0.38)
            fig.savefig(output_dir / "figure_frozen_reference_reliability.png", dpi=180, bbox_inches="tight")
            plt.show()
            """),
            md("## Rule, explanation and ablation evidence"),
            code("""
            rules = pd.concat([
                pd.read_csv(required["ieee_rules"]).assign(dataset="IEEE-CIS"),
                pd.read_csv(required["baf_rules"]).assign(dataset="BAF"),
            ], ignore_index=True)
            explanations = pd.concat([
                pd.read_csv(required["ieee_explanations"]).assign(dataset="IEEE-CIS"),
                pd.read_csv(required["baf_explanations"]).assign(dataset="BAF"),
            ], ignore_index=True)
            satisfaction = pd.concat([
                pd.read_csv(required["ieee_satisfaction"]).assign(dataset="IEEE-CIS"),
                pd.read_csv(required["baf_satisfaction"]).assign(dataset="BAF"),
            ], ignore_index=True)
            stability = pd.concat([
                pd.read_csv(required["ieee_stability"]).assign(dataset="IEEE-CIS"),
                pd.read_csv(required["baf_stability"]).assign(dataset="BAF"),
            ], ignore_index=True)
            fitted_thresholds = pd.concat([
                pd.read_csv(required["ieee_thresholds"]).assign(dataset="IEEE-CIS"),
                pd.read_csv(required["baf_thresholds"]).assign(dataset="BAF"),
            ], ignore_index=True)
            rule_rationale = pd.concat([
                pd.read_csv(required["ieee_rationale"]).assign(dataset="IEEE-CIS"),
                pd.read_csv(required["baf_rationale"]).assign(dataset="BAF"),
            ], ignore_index=True)
            ablation = pd.read_csv(required["ieee_ablation"])
            display(
                rules.round(4), satisfaction.round(4), stability.round(4),
                explanations.round(4), ablation.round(4), fitted_thresholds, rule_rationale,
            )
            rules.to_csv(output_dir / "table_cross_dataset_rule_quality.csv", index=False)
            satisfaction.to_csv(output_dir / "table_cross_dataset_knowledge_base_satisfaction.csv", index=False)
            stability.to_csv(output_dir / "table_cross_dataset_rule_stability.csv", index=False)
            explanations.to_csv(output_dir / "table_cross_dataset_explanation_quality.csv", index=False)
            ablation.to_csv(output_dir / "table_ieee_rule_ablation.csv", index=False)
            fitted_thresholds.to_csv(output_dir / "table_cross_dataset_fitted_rule_thresholds.csv", index=False)
            rule_rationale.to_csv(output_dir / "table_cross_dataset_rule_rationale.csv", index=False)
            """),
            code("""
            test_rules = rules.query("split == 'test'") if "split" in rules else rules
            fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
            for axis, dataset in zip(axes, ("IEEE-CIS", "BAF")):
                subset = (
                    test_rules.loc[(test_rules["dataset"] == dataset) & test_rules["lift"].notna()]
                    .query("active_count > 0")
                    .sort_values("lift")
                )
                positions = np.arange(len(subset))
                axis.barh(positions, subset["lift"], color="#4C72B0", edgecolor="#2F4B66")
                axis.set_yticks(positions, subset["rule"])
                axis.axvline(1.0, color="#4A4A4A", linestyle="--", linewidth=1.2)
                axis.set_title(f"{dataset} test rule lift with denominators")
                axis.set_xlabel("Fraud-rate lift over test prevalence")
                axis.set_ylabel("Rule")
                axis.set_xlim(0, max(1.2, float(subset["lift"].max()) * 1.32))
                for position, (_, row) in enumerate(subset.iterrows()):
                    axis.text(
                        row["lift"] + float(subset["lift"].max()) * 0.025,
                        position,
                        f"n={int(row['active_count'])}; coverage={row['coverage']:.2%}",
                        va="center", fontsize=9,
                    )
            fig.subplots_adjust(left=0.15, right=0.98, bottom=0.11, top=0.90, wspace=0.72)
            fig.savefig(output_dir / "figure_rule_lift_with_denominators.png", dpi=180, bbox_inches="tight")
            plt.show()
            """),
            code("""
            explanation_plot = explanations.copy()
            x = np.arange(len(explanation_plot))
            width = 0.34
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
            axes[0].bar(
                x - width / 2, explanation_plot["all_alert_precision"], width,
                label="All predicted alerts", color="#9CB8D2", edgecolor="#2F4B66",
            )
            axes[0].bar(
                x + width / 2, explanation_plot["explained_alert_precision"], width,
                label="Alerts with rule evidence", color="#4C72B0", edgecolor="#2F4B66",
            )
            axes[0].set_xticks(x, explanation_plot["dataset"])
            axes[0].set_ylabel("Fraud precision")
            axes[0].set_title("Alert precision with and without rule-evidence filtering")
            axes[0].legend(loc="upper right")

            gain = explanation_plot["explained_alert_precision_gain"].to_numpy(float)
            ci_low = explanation_plot["precision_gain_ci_low"].to_numpy(float)
            ci_high = explanation_plot["precision_gain_ci_high"].to_numpy(float)
            for position, (point, low, high) in enumerate(zip(gain, ci_low, ci_high)):
                if np.isfinite([point, low, high]).all():
                    axes[1].hlines(position, low, high, color="#8C5A3C", linewidth=2)
                    axes[1].plot(point, position, "o", markersize=8, color="#DD8452")
                    axes[1].vlines([low, high], position - 0.08, position + 0.08, color="#8C5A3C")
                else:
                    axes[1].text(0, position, "undefined", ha="center", va="center", fontsize=9)
            axes[1].axvline(0, color="#4A4A4A", linestyle="--", linewidth=1)
            axes[1].set_yticks(x, explanation_plot["dataset"])
            axes[1].set_xlabel("Explained-alert precision gain (95% bootstrap CI)")
            axes[1].set_title("Selective explanation precision gain")
            for position, (_, row) in enumerate(explanation_plot.iterrows()):
                if np.isfinite(row["precision_gain_ci_high"]):
                    axes[1].annotate(
                        f"alert coverage={row['explanation_coverage_alerts']:.1%}; n={int(row['explained_alert_count'])}",
                        (row["precision_gain_ci_high"], position), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9,
                    )
            fig.subplots_adjust(left=0.10, right=0.95, bottom=0.14, top=0.88, wspace=0.38)
            fig.savefig(output_dir / "figure_cross_dataset_explanation_evidence.png", dpi=180, bbox_inches="tight")
            plt.show()
            """),
            code("""
            diagnostic = ablation.dropna(subset=["coverage_alerts", "precision_gain"]).copy()
            diagnostic["family"] = np.select(
                [
                    diagnostic["ablation"].str.startswith("only:"),
                    diagnostic["ablation"].str.startswith("without:"),
                    diagnostic["ablation"].eq("full_rule_set"),
                    diagnostic["ablation"].eq("full_rule_set_sensitivity"),
                ],
                ["only one rule", "leave one out", "full set", "activation sensitivity"],
                default="other",
            )
            fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))
            palette = {
                "only one rule": "#4C72B0", "leave one out": "#DD8452",
                "full set": "#2F4B66", "activation sensitivity": "#9B8F45",
            }
            subset_diagnostics = diagnostic.loc[diagnostic["family"].isin(["only one rule", "leave one out", "full set"])]
            for family, group in subset_diagnostics.groupby("family"):
                axes[0].scatter(
                    group["coverage_alerts"], group["precision_gain"],
                    label=family, color=palette[family], s=60, alpha=0.85,
                )
            full = diagnostic.loc[diagnostic["ablation"] == "full_rule_set"].iloc[0]
            axes[0].annotate("full rule set", (full["coverage_alerts"], full["precision_gain"]),
                             xytext=(8, 8), textcoords="offset points")
            axes[0].axhline(0, color="#4A4A4A", linestyle="--", linewidth=1)
            axes[0].set_xlabel("Alert explanation coverage")
            axes[0].set_ylabel("Explained-alert precision gain")
            axes[0].set_title("IEEE-CIS post-hoc rule-subset diagnostics")
            axes[0].legend(loc="best")

            sensitivity = diagnostic.loc[diagnostic["family"] == "activation sensitivity"].sort_values("activation_threshold")
            axes[1].plot(
                sensitivity["coverage_alerts"], sensitivity["precision_gain"],
                marker="o", color=palette["activation sensitivity"], linewidth=2,
            )
            label_offsets = [(7, 8), (7, 20), (7, -14), (7, -26)]
            for label_index, (_, row) in enumerate(sensitivity.iterrows()):
                axes[1].annotate(
                    f"t={row['activation_threshold']:.2f}",
                    (row["coverage_alerts"], row["precision_gain"]),
                    xytext=label_offsets[label_index % len(label_offsets)],
                    textcoords="offset points", fontsize=9,
                )
            axes[1].axhline(0, color="#4A4A4A", linestyle="--", linewidth=1)
            axes[1].set_xlabel("Alert explanation coverage")
            axes[1].set_ylabel("Explained-alert precision gain")
            axes[1].set_title("IEEE-CIS activation-threshold sensitivity (post-hoc)")
            fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.88, wspace=0.26)
            fig.savefig(output_dir / "figure_ieee_ablation_coverage_precision_tradeoff.png", dpi=180, bbox_inches="tight")
            plt.show()
            """),
            md("## Thesis evidence matrix"),
            code("""
            ieee_explanation = explanations.loc[explanations["dataset"] == "IEEE-CIS"].iloc[0]
            baf_explanation = explanations.loc[explanations["dataset"] == "BAF"].iloc[0]
            ieee_reference = reference_rows.loc[reference_rows["dataset"] == "IEEE-CIS"].iloc[0]
            baf_reference = reference_rows.loc[reference_rows["dataset"] == "BAF"].iloc[0]

            def enrichment_status(row):
                values = row[[
                    "explained_alert_precision_gain", "precision_gain_ci_low", "precision_gain_ci_high",
                ]].to_numpy(float)
                if not np.isfinite(values).all():
                    return "undefined"
                if float(row["precision_gain_ci_low"]) > 0:
                    return "positive"
                if float(row["precision_gain_ci_high"]) < 0:
                    return "negative"
                return "inconclusive (CI includes zero)"

            enrichment = {
                "IEEE-CIS": enrichment_status(ieee_explanation),
                "BAF": enrichment_status(baf_explanation),
            }
            positive_datasets = [name for name, status in enrichment.items() if status == "positive"]
            if len(positive_datasets) == len(enrichment):
                rq3_supported_claim = (
                    "Positive alert enrichment from rule-evidence filtering is supported on both evaluated benchmarks."
                )
            elif positive_datasets:
                remaining = [name for name in enrichment if name not in positive_datasets]
                rq3_supported_claim = (
                    f"Positive enrichment is supported for {', '.join(positive_datasets)}, but cross-dataset "
                    f"replication is not established because {', '.join(remaining)} is "
                    f"{', '.join(enrichment[name] for name in remaining)}."
                )
            else:
                status_text = "; ".join(f"{name}: {status}" for name, status in enrichment.items())
                rq3_supported_claim = (
                    f"The current intervals do not establish positive alert enrichment on either benchmark ({status_text})."
                )
            evidence_matrix = pd.DataFrame([
                {
                    "research_question": "RQ1 - Leakage-aware fraud prediction",
                    "observed_evidence": (
                        f"IEEE {ieee_reference['model']} raw PR-AUC={ieee_reference['raw_pr_auc_mean']:.4f}; "
                        f"BAF {baf_reference['model']} raw PR-AUC={baf_reference['raw_pr_auc_mean']:.4f}."
                    ),
                    "supported_claim": "The locked temporal protocol yields traceable predictive measurements and validation-selected frozen references on both benchmarks.",
                    "claim_boundary": "No SOTA, universal superiority, or production-readiness claim.",
                },
                {
                    "research_question": "RQ2 - Fuzzy knowledge representation",
                    "observed_evidence": "Train-fitted fuzzy rules, fitted thresholds, per-rule coverage/lift and balanced KB satisfaction are reported.",
                    "supported_claim": "Domain/data-informed hypotheses can be represented as an auditable fuzzy-rule layer.",
                    "claim_boundary": "LTN-inspired diagnostic, not an end-to-end co-trained LTN predictor.",
                },
                {
                    "research_question": "RQ3 - Selective alert audit value",
                    "observed_evidence": (
                        f"Precision gain: IEEE {ieee_explanation['explained_alert_precision_gain']:.3f} "
                        f"(CI {ieee_explanation['precision_gain_ci_low']:.3f} to {ieee_explanation['precision_gain_ci_high']:.3f}); "
                        f"BAF {baf_explanation['explained_alert_precision_gain']:.3f} "
                        f"(CI {baf_explanation['precision_gain_ci_low']:.3f} to {baf_explanation['precision_gain_ci_high']:.3f})."
                    ),
                    "supported_claim": rq3_supported_claim,
                    "claim_boundary": "Association/enrichment, not causal or model-faithful explanation.",
                },
                {
                    "research_question": "RQ4 - Coverage-precision trade-off",
                    "observed_evidence": "Rule-subset and activation-threshold diagnostics expose changes in alert coverage and explained-alert precision.",
                    "supported_claim": "Post-hoc diagnostics quantify how explanation coverage and precision gain vary across rule subsets and activation thresholds.",
                    "claim_boundary": "Locked-test ablation is post-hoc and cannot select a new unbiased final rule set.",
                },
                {
                    "research_question": "RQ5 - Cross-dataset applicability and reproducibility",
                    "observed_evidence": "Frozen checksums, split alignment, lineage manifests and BAF-specific replication are verified.",
                    "supported_claim": "The same audited pipeline is reproducibly executed on a second benchmark with dataset-specific rules and models.",
                    "claim_boundary": "Not transfer of the same model/rules and not external institutional validation.",
                },
            ])
            display(evidence_matrix)
            evidence_matrix.to_csv(output_dir / "table_thesis_evidence_matrix.csv", index=False)
            """),
            md("## Takeaways"),
            code("""
            display(reference_rows[[
                "dataset", "model", "raw_pr_auc_mean", "raw_pr_auc_std",
                "test_prevalence", "pr_auc_over_prevalence", "fbeta_mean", "brier_mean", "ece_mean",
            ]].round(4))
            display(Markdown(
                "- Reference models above come from frozen manifests selected on validation; Notebook 08 never reselects from test.\\n"
                "- Raw PR-AUC supports ranking; calibrated probabilities support Brier/ECE/NLL and locked-threshold decisions.\\n"
                "- The paired bootstrap is row-level uncertainty for the frozen reference seed, while three-seed SD is descriptive run variability.\\n"
                f"- Rule-evidence result: {rq3_supported_claim}\\n"
                "- BAF supports framework replication with BAF-specific rules, not model/rule transfer, causal explanation, or production generalization."
            ))
            """),
        ],
    )


def build_all_notebooks() -> dict[str, nbf.NotebookNode]:
    """Build the canonical source for every thesis notebook."""
    return {
        "01_Data_Exploration.ipynb": build_exploration(),
        "02_IEEE_CIS_Model_Benchmarks.ipynb": build_benchmark("02", "ieee_cis", "IEEE-CIS"),
        "03_BAF_Model_Benchmarks.ipynb": build_benchmark("03", "baf", "BAF"),
        "04_IEEE_CIS_LTN_Rule_Analysis.ipynb": build_rule_analysis(),
        "05_IEEE_CIS_Rule_Explanation_Evaluation.ipynb": build_explanation(),
        "06_IEEE_CIS_Rule_Ablation.ipynb": build_ablation(),
        "07_BAF_LTN_Generalization.ipynb": build_baf_generalization(),
        "08_Cross_Dataset_Result_Synthesis.ipynb": build_synthesis(),
    }


def _same_source(first: nbf.NotebookNode, second: nbf.NotebookNode) -> bool:
    def source_text(cell: nbf.NotebookNode) -> str:
        source = cell.source
        return "".join(source) if isinstance(source, list) else source

    return [
        (cell.cell_type, source_text(cell)) for cell in first.cells
    ] == [
        (cell.cell_type, source_text(cell)) for cell in second.cells
    ]


def _same_canonical_content(first: nbf.NotebookNode, second: nbf.NotebookNode) -> bool:
    """Compare canonical cell source while ignoring Kaggle execution metadata and IDs."""
    return _same_source(first, second)


def _preserve_execution_when_source_is_unchanged(
    generated: nbf.NotebookNode,
    destination: Path,
) -> bool:
    """Keep expensive executed outputs only when the complete canonical source is unchanged."""
    if not destination.exists():
        return False
    existing = nbf.read(destination, as_version=4)
    if not _same_source(generated, existing):
        return False
    generated.metadata = existing.metadata
    for generated_cell, existing_cell in zip(generated.cells, existing.cells):
        generated_cell.metadata = existing_cell.metadata
        if generated_cell.cell_type == "code":
            generated_cell.execution_count = existing_cell.get("execution_count")
            generated_cell.outputs = existing_cell.get("outputs", [])
    return True


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = build_all_notebooks()
    for filename, notebook_node in notebooks.items():
        destination = NOTEBOOK_DIR / filename
        if destination.exists():
            existing = nbf.read(destination, as_version=4)
            if _same_canonical_content(notebook_node, existing):
                nbf.validate(existing)
                print(f"Kept notebooks/{filename} byte-for-byte (canonical content unchanged)")
                continue
        for index, cell in enumerate(notebook_node.cells):
            identity = f"{filename}:{index}:{cell.cell_type}:{cell.source}".encode("utf-8")
            cell["id"] = hashlib.sha1(identity).hexdigest()[:12]
        nbf.validate(notebook_node)
        nbf.write(notebook_node, destination)
        print(f"Wrote notebooks/{filename} (canonical content changed; outputs cleared)")


if __name__ == "__main__":
    main()
