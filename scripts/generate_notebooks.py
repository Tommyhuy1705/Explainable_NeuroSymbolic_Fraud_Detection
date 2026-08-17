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


SETUP = r'''
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


def exploration_section(label: str, config_name: str, synthetic_factory: str, slug: str) -> list:
    return [
        md(f"## {label}"),
        code(f'''
        config = load_config(PROJECT_ROOT / "configs/{config_name}")
        try:
            frame = load_fraud_dataframe(config, max_rows=max_rows)
            data_source = "dataset_file"
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
        summary = pd.DataFrame({{
            "dataset": [config["dataset"]["name"]],
            "rows": [len(frame)],
            "columns": [frame.shape[1]],
            "fraud_count": [int(frame[target].sum())],
            "fraud_rate": [float(frame[target].mean())],
            "duplicate_rows": [int(frame.duplicated().sum())],
            "data_source": [data_source],
            "selected_features": [len(prepared.feature_names)],
        }})
        missing = frame.isna().mean().sort_values(ascending=False).rename("missing_fraction").to_frame()
        dataset_summaries.append(summary)
        display(summary, integrity, missing.head(20))
        integrity.to_csv(output_dir / "{slug}_split_summary.csv", index=False)
        missing.to_csv(output_dir / "{slug}_missingness.csv")
        '''),
        code(f'''
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        integrity.plot.bar(x="split", y="fraud_rate", ax=axes[0], legend=False, color="#C44E52")
        axes[0].set_title("{label.split('. ', 1)[-1]} fraud rate by split")
        axes[0].set_ylabel("Fraud rate")
        missing.head(20).sort_values("missing_fraction").plot.barh(ax=axes[1], legend=False, color="#4C72B0")
        axes[1].set_title("Top missing columns")
        axes[1].set_xlabel("Missing fraction")
        plt.tight_layout()
        fig.savefig(output_dir / "{slug}_data_quality.png", dpi=160, bbox_inches="tight")
        plt.show()
        '''),
    ]


def build_exploration() -> nbf.NotebookNode:
    cells = [
        code(SETUP),
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
    cells += exploration_section("II. BAF", "baf.yaml", "make_synthetic_baf_data", "baf")
    cells += [
        md("## III. So sánh tổng quan"),
        code("""
        combined_summary = pd.concat(dataset_summaries, ignore_index=True)
        display(combined_summary)
        combined_summary.to_csv(output_dir / "dataset_summary.csv", index=False)
        ax = combined_summary.plot.bar(x="dataset", y="fraud_rate", legend=False, color=["#4C72B0", "#55A868"])
        ax.set_title("Fraud prevalence across datasets")
        ax.set_ylabel("Fraud rate")
        plt.tight_layout()
        plt.savefig(output_dir / "dataset_fraud_rate_comparison.png", dpi=160, bbox_inches="tight")
        plt.show()
        """),
        md("## Takeaways"),
        code("""
        display(Markdown(
            "- Review every displayed split before model training.\\n"
            "- For BAF, the required final split is months **0-4 / 5 / 6-7** with no overlap.\\n"
            "- `dataset_file` means the published benchmark file was loaded; BAF itself remains a privacy-preserving synthetic benchmark."
        ))
        """),
    ]
    return notebook("01 - Multi-Dataset Exploration and Split Audit", cells)


def build_benchmark(number: str, dataset_name: str, label: str) -> nbf.NotebookNode:
    return notebook(
        f"{number} - {label} Predictive Model Benchmarks",
        [
            code(SETUP),
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
            Đây là LTN-style explanation layer, không phải end-to-end LTN predictor.
            """),
            code("""
            from src.data import load_config, prepare_dataset
            from src.experiment import load_experiment_data
            from src.explanation import rule_quality_table
            from src.logic import FraudKnowledgeBase, FraudRuleEngine

            config = load_config(PROJECT_ROOT / "configs/ieee_cis.yaml")
            output_dir = OUTPUT_BASE / "04_ieee_cis_ltn_rule_analysis"
            output_dir.mkdir(parents=True, exist_ok=True)
            frame, data_source = load_experiment_data(
                config, max_rows=12000 if QUICK_RUN else None,
                synthetic_fallback=ALLOW_SYNTHETIC_FALLBACK,
                synthetic_rows=12000 if QUICK_RUN else 6000,
            )
            prepared = prepare_dataset(frame, config)
            target = config["dataset"]["target_column"]
            engine = FraudRuleEngine(config["logic"]["rules"]).fit(prepared.train_frame, target)
            knowledge_base = FraudKnowledgeBase(engine)
            print({"data_source": data_source, "active_rules": len(engine.rules), "skipped": engine.skipped_rules})
            display(engine.fitted_thresholds())
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
            engine.fitted_thresholds().to_csv(output_dir / "ieee_fitted_rule_thresholds.csv", index=False)
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
            """),
            md("## Takeaways"),
            code("""
            strongest = test_quality.sort_values("lift", ascending=False).iloc[0]
            test_satisfaction = satisfaction.query("split == 'test'").iloc[0]
            display(Markdown(
                f"- Highest test rule lift: **{strongest['rule']} = {strongest['lift']:.3f}**.\\n"
                f"- Test balanced knowledge-base satisfaction: **{test_satisfaction['balanced_satisfaction']:.3f}**.\\n"
                "- Rule lift and satisfaction measure association/logic agreement, not causality."
            ))
            """),
        ],
    )


FROZEN_IEEE_SETUP = '''
from src.artifacts import assert_frozen_alignment, load_frozen_reference_artifact
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
'''


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
            cases["case_type"] = np.select(
                [predicted_alert & (prepared.y_test == 1), predicted_alert & (prepared.y_test == 0),
                 (~predicted_alert) & (prepared.y_test == 1)],
                ["true_positive", "false_positive", "false_negative"], default="true_negative",
            )
            selected_cases = pd.concat([
                group.sort_values("predicted_probability", ascending=False).head(3)
                for _, group in cases.groupby("case_type")
            ])
            display(selected_cases[["case_type", "y_true", "predicted_probability", "rule_names", "explanation"]])
            quality_frame.to_csv(output_dir / "ieee_explanation_quality.csv", index=False)
            rule_quality.to_csv(output_dir / "ieee_test_rule_quality.csv", index=False)
            selected_cases.to_json(output_dir / "ieee_explanation_cases.json", orient="records", indent=2)
            """),
            md("## Takeaways"),
            code("""
            display(Markdown(
                f"- Alert explanation coverage: **{quality['explanation_coverage_alerts']:.3f}**.\\n"
                f"- Explained-alert precision gain: **{quality['explained_alert_precision_gain']:.3f}** "
                f"(95% CI [{quality['precision_gain_ci_low']:.3f}, {quality['precision_gain_ci_high']:.3f}]).\\n"
                f"- Contradiction rate: **{quality['contradiction_rate']:.3f}**.\\n"
                "- These are rule-evidence diagnostics, not causal explanations."
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
            Chỉ rule subset thay đổi giữa các điều kiện.
            """),
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
                explained_precision = prepared.y_test[explained_alert].mean() if explained_alert.any() else 0.0
                base_precision = prepared.y_test[predicted_alert].mean() if predicted_alert.any() else 0.0
                return {
                    "ablation": name, "activation_threshold": activation_threshold, "rule_count": len(columns),
                    "coverage_all": explained.mean(),
                    "coverage_alerts": explained_alert.sum() / max(predicted_alert.sum(), 1),
                    "explained_alert_precision": explained_precision,
                    "precision_gain": explained_precision - base_precision,
                    "prediction_rule_consistency": (predicted_alert == explained).mean(),
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
            """),
            md("## Takeaways"),
            code("""
            full = ablation.query("ablation == 'full_rule_set'").iloc[0]
            best = ablation[ablation["ablation"].str.startswith("without:")].sort_values("precision_gain", ascending=False).iloc[0]
            display(Markdown(
                f"- Full-set alert coverage: **{full['coverage_alerts']:.3f}**.\\n"
                f"- Full-set precision gain: **{full['precision_gain']:.3f}**.\\n"
                f"- Best leave-one-out condition: **{best['ablation']}**.\\n"
                "- Any change is attributable to the rule subset because predictor outputs are frozen."
            ))
            """),
        ],
    )


def build_baf_generalization() -> nbf.NotebookNode:
    return notebook(
        "07 - BAF LTN and Explanation Generalization",
        [
            code(SETUP),
            md("""
            ## Thiết lập

            Notebook chỉ đọc frozen BAF reference predictor từ Notebook 03. Rules fit trên train months 0-4,
            được kiểm tra trên validation month 5 và locked test months 6-7.
            """),
            code("""
            from src.artifacts import assert_frozen_alignment, load_frozen_reference_artifact
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
            probabilities = artifact["test_probability"]
            threshold = float(artifact["manifest"]["threshold"])
            print({"data_source": data_source, "manifest": artifact["manifest"]})
            """),
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
            display(rule_quality.round(4), satisfaction.round(4), explanation_quality.round(4))
            rule_quality.to_csv(output_dir / "baf_rule_quality.csv", index=False)
            satisfaction.to_csv(output_dir / "baf_knowledge_base_satisfaction.csv", index=False)
            explanation_quality.to_csv(output_dir / "baf_explanation_quality.csv", index=False)
            """),
            code("""
            stability = validation_quality.merge(test_quality, on="rule", suffixes=("_validation", "_test"))
            stability["coverage_delta"] = stability["coverage_test"] - stability["coverage_validation"]
            stability["lift_delta"] = stability["lift_test"] - stability["lift_validation"]
            display(stability[["rule", "coverage_delta", "lift_delta"]].round(4))
            stability.to_csv(output_dir / "baf_rule_stability.csv", index=False)
            """),
            md("## Takeaways"),
            code("""
            best_rule = test_quality.sort_values("lift", ascending=False).iloc[0]
            display(Markdown(
                f"- Frozen predictor: **{artifact['manifest']['model']}**, seed **{artifact['manifest']['reference_seed']}**.\\n"
                f"- Highest BAF test rule lift: **{best_rule['rule']} = {best_rule['lift']:.3f}**.\\n"
                f"- Alert explanation coverage: **{quality['explanation_coverage_alerts']:.3f}**.\\n"
                "- Cross-dataset evidence is bounded to these public benchmark distributions."
            ))
            """),
        ],
    )


def build_synthesis() -> nbf.NotebookNode:
    return notebook(
        "08 - Cross-Dataset Result Synthesis",
        [
            code(SETUP),
            md("""
            ## Thiết lập

            Notebook tổng hợp CSV/JSON từ các notebook trước, không train model và không thay đổi threshold.
            Attach outputs của Notebook 02-07 bằng Add Input khi chạy trên Kaggle.
            """),
            code("""
            from src.artifacts import find_result_file, load_frozen_reference_artifact, sha256_file
            from src.data import load_config

            ieee_config = load_config(PROJECT_ROOT / "configs/ieee_cis.yaml")
            baf_config = load_config(PROJECT_ROOT / "configs/baf.yaml")
            ieee_artifact = load_frozen_reference_artifact("ieee_cis", expected_config=ieee_config, search_roots=INPUT_ROOTS)
            baf_artifact = load_frozen_reference_artifact("baf", expected_config=baf_config, search_roots=INPUT_ROOTS)
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
                "ieee_explanations": find_result_file("ieee_explanation_quality.csv", INPUT_ROOTS, "05_ieee_cis_rule_explanation_evaluation"),
                "ieee_ablation": find_result_file("ieee_rule_ablation.csv", INPUT_ROOTS, "06_ieee_cis_rule_ablation"),
                "baf_rules": find_result_file("baf_rule_quality.csv", INPUT_ROOTS, "07_baf_ltn_generalization"),
                "baf_explanations": find_result_file("baf_explanation_quality.csv", INPUT_ROOTS, "07_baf_ltn_generalization"),
            }
            missing = [str(path) for path in required.values() if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Required upstream outputs are missing: {missing}")
            input_manifest = pd.DataFrame([
                {"artifact": name, "path": str(path), "sha256": sha256_file(path)}
                for name, path in required.items()
            ])
            display(input_manifest)
            input_manifest.to_csv(output_dir / "input_artifact_manifest.csv", index=False)
            """),
            md("## Predictive results"),
            code("""
            predictive = pd.concat([
                pd.read_csv(required["ieee_predictive"]).assign(dataset="IEEE-CIS"),
                pd.read_csv(required["baf_predictive"]).assign(dataset="BAF"),
            ], ignore_index=True)
            predictive_test = predictive.query("split == 'test'").copy()
            display(predictive_test.round(4))
            predictive_test.to_csv(output_dir / "table_cross_dataset_predictive_test.csv", index=False)

            bootstrap = pd.concat([
                pd.read_csv(required["ieee_bootstrap"]).assign(dataset="IEEE-CIS"),
                pd.read_csv(required["baf_bootstrap"]).assign(dataset="BAF"),
            ], ignore_index=True)
            display(bootstrap.round(5))
            bootstrap.to_csv(output_dir / "table_cross_dataset_predictive_bootstrap.csv", index=False)
            """),
            md("## Logic, explanation and ablation results"),
            code("""
            rules = pd.concat([
                pd.read_csv(required["ieee_rules"]).assign(dataset="IEEE-CIS"),
                pd.read_csv(required["baf_rules"]).assign(dataset="BAF"),
            ], ignore_index=True)
            explanations = pd.concat([
                pd.read_csv(required["ieee_explanations"]).assign(dataset="IEEE-CIS"),
                pd.read_csv(required["baf_explanations"]).assign(dataset="BAF"),
            ], ignore_index=True)
            ablation = pd.read_csv(required["ieee_ablation"])
            display(rules.round(4), explanations.round(4), ablation.round(4))
            rules.to_csv(output_dir / "table_cross_dataset_rule_quality.csv", index=False)
            explanations.to_csv(output_dir / "table_cross_dataset_explanation_quality.csv", index=False)
            ablation.to_csv(output_dir / "table_ieee_rule_ablation.csv", index=False)
            """),
            code("""
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            sns.barplot(data=predictive_test, x="model", y="raw_pr_auc_mean", hue="dataset", ax=axes[0])
            axes[0].set_title("Raw test PR-AUC across datasets")
            axes[0].tick_params(axis="x", rotation=20)
            test_rules = rules.query("split == 'test'") if "split" in rules else rules
            sns.barplot(data=test_rules, x="rule", y="lift", hue="dataset", ax=axes[1])
            axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
            axes[1].set_title("Test rule lift across datasets")
            axes[1].tick_params(axis="x", rotation=35)
            plt.tight_layout()
            fig.savefig(output_dir / "figure_cross_dataset_summary.png", dpi=160, bbox_inches="tight")
            plt.show()
            """),
            md("## Takeaways"),
            code("""
            selected = predictive_test.sort_values(["dataset", "raw_pr_auc_mean"], ascending=[True, False]).groupby("dataset").head(1)
            display(selected[["dataset", "model", "raw_pr_auc_mean", "fbeta_mean", "brier_mean", "ece_mean"]].round(4))
            display(Markdown(
                "- Final claims must be based on the frozen artifacts and attached upstream tables listed above.\\n"
                "- Predictive ranking, calibration, rule evidence and explanation quality are reported separately.\\n"
                "- Cross-dataset consistency does not imply production or causal generalization."
            ))
            """),
        ],
    )


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "01_Data_Exploration.ipynb": build_exploration(),
        "02_IEEE_CIS_Model_Benchmarks.ipynb": build_benchmark("02", "ieee_cis", "IEEE-CIS"),
        "03_BAF_Model_Benchmarks.ipynb": build_benchmark("03", "baf", "BAF"),
        "04_IEEE_CIS_LTN_Rule_Analysis.ipynb": build_rule_analysis(),
        "05_IEEE_CIS_Rule_Explanation_Evaluation.ipynb": build_explanation(),
        "06_IEEE_CIS_Rule_Ablation.ipynb": build_ablation(),
        "07_BAF_LTN_Generalization.ipynb": build_baf_generalization(),
        "08_Cross_Dataset_Result_Synthesis.ipynb": build_synthesis(),
    }
    for filename, notebook_node in notebooks.items():
        for index, cell in enumerate(notebook_node.cells):
            identity = f"{filename}:{index}:{cell.cell_type}:{cell.source}".encode("utf-8")
            cell["id"] = hashlib.sha1(identity).hexdigest()[:12]
        nbf.validate(notebook_node)
        nbf.write(notebook_node, NOTEBOOK_DIR / filename)
        print(f"Wrote notebooks/{filename}")


if __name__ == "__main__":
    main()
