"""Generate the seven dataset-scoped thesis notebooks with nbformat."""

from __future__ import annotations

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
import sys

def find_project_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
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
KAGGLE = Path("/kaggle").exists()
OUTPUT_BASE = Path("/kaggle/working/thesis_outputs") if KAGGLE else PROJECT_ROOT / "results/runs/notebooks"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 120)
print({"project_root": str(PROJECT_ROOT), "quick_run": QUICK_RUN, "kaggle": KAGGLE})
'''


def notebook(title: str, cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(cells=[md(f"# {title}"), *cells])
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {"accelerator": "none", "dataSources": []},
    }
    return nb


def build_data_exploration() -> nbf.NotebookNode:
    return notebook(
        "01 — Data Exploration and Quality Audit",
        [
            code(SETUP),
            md("""
            ## Thiết lập

            - IEEE-CIS `TransactionDT` provides an ordering proxy, not a real calendar timestamp.
            - Preprocessing statistics must be fitted on train only.
            - Quick mode limits rows for runtime validation.
            """),
            code("""
            from src.data import (
                load_config,
                load_fraud_dataframe,
                make_synthetic_baf_data,
                make_synthetic_fraud_data,
                prepare_dataset,
            )

            if DATASET_NAME not in {"ieee_cis", "baf"}:
                raise ValueError(f"Unsupported dataset: {DATASET_NAME}")
            CONFIG_PATH = PROJECT_ROOT / "configs" / f"{DATASET_NAME}.yaml"
            config = load_config(CONFIG_PATH)
            output_dir = OUTPUT_BASE / f"01_data_exploration_{DATASET_NAME}"
            output_dir.mkdir(parents=True, exist_ok=True)
            max_rows = 12000 if QUICK_RUN else None
            try:
                frame = load_fraud_dataframe(config, max_rows=max_rows)
                data_source = "real"
            except FileNotFoundError:
                if not ALLOW_SYNTHETIC_FALLBACK:
                    raise
                factory = make_synthetic_fraud_data if DATASET_NAME == "ieee_cis" else make_synthetic_baf_data
                frame = factory(max_rows or 6000, seed=config["project"]["seed"])
                data_source = "synthetic"
            print({"data_source": data_source, "rows": len(frame), "columns": frame.shape[1]})
            display(frame.head())
            """),
            md("## Data"),
            code("""
            target = config["dataset"]["target_column"]
            time_column = config["dataset"]["time_column"]
            summary = pd.DataFrame({
                "rows": [len(frame)],
                "columns": [frame.shape[1]],
                "fraud_count": [int(frame[target].sum())],
                "fraud_rate": [float(frame[target].mean())],
                "duplicate_rows": [int(frame.duplicated().sum())],
                "data_source": [data_source],
            })
            display(summary)

            missing = frame.isna().mean().sort_values(ascending=False).rename("missing_fraction").to_frame()
            display(missing.head(20))
            """),
            code("""
            prepared = prepare_dataset(frame, config)
            split_summary = pd.DataFrame([
                {"split": "train", "rows": len(prepared.y_train), "fraud_rate": prepared.y_train.mean(),
                 "time_min": prepared.train_frame[time_column].min(), "time_max": prepared.train_frame[time_column].max()},
                {"split": "validation", "rows": len(prepared.y_validation), "fraud_rate": prepared.y_validation.mean(),
                 "time_min": prepared.validation_frame[time_column].min(), "time_max": prepared.validation_frame[time_column].max()},
                {"split": "test", "rows": len(prepared.y_test), "fraud_rate": prepared.y_test.mean(),
                 "time_min": prepared.test_frame[time_column].min(), "time_max": prepared.test_frame[time_column].max()},
            ])
            display(split_summary)
            print({"selected_features": len(prepared.feature_names), "matrix_shape": prepared.X_train.shape})
            """),
            md("## Results"),
            code("""
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            split_summary.plot.bar(x="split", y="fraud_rate", ax=axes[0], legend=False, color="#C44E52")
            axes[0].set_title("Fraud rate by split")
            axes[0].set_ylabel("Fraud rate")
            missing.head(20).sort_values("missing_fraction").plot.barh(ax=axes[1], legend=False, color="#4C72B0")
            axes[1].set_title("Top missing columns")
            axes[1].set_xlabel("Missing fraction")
            plt.tight_layout()
            fig.savefig(output_dir / "data_quality_overview.png", dpi=160, bbox_inches="tight")
            plt.show()

            summary.to_csv(output_dir / "dataset_summary.csv", index=False)
            split_summary.to_csv(output_dir / "split_summary.csv", index=False)
            missing.to_csv(output_dir / "missingness.csv")
            """),
            md("## Takeaways"),
            code("""
            display(Markdown(
                f"- Data source: **{data_source}** with **{len(frame):,}** rows and **{frame.shape[1]}** columns.\\n"
                f"- Overall fraud rate: **{frame[target].mean():.4%}**.\\n"
                f"- Preprocessor retained **{len(prepared.feature_names)}** train-fitted features.\\n"
                f"- Temporal order check: **{prepared.train_frame[time_column].max() <= prepared.validation_frame[time_column].min() <= prepared.test_frame[time_column].min()}**."
            ))
            """),
        ],
    )


def build_predictive_benchmarks() -> nbf.NotebookNode:
    return notebook(
        "02 — Predictive Model Benchmarks",
        [
            code(SETUP),
            md("""
            ## Thiết lập

            - PR-AUC is the primary metric.
            - F2 is the default operating-threshold objective.
            - Full mode repeats the benchmark over three independent seeds.
            - Quick mode is an explicit smoke test, not thesis evidence.
            """),
            code("""
            from src.experiment import run_repeated_predictive_benchmarks

            if DATASET_NAME not in {"ieee_cis", "baf"}:
                raise ValueError(f"Unsupported dataset: {DATASET_NAME}")
            output_dir = OUTPUT_BASE / f"02_predictive_benchmarks_{DATASET_NAME}"
            result = run_repeated_predictive_benchmarks(
                PROJECT_ROOT / "configs" / f"{DATASET_NAME}.yaml",
                output_dir=output_dir,
                model_names=("mlp", "tabular_resnet", "tree"),
                quick_run=QUICK_RUN,
                synthetic_fallback=ALLOW_SYNTHETIC_FALLBACK,
            )
            metrics = result["metrics"]
            summary = result["summary"]
            print({"data_sources": result["data_sources"], "seeds": result["seeds"], "output_dir": str(output_dir)})
            display(summary.round(4))
            """),
            md("## Data"),
            code("""
            display(result["data_summary"])
            """),
            md("## Results"),
            code("""
            test_metrics = summary.query("split == 'test'").copy()
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].bar(test_metrics["model"], test_metrics["pr_auc_mean"], yerr=test_metrics["pr_auc_std"].fillna(0), color="#4C72B0", capsize=4)
            axes[0].set_title("Test PR-AUC mean ± SD")
            axes[0].tick_params(axis="x", rotation=20)
            axes[1].bar(test_metrics["model"], test_metrics["fbeta_mean"], yerr=test_metrics["fbeta_std"].fillna(0), color="#55A868", capsize=4)
            axes[1].set_title("Test F2 mean ± SD")
            axes[1].tick_params(axis="x", rotation=20)
            plt.tight_layout()
            fig.savefig(output_dir / "predictive_model_comparison.png", dpi=160, bbox_inches="tight")
            plt.show()
            """),
            code("""
            for seed, model_histories in result["histories"].items():
                for model_name, history in model_histories.items():
                    display(Markdown(f"### {model_name} training history — seed {seed}"))
                    display(pd.DataFrame(history))
            """),
            md("## Takeaways"),
            code("""
            best = test_metrics.sort_values("pr_auc_mean", ascending=False).iloc[0]
            display(Markdown(
                f"- Data source: **{', '.join(result['data_sources'])}**.\\n"
                f"- Seeds: **{result['seeds']}**.\\n"
                f"- Highest mean test PR-AUC: **{best['model']} = {best['pr_auc_mean']:.4f} ± {best['pr_auc_std']:.4f}**.\\n"
                f"- Mean locked-threshold F2: **{best['fbeta_mean']:.4f} ± {best['fbeta_std']:.4f}**.\\n"
                "- Interpret only real-data, non-quick runs as thesis evidence."
            ))
            """),
        ],
    )


def build_rule_analysis() -> nbf.NotebookNode:
    return notebook(
        "03 — LTN-Style Fraud Rule Analysis",
        [
            code(SETUP),
            code("""
            from src.data import load_config, prepare_dataset
            from src.experiment import load_experiment_data
            from src.explanation import rule_quality_table
            from src.logic import FraudKnowledgeBase, FraudRuleEngine

            config = load_config(PROJECT_ROOT / "configs/ieee_cis.yaml")
            output_dir = OUTPUT_BASE / "03_rule_analysis"
            output_dir.mkdir(parents=True, exist_ok=True)
            frame, data_source = load_experiment_data(
                config,
                max_rows=12000 if QUICK_RUN else None,
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
            md("""
            ### Differentiable tensor-logic check

            The following diagnostic demonstrates a learnable fuzzy predicate and quantified satisfaction with PyTorch. It is trained on the training split only and is not used to select or alter the main rule-quality results below.
            """),
            code("""
            import torch
            from src.logic import SoftThresholdPredicate, TensorLogic

            amount = torch.tensor(prepared.train_frame["TransactionAmt"].to_numpy(float), dtype=torch.float32)
            amount_center = torch.nanmedian(amount)
            amount_scale = torch.nan_to_num(amount.std(), nan=1.0).clamp_min(1e-6)
            standardized_amount = torch.nan_to_num((amount - amount_center) / amount_scale)
            labels = torch.tensor(prepared.y_train, dtype=torch.float32)
            initial_threshold = float(torch.quantile(standardized_amount, 0.90))
            tensor_predicate = SoftThresholdPredicate(initial_threshold, temperature=0.5, learnable=True)
            optimizer = torch.optim.Adam(tensor_predicate.parameters(), lr=0.03)
            initial_parameter = tensor_predicate.threshold.detach().clone()
            history = []
            for step in range(30 if QUICK_RUN else 100):
                optimizer.zero_grad()
                evidence = tensor_predicate(standardized_amount)
                positive_satisfaction = TensorLogic.forall(evidence[labels == 1])
                negative_satisfaction = TensorLogic.forall(1.0 - evidence[labels == 0])
                satisfaction = 0.5 * (positive_satisfaction + negative_satisfaction)
                regularization = 0.01 * (tensor_predicate.threshold - initial_parameter).pow(2)
                loss = 1.0 - satisfaction + regularization
                loss.backward()
                optimizer.step()
                history.append(float(satisfaction.detach()))

            tensor_demo = pd.DataFrame({
                "initial_threshold_standardized": [float(initial_parameter)],
                "learned_threshold_standardized": [float(tensor_predicate.threshold.detach())],
                "initial_satisfaction": [history[0]],
                "final_satisfaction": [history[-1]],
            })
            display(tensor_demo.round(4))
            """),
            md("## Data"),
            code("""
            validation_truth = engine.evaluate(prepared.validation_frame)
            test_truth = engine.evaluate(prepared.test_frame)
            activation_threshold = float(config["logic"]["activation_threshold"])
            validation_quality = rule_quality_table(validation_truth, prepared.y_validation, activation_threshold).assign(split="validation")
            test_quality = rule_quality_table(test_truth, prepared.y_test, activation_threshold).assign(split="test")
            quality = pd.concat([validation_quality, test_quality], ignore_index=True)
            display(quality.round(4))
            """),
            md("## Results"),
            code("""
            stability = validation_quality.merge(test_quality, on="rule", suffixes=("_validation", "_test"))
            stability["coverage_delta"] = stability["coverage_test"] - stability["coverage_validation"]
            stability["lift_delta"] = stability["lift_test"] - stability["lift_validation"]
            display(stability[["rule", "coverage_validation", "coverage_test", "coverage_delta", "lift_validation", "lift_test", "lift_delta"]].round(4))

            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            sns.barplot(data=quality, x="rule", y="coverage", hue="split", ax=axes[0])
            axes[0].set_title("Rule coverage")
            axes[0].tick_params(axis="x", rotation=35)
            sns.barplot(data=quality, x="rule", y="lift", hue="split", ax=axes[1])
            axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
            axes[1].set_title("Fraud lift")
            axes[1].tick_params(axis="x", rotation=35)
            plt.tight_layout()
            fig.savefig(output_dir / "rule_quality.png", dpi=160, bbox_inches="tight")
            plt.show()

            quality.to_csv(output_dir / "rule_quality.csv", index=False)
            stability.to_csv(output_dir / "rule_stability.csv", index=False)
            engine.fitted_thresholds().to_csv(output_dir / "fitted_rule_thresholds.csv", index=False)
            """),
            md("## Takeaways"),
            code("""
            strongest = test_quality.sort_values("lift", ascending=False).iloc[0]
            display(Markdown(
                f"- Data source: **{data_source}**.\\n"
                f"- Active rules: **{len(engine.rules)}**; skipped rules: **{len(engine.skipped_rules)}**.\\n"
                f"- Highest test lift: **{strongest['rule']} = {strongest['lift']:.3f}** at coverage **{strongest['coverage']:.3f}**.\\n"
                "- Lift indicates association with fraud labels, not causality."
            ))
            """),
        ],
    )


def build_explanation_evaluation() -> nbf.NotebookNode:
    return notebook(
        "04 — Rule Explanation Evaluation",
        [
            code(SETUP),
            code("""
            from src.experiment import run_predictive_benchmarks
            from src.explanation import RuleExplainer, bootstrap_explanation_precision_gain, explanation_quality_metrics, rule_quality_table
            from src.logic import FraudRuleEngine

            output_dir = OUTPUT_BASE / "04_explanation_evaluation"
            result = run_predictive_benchmarks(
                PROJECT_ROOT / "configs/ieee_cis.yaml",
                output_dir=output_dir,
                model_names=("tree",),
                quick_run=QUICK_RUN,
                synthetic_fallback=ALLOW_SYNTHETIC_FALLBACK,
            )
            config = result["config"]
            prepared = result["prepared"]
            target = config["dataset"]["target_column"]
            engine = FraudRuleEngine(config["logic"]["rules"]).fit(prepared.train_frame, target)
            probabilities = result["test_probabilities"]["tree"]
            threshold = result["thresholds"]["tree"]
            explainer = RuleExplainer(
                engine,
                config["logic"]["activation_threshold"],
                config["logic"]["top_k_rules"],
            )
            explanations = explainer.explain(prepared.test_frame, probabilities, threshold)
            explanations["y_true"] = prepared.y_test
            print({"data_source": result["data_source"], "threshold": threshold})
            display(explanations.head())
            """),
            md("## Data"),
            code("""
            truth = engine.evaluate(prepared.test_frame)
            rule_quality = rule_quality_table(truth, prepared.y_test, config["logic"]["activation_threshold"])
            display(rule_quality.round(4))
            """),
            md("## Results"),
            code("""
            quality = explanation_quality_metrics(explanations, prepared.y_test, probabilities, threshold)
            quality.update(bootstrap_explanation_precision_gain(
                explanations,
                prepared.y_test,
                probabilities,
                threshold,
                n_bootstrap=config["evaluation"]["bootstrap_iterations"],
                seed=config["project"]["seed"],
            ))
            quality_frame = pd.DataFrame([quality])
            display(quality_frame.round(4))

            predicted_alert = probabilities >= threshold
            case_frame = explanations.copy()
            case_frame["case_type"] = np.select(
                [predicted_alert & (prepared.y_test == 1), predicted_alert & (prepared.y_test == 0), (~predicted_alert) & (prepared.y_test == 1)],
                ["true_positive", "false_positive", "false_negative"],
                default="true_negative",
            )
            selected_cases = pd.concat([
                group.sort_values("predicted_probability", ascending=False).head(3)
                for _, group in case_frame.groupby("case_type")
            ]).sort_values(["case_type", "predicted_probability"], ascending=[True, False])
            display(selected_cases[["case_type", "y_true", "predicted_probability", "rule_names", "rule_strengths", "explanation"]])

            quality_frame.to_csv(output_dir / "explanation_quality.csv", index=False)
            rule_quality.to_csv(output_dir / "test_rule_quality.csv", index=False)
            selected_cases.to_json(output_dir / "explanation_cases.json", orient="records", indent=2)
            """),
            code("""
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(explanations["max_rule_strength"], bins=30, ax=axes[0], color="#4C72B0")
            axes[0].axvline(config["logic"]["activation_threshold"], color="black", linestyle="--")
            axes[0].set_title("Maximum rule strength")
            sns.countplot(data=case_frame, x="case_type", hue="explained", ax=axes[1])
            axes[1].set_title("Explained status by outcome")
            axes[1].tick_params(axis="x", rotation=25)
            plt.tight_layout()
            fig.savefig(output_dir / "explanation_overview.png", dpi=160, bbox_inches="tight")
            plt.show()
            """),
            md("## Takeaways"),
            code("""
            display(Markdown(
                f"- Test alert explanation coverage: **{quality['explanation_coverage_alerts']:.3f}**.\\n"
                f"- Explained-alert precision gain: **{quality['explained_alert_precision_gain']:.3f}** (95% bootstrap CI **[{quality['precision_gain_ci_low']:.3f}, {quality['precision_gain_ci_high']:.3f}]**).\\n"
                f"- Mean rule count: **{quality['mean_rule_count']:.2f}**.\\n"
                f"- Prediction-rule consistency: **{quality['prediction_rule_consistency']:.3f}**.\\n"
                f"- Contradiction rate: **{quality['contradiction_rate']:.3f}**.\\n"
                "- These values quantify rule evidence behavior; they do not establish causal explanations."
            ))
            """),
        ],
    )


def build_ablation() -> nbf.NotebookNode:
    return notebook(
        "05 — Logical Rule Ablation",
        [
            code(SETUP),
            code("""
            from src.experiment import run_predictive_benchmarks
            from src.logic import FraudRuleEngine

            output_dir = OUTPUT_BASE / "05_rule_ablation"
            result = run_predictive_benchmarks(
                PROJECT_ROOT / "configs/ieee_cis.yaml",
                output_dir=output_dir,
                model_names=("tree",),
                quick_run=QUICK_RUN,
                synthetic_fallback=ALLOW_SYNTHETIC_FALLBACK,
            )
            config = result["config"]
            prepared = result["prepared"]
            target = config["dataset"]["target_column"]
            engine = FraudRuleEngine(config["logic"]["rules"]).fit(prepared.train_frame, target)
            truth = engine.evaluate(prepared.test_frame)
            probabilities = result["test_probabilities"]["tree"]
            threshold = result["thresholds"]["tree"]
            predicted_alert = probabilities >= threshold
            activation = float(config["logic"]["activation_threshold"])
            print({"data_source": result["data_source"], "rules": truth.columns.tolist()})
            """),
            md("## Data"),
            code("""
            display(truth.describe().T[["mean", "std", "min", "max"]].round(4))
            """),
            md("## Results"),
            code("""
            def score_rule_subset(name, columns):
                if columns:
                    rule_score = truth[columns].max(axis=1).to_numpy(float)
                else:
                    rule_score = np.zeros(len(truth), dtype=float)
                explained = rule_score >= activation
                explained_alert = explained & predicted_alert
                precision = prepared.y_test[explained_alert].mean() if explained_alert.any() else 0.0
                base_precision = prepared.y_test[predicted_alert].mean() if predicted_alert.any() else 0.0
                return {
                    "ablation": name,
                    "rule_count": len(columns),
                    "coverage_all": explained.mean(),
                    "coverage_alerts": explained_alert.sum() / max(predicted_alert.sum(), 1),
                    "explained_alert_precision": precision,
                    "precision_gain": precision - base_precision,
                    "prediction_rule_consistency": (predicted_alert == explained).mean(),
                }

            all_rules = truth.columns.tolist()
            rows = [score_rule_subset("full_rule_set", all_rules), score_rule_subset("no_rules", [])]
            rows.extend(score_rule_subset(f"only:{rule}", [rule]) for rule in all_rules)
            rows.extend(score_rule_subset(f"without:{rule}", [item for item in all_rules if item != rule]) for rule in all_rules)
            ablation = pd.DataFrame(rows).sort_values(["coverage_alerts", "precision_gain"], ascending=False)
            display(ablation.round(4))

            ablation.to_csv(output_dir / "rule_ablation.csv", index=False)
            """),
            code("""
            plot_data = ablation[ablation["ablation"].str.startswith("without:")].copy()
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            sns.barplot(data=plot_data, x="ablation", y="coverage_alerts", ax=axes[0], color="#4C72B0")
            axes[0].tick_params(axis="x", rotation=35)
            axes[0].set_title("Alert coverage after removing one rule")
            sns.barplot(data=plot_data, x="ablation", y="precision_gain", ax=axes[1], color="#55A868")
            axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
            axes[1].tick_params(axis="x", rotation=35)
            axes[1].set_title("Precision gain after removing one rule")
            plt.tight_layout()
            fig.savefig(output_dir / "rule_ablation.png", dpi=160, bbox_inches="tight")
            plt.show()
            """),
            md("## Takeaways"),
            code("""
            full = ablation.query("ablation == 'full_rule_set'").iloc[0]
            best = ablation[ablation["ablation"].str.startswith("without:")].sort_values("precision_gain", ascending=False).iloc[0]
            display(Markdown(
                f"- Full-set alert coverage: **{full['coverage_alerts']:.3f}**.\\n"
                f"- Full-set precision gain: **{full['precision_gain']:.3f}**.\\n"
                f"- Highest leave-one-out precision gain: **{best['ablation']} = {best['precision_gain']:.3f}**.\\n"
                "- Interpret ablation only after confirming real data and locked predictor probabilities."
            ))
            """),
        ],
    )


def build_baf() -> nbf.NotebookNode:
    return notebook(
        "06 — BAF External Generalization",
        [
            code(SETUP),
            code("""
            from src.experiment import run_predictive_benchmarks
            from src.explanation import RuleExplainer, bootstrap_explanation_precision_gain, explanation_quality_metrics, rule_quality_table
            from src.logic import FraudRuleEngine

            output_dir = OUTPUT_BASE / "06_baf_generalization"
            result = run_predictive_benchmarks(
                PROJECT_ROOT / "configs/baf.yaml",
                output_dir=output_dir,
                model_names=("mlp", "tabular_resnet", "tree"),
                quick_run=QUICK_RUN,
                synthetic_fallback=ALLOW_SYNTHETIC_FALLBACK,
            )
            config = result["config"]
            prepared = result["prepared"]
            display(result["metrics"].round(4))
            print({"data_source": result["data_source"], "rows": len(result["frame"])})
            """),
            md("## Data"),
            code("""
            split_summary = pd.DataFrame({
                "split": ["train", "validation", "test"],
                "rows": [len(prepared.y_train), len(prepared.y_validation), len(prepared.y_test)],
                "fraud_rate": [prepared.y_train.mean(), prepared.y_validation.mean(), prepared.y_test.mean()],
            })
            display(split_summary)
            """),
            md("## Results"),
            code("""
            target = config["dataset"]["target_column"]
            engine = FraudRuleEngine(config["logic"]["rules"]).fit(prepared.train_frame, target)
            test_truth = engine.evaluate(prepared.test_frame)
            rule_quality = rule_quality_table(test_truth, prepared.y_test, config["logic"]["activation_threshold"])
            display(rule_quality.round(4))

            model_key = "tree"
            probabilities = result["test_probabilities"][model_key]
            threshold = result["thresholds"][model_key]
            explainer = RuleExplainer(engine, config["logic"]["activation_threshold"], config["logic"]["top_k_rules"])
            explanations = explainer.explain(prepared.test_frame, probabilities, threshold)
            quality = explanation_quality_metrics(explanations, prepared.y_test, probabilities, threshold)
            quality.update(bootstrap_explanation_precision_gain(
                explanations,
                prepared.y_test,
                probabilities,
                threshold,
                n_bootstrap=config["evaluation"]["bootstrap_iterations"],
                seed=config["project"]["seed"],
            ))
            display(pd.DataFrame([quality]).round(4))

            rule_quality.to_csv(output_dir / "baf_rule_quality.csv", index=False)
            pd.DataFrame([quality]).to_csv(output_dir / "baf_explanation_quality.csv", index=False)
            """),
            code("""
            test_metrics = result["metrics"].query("split == 'test'")
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            sns.barplot(data=test_metrics, x="model", y="pr_auc", ax=axes[0], color="#4C72B0")
            axes[0].set_title("BAF test PR-AUC")
            axes[0].tick_params(axis="x", rotation=20)
            sns.barplot(data=rule_quality, x="rule", y="lift", ax=axes[1], color="#C44E52")
            axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
            axes[1].set_title("BAF rule lift")
            axes[1].tick_params(axis="x", rotation=35)
            plt.tight_layout()
            fig.savefig(output_dir / "baf_generalization.png", dpi=160, bbox_inches="tight")
            plt.show()
            """),
            md("## Takeaways"),
            code("""
            best_model = test_metrics.sort_values("pr_auc", ascending=False).iloc[0]
            best_rule = rule_quality.sort_values("lift", ascending=False).iloc[0]
            display(Markdown(
                f"- Data source: **{result['data_source']}**.\\n"
                f"- Highest BAF test PR-AUC: **{best_model['model']} = {best_model['pr_auc']:.4f}**.\\n"
                f"- Highest BAF rule lift: **{best_rule['rule']} = {best_rule['lift']:.3f}**.\\n"
                f"- Tree-alert explanation coverage: **{quality['explanation_coverage_alerts']:.3f}**.\\n"
                "- Cross-dataset comparison is valid only for real-data full runs."
            ))
            """),
        ],
    )


def _exploration_section(
    roman_number: str,
    dataset_label: str,
    config_filename: str,
    synthetic_factory: str,
    output_slug: str,
) -> list:
    load_source = """
    config = load_config(PROJECT_ROOT / "configs/__CONFIG__")
    try:
        frame = load_fraud_dataframe(config, max_rows=max_rows)
        data_source = "real"
    except FileNotFoundError:
        if not ALLOW_SYNTHETIC_FALLBACK:
            raise
        frame = __FACTORY__(max_rows or 6000, seed=config["project"]["seed"])
        data_source = "synthetic"
    print({"dataset": "__LABEL__", "data_source": data_source, "rows": len(frame), "columns": frame.shape[1]})
    display(frame.head())
    """.replace("__CONFIG__", config_filename).replace("__FACTORY__", synthetic_factory).replace("__LABEL__", dataset_label)

    summary_source = """
    target = config["dataset"]["target_column"]
    time_column = config["dataset"]["time_column"]
    summary = pd.DataFrame({
        "dataset": ["__LABEL__"],
        "rows": [len(frame)],
        "columns": [frame.shape[1]],
        "fraud_count": [int(frame[target].sum())],
        "fraud_rate": [float(frame[target].mean())],
        "duplicate_rows": [int(frame.duplicated().sum())],
        "data_source": [data_source],
    })
    missing = frame.isna().mean().sort_values(ascending=False).rename("missing_fraction").to_frame()
    prepared = prepare_dataset(frame, config)
    split_summary = pd.DataFrame([
        {"split": "train", "rows": len(prepared.y_train), "fraud_rate": prepared.y_train.mean(),
         "time_min": prepared.train_frame[time_column].min(), "time_max": prepared.train_frame[time_column].max()},
        {"split": "validation", "rows": len(prepared.y_validation), "fraud_rate": prepared.y_validation.mean(),
         "time_min": prepared.validation_frame[time_column].min(), "time_max": prepared.validation_frame[time_column].max()},
        {"split": "test", "rows": len(prepared.y_test), "fraud_rate": prepared.y_test.mean(),
         "time_min": prepared.test_frame[time_column].min(), "time_max": prepared.test_frame[time_column].max()},
    ])
    dataset_summaries.append(summary.copy())
    display(summary, missing.head(20), split_summary)
    """.replace("__LABEL__", dataset_label)

    plot_source = """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    split_summary.plot.bar(x="split", y="fraud_rate", ax=axes[0], legend=False, color="#C44E52")
    axes[0].set_title("__LABEL__ fraud rate by split")
    axes[0].set_ylabel("Fraud rate")
    missing.head(20).sort_values("missing_fraction").plot.barh(ax=axes[1], legend=False, color="#4C72B0")
    axes[1].set_title("__LABEL__ top missing columns")
    axes[1].set_xlabel("Missing fraction")
    plt.tight_layout()
    fig.savefig(output_dir / "__SLUG___data_quality.png", dpi=160, bbox_inches="tight")
    plt.show()

    split_summary.to_csv(output_dir / "__SLUG___split_summary.csv", index=False)
    missing.to_csv(output_dir / "__SLUG___missingness.csv")
    """.replace("__LABEL__", dataset_label).replace("__SLUG__", output_slug)

    return [
        md(f"## {roman_number}. {dataset_label}"),
        code(load_source),
        code(summary_source),
        code(plot_source),
    ]


def build_multi_dataset_exploration() -> nbf.NotebookNode:
    cells = [
        code(SETUP),
        code("""
        from src.data import (
            load_config,
            load_fraud_dataframe,
            make_synthetic_baf_data,
            make_synthetic_fraud_data,
            prepare_dataset,
        )

        max_rows = 12000 if QUICK_RUN else None
        output_dir = OUTPUT_BASE / "01_data_exploration"
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_summaries = []
        """),
    ]
    cells.extend(_exploration_section("I", "IEEE-CIS", "ieee_cis.yaml", "make_synthetic_fraud_data", "ieee_cis"))
    cells.extend(_exploration_section("II", "BAF", "baf.yaml", "make_synthetic_baf_data", "baf"))
    cells.extend([
        md("## III. So sánh tổng quan"),
        code("""
        combined_summary = pd.concat(dataset_summaries, ignore_index=True)
        display(combined_summary)
        ax = combined_summary.plot.bar(x="dataset", y="fraud_rate", legend=False, color=["#4C72B0", "#55A868"])
        ax.set_title("Fraud rate across datasets")
        ax.set_ylabel("Fraud rate")
        plt.tight_layout()
        plt.savefig(output_dir / "dataset_fraud_rate_comparison.png", dpi=160, bbox_inches="tight")
        plt.show()
        combined_summary.to_csv(output_dir / "dataset_summary.csv", index=False)
        """),
    ])
    return notebook("01 — Multi-Dataset Exploration and Quality Audit", cells)


def build_dataset_benchmark(dataset_name: str, notebook_number: str, dataset_label: str) -> nbf.NotebookNode:
    benchmark = build_predictive_benchmarks()
    benchmark.cells[0].source = f"# {notebook_number} — {dataset_label} Predictive Model Benchmarks"
    for cell in benchmark.cells:
        if cell.cell_type != "code":
            continue
        cell.source = cell.source.replace(
            'if DATASET_NAME not in {"ieee_cis", "baf"}:\n    raise ValueError(f"Unsupported dataset: {DATASET_NAME}")\n',
            "",
        )
        cell.source = cell.source.replace(
            'OUTPUT_BASE / f"02_predictive_benchmarks_{DATASET_NAME}"',
            f'OUTPUT_BASE / "{notebook_number}_{dataset_name}_model_benchmarks"',
        )
        cell.source = cell.source.replace(
            'PROJECT_ROOT / "configs" / f"{DATASET_NAME}.yaml"',
            f'PROJECT_ROOT / "configs/{dataset_name}.yaml"',
        )
    return benchmark


def _retitle(notebook_node: nbf.NotebookNode, title: str) -> nbf.NotebookNode:
    notebook_node.cells[0].source = f"# {title}"
    return notebook_node


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "01_Data_Exploration.ipynb": build_multi_dataset_exploration(),
        "02_IEEE_CIS_Model_Benchmarks.ipynb": build_dataset_benchmark("ieee_cis", "02", "IEEE-CIS"),
        "03_BAF_Model_Benchmarks.ipynb": build_dataset_benchmark("baf", "03", "BAF"),
        "04_IEEE_CIS_LTN_Rule_Analysis.ipynb": _retitle(build_rule_analysis(), "04 — IEEE-CIS LTN-Style Fraud Rule Analysis"),
        "05_IEEE_CIS_Rule_Explanation_Evaluation.ipynb": _retitle(build_explanation_evaluation(), "05 — IEEE-CIS Rule Explanation Evaluation"),
        "06_IEEE_CIS_Rule_Ablation.ipynb": _retitle(build_ablation(), "06 — IEEE-CIS Logical Rule Ablation"),
        "07_BAF_LTN_Generalization.ipynb": _retitle(build_baf(), "07 — BAF LTN and Explanation Generalization"),
    }
    obsolete = {
        "02_Predictive_Model_Benchmarks.ipynb",
        "03_LTN_Rule_Analysis.ipynb",
        "04_Rule_Explanation_Evaluation.ipynb",
        "05_Rule_Ablation.ipynb",
        "06_BAF_Generalization.ipynb",
    }
    for filename in obsolete:
        (NOTEBOOK_DIR / filename).unlink(missing_ok=True)
    for filename, notebook_node in notebooks.items():
        nbf.validate(notebook_node)
        nbf.write(notebook_node, NOTEBOOK_DIR / filename)
        print(f"Wrote notebooks/{filename}")


if __name__ == "__main__":
    main()
