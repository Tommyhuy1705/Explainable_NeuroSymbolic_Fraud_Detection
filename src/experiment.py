"""Reusable experiment orchestration shared by scripts and Kaggle notebooks."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.artifacts import (
    capture_environment,
    sha256_file,
    stable_config_hash,
    write_frozen_reference_artifact,
)
from src.data import (
    load_config,
    load_fraud_dataframe,
    make_synthetic_baf_data,
    make_synthetic_fraud_data,
    prepare_dataset,
    split_integrity_summary,
)
from src.evaluation import (
    evaluate_binary_predictions,
    paired_bootstrap_pr_auc_difference,
    select_probability_calibrator,
    select_threshold,
)
from src.models import FraudMLP, TabularResNet, build_tree_classifier, tree_backend_name
from src.training import predict_torch_proba, set_global_seed, train_torch_model


def load_experiment_data(
    config: dict[str, Any],
    data_root: str | Path | None = None,
    max_rows: int | None = None,
    synthetic_fallback: bool = False,
    synthetic_rows: int = 6000,
) -> tuple[pd.DataFrame, str]:
    """Load configured data, optionally falling back to deterministic smoke data."""
    try:
        return load_fraud_dataframe(config, data_root, max_rows), "real"
    except FileNotFoundError:
        if not synthetic_fallback:
            raise
        seed = config["project"].get("seed", 42)
        if config["dataset"]["name"] == "ieee_cis":
            frame = make_synthetic_fraud_data(synthetic_rows, seed)
        elif config["dataset"]["name"] == "baf":
            frame = make_synthetic_baf_data(synthetic_rows, seed)
        else:
            raise
        return frame, "synthetic"


def _quick_model_config(config: dict[str, Any], model_name: str, quick_run: bool) -> dict[str, Any]:
    model_config = dict(config["models"][model_name])
    if quick_run and model_name in {"mlp", "tabular_resnet"}:
        model_config["epochs"] = min(int(model_config.get("epochs", 20)), 3)
        model_config["patience"] = min(int(model_config.get("patience", 4)), 2)
        model_config["batch_size"] = min(int(model_config.get("batch_size", 2048)), 1024)
    if quick_run and model_name == "tree":
        model_config["n_estimators"] = min(int(model_config.get("n_estimators", 500)), 80)
        model_config["max_depth"] = min(max(int(model_config.get("max_depth", 6)), 2), 6)
    return model_config


def _fit_tree(model, X_train: np.ndarray, y_train: np.ndarray) -> None:
    positives = max(int((y_train == 1).sum()), 1)
    negatives = max(int((y_train == 0).sum()), 1)
    parameters = model.get_params(deep=False)
    if "scale_pos_weight" in parameters:
        model.set_params(scale_pos_weight=negatives / positives)
    elif "class_weight" in parameters:
        model.set_params(class_weight="balanced")
    model.fit(X_train, y_train)


def _evaluate_probabilities(
    config: dict[str, Any],
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    y_test: np.ndarray,
    test_probabilities: np.ndarray,
) -> tuple[
    dict[str, float],
    dict[str, float],
    float,
    np.ndarray,
    np.ndarray,
    str,
    pd.DataFrame,
]:
    evaluation = config["evaluation"]
    selection = select_probability_calibrator(
        y_validation,
        validation_probabilities,
        methods=evaluation.get("calibration_methods", [evaluation.get("calibration", "none")]),
        selection_metric=str(evaluation.get("calibration_selection_metric", "brier")),
        fit_fraction=float(evaluation.get("calibration_fit_fraction", 0.5)),
        n_calibration_bins=int(evaluation.get("n_calibration_bins", 15)),
    )
    calibrated_validation = selection.calibrator.transform(validation_probabilities)
    calibrated_test = selection.calibrator.transform(test_probabilities)
    threshold, _ = select_threshold(
        y_validation[selection.selection_indices],
        calibrated_validation[selection.selection_indices],
        objective=str(evaluation.get("threshold_objective", "f2")),
        beta=float(evaluation.get("beta", 2.0)),
    )
    common = {
        "threshold": threshold,
        "beta": float(evaluation.get("beta", 2.0)),
        "n_calibration_bins": int(evaluation.get("n_calibration_bins", 15)),
    }
    validation_metrics = evaluate_binary_predictions(y_validation, calibrated_validation, **common)
    test_metrics = evaluate_binary_predictions(y_test, calibrated_test, **common)
    validation_metrics.update(
        {
            "raw_pr_auc": float(average_precision_score(y_validation, validation_probabilities)),
            "raw_roc_auc": float(roc_auc_score(y_validation, validation_probabilities)),
        }
    )
    test_metrics.update(
        {
            "raw_pr_auc": float(average_precision_score(y_test, test_probabilities)),
            "raw_roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        }
    )
    return (
        validation_metrics,
        test_metrics,
        threshold,
        calibrated_validation,
        calibrated_test,
        selection.method,
        pd.DataFrame(selection.comparison),
    )


def run_predictive_benchmarks(
    config_path: str | Path,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_names: Iterable[str] = ("mlp", "tabular_resnet", "tree"),
    quick_run: bool = False,
    max_rows: int | None = None,
    synthetic_fallback: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run selected predictive models under one locked evaluation protocol."""
    config = load_config(config_path)
    model_names = tuple(model_names)
    if quick_run and max_rows is None:
        max_rows = 12000
    frame, data_source = load_experiment_data(
        config,
        data_root,
        max_rows,
        synthetic_fallback=synthetic_fallback,
        synthetic_rows=max_rows or 6000,
    )
    prepared = prepare_dataset(frame, config)
    seed = int(config["project"].get("seed", 42) if seed is None else seed)
    rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    raw_validation_probabilities: dict[str, np.ndarray] = {}
    raw_test_probabilities: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    calibration_methods: dict[str, str] = {}
    calibration_comparisons: list[pd.DataFrame] = []
    display_names: dict[str, str] = {}
    histories: dict[str, list[dict[str, float]]] = {}

    for model_name in model_names:
        # Seed before model construction so parameter initialization is reproducible.
        set_global_seed(seed)
        if model_name == "mlp":
            model_config = _quick_model_config(config, "mlp", quick_run)
            model = FraudMLP(
                prepared.X_train.shape[1],
                hidden_dims=model_config.get("hidden_dims", [256, 128]),
                dropout=float(model_config.get("dropout", 0.2)),
            )
            result = train_torch_model(
                model,
                prepared.X_train,
                prepared.y_train,
                prepared.X_validation,
                prepared.y_validation,
                model_config,
                seed,
            )
            model = result.model
            val_probability = predict_torch_proba(model, prepared.X_validation, device=result.device)
            test_probability = predict_torch_proba(model, prepared.X_test, device=result.device)
            histories[model_name] = result.history
            display_name = "MLP"
        elif model_name == "tabular_resnet":
            model_config = _quick_model_config(config, "tabular_resnet", quick_run)
            model = TabularResNet(
                prepared.X_train.shape[1],
                hidden_dim=int(model_config.get("hidden_dim", 192)),
                num_blocks=int(model_config.get("num_blocks", 3)),
                dropout=float(model_config.get("dropout", 0.15)),
            )
            result = train_torch_model(
                model,
                prepared.X_train,
                prepared.y_train,
                prepared.X_validation,
                prepared.y_validation,
                model_config,
                seed,
            )
            model = result.model
            val_probability = predict_torch_proba(model, prepared.X_validation, device=result.device)
            test_probability = predict_torch_proba(model, prepared.X_test, device=result.device)
            histories[model_name] = result.history
            display_name = "TabularResNet"
        elif model_name == "tree":
            model_config = _quick_model_config(config, "tree", quick_run)
            model = build_tree_classifier(model_config, seed)
            _fit_tree(model, prepared.X_train, prepared.y_train)
            val_probability = model.predict_proba(prepared.X_validation)[:, 1]
            test_probability = model.predict_proba(prepared.X_test)[:, 1]
            display_name = tree_backend_name(str(model_config.get("backend", "xgboost")))
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        raw_val_probability = np.asarray(val_probability, dtype=float).copy()
        raw_test_probability = np.asarray(test_probability, dtype=float).copy()
        (
            val_metrics,
            test_metrics,
            threshold,
            val_probability,
            test_probability,
            calibration_method,
            calibration_comparison,
        ) = _evaluate_probabilities(
            config,
            prepared.y_validation,
            raw_val_probability,
            prepared.y_test,
            raw_test_probability,
        )
        calibration_comparison.insert(0, "model", display_name)
        calibration_comparison.insert(1, "model_key", model_name)
        calibration_comparison["selected"] = calibration_comparison["method"].eq(calibration_method)
        calibration_comparisons.append(calibration_comparison)
        for split_name, metrics in (("validation", val_metrics), ("test", test_metrics)):
            rows.append({"model": display_name, "model_key": model_name, "split": split_name, **metrics})
        models[model_name] = model
        validation_probabilities[model_name] = val_probability
        test_probabilities[model_name] = test_probability
        raw_validation_probabilities[model_name] = raw_val_probability
        raw_test_probabilities[model_name] = raw_test_probability
        thresholds[model_name] = threshold
        calibration_methods[model_name] = calibration_method
        display_names[model_name] = display_name

    metrics = pd.DataFrame(rows)
    calibration_comparison_frame = pd.concat(calibration_comparisons, ignore_index=True)
    destination = Path(output_dir or config["project"]["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(destination / "predictive_metrics.csv", index=False)
    calibration_comparison_frame.to_csv(destination / "calibration_comparison.csv", index=False)
    np.savez_compressed(
        destination / "predictions.npz",
        y_validation=prepared.y_validation,
        y_test=prepared.y_test,
        **{f"validation_{name}": values for name, values in validation_probabilities.items()},
        **{f"test_{name}": values for name, values in test_probabilities.items()},
        **{f"validation_raw_{name}": values for name, values in raw_validation_probabilities.items()},
        **{f"test_raw_{name}": values for name, values in raw_test_probabilities.items()},
    )
    time_column = str(config["dataset"]["time_column"])
    split_summary = split_integrity_summary(
        prepared.train_frame, prepared.validation_frame, prepared.test_frame, time_column
    )
    split_summary["fraud_rate"] = [
        float(prepared.y_train.mean()),
        float(prepared.y_validation.mean()),
        float(prepared.y_test.mean()),
    ]
    predictions_path = destination / "predictions.npz"
    metadata = {
        "data_source": data_source,
        "dataset_name": config["dataset"]["name"],
        "configured_files": frame.attrs.get("configured_files", []),
        "rows": len(frame),
        "features": prepared.X_train.shape[1],
        "feature_names": prepared.feature_names,
        "thresholds": thresholds,
        "calibration_methods": calibration_methods,
        "display_names": display_names,
        "models": list(model_names),
        "seed": seed,
        "quick_run": quick_run,
        "config_sha256": stable_config_hash(config),
        "predictions_sha256": sha256_file(predictions_path),
        "split_summary": split_summary.to_dict(orient="records"),
        "environment": capture_environment(Path(config_path).resolve().parents[1]),
    }
    (destination / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    (destination / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    split_summary.to_csv(destination / "split_summary.csv", index=False)
    return {
        "config": config,
        "frame": frame,
        "prepared": prepared,
        "metrics": metrics,
        "models": models,
        "validation_probabilities": validation_probabilities,
        "test_probabilities": test_probabilities,
        "raw_validation_probabilities": raw_validation_probabilities,
        "raw_test_probabilities": raw_test_probabilities,
        "thresholds": thresholds,
        "calibration_methods": calibration_methods,
        "calibration_comparison": calibration_comparison_frame,
        "split_summary": split_summary,
        "histories": histories,
        "output_dir": destination,
        "data_source": data_source,
    }


def run_repeated_predictive_benchmarks(
    config_path: str | Path,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_names: Iterable[str] = ("mlp", "tabular_resnet", "tree"),
    quick_run: bool = False,
    max_rows: int | None = None,
    synthetic_fallback: bool = False,
    seeds: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Repeat the locked benchmark and report mean/std across independent seeds."""
    config = load_config(config_path)
    model_names = tuple(model_names)
    configured_seeds = list(seeds or config["evaluation"].get("seed_list", [42, 123, 2026]))
    if not configured_seeds:
        raise ValueError("At least one experiment seed is required")
    effective_seeds = configured_seeds[:1] if quick_run else configured_seeds
    destination = Path(output_dir or config["project"]["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)

    metric_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    data_summary: pd.DataFrame | None = None
    histories: dict[int, dict[str, list[dict[str, float]]]] = {}
    data_sources: set[str] = set()

    for run_seed in effective_seeds:
        run = run_predictive_benchmarks(
            config_path,
            data_root=data_root,
            output_dir=destination / f"seed_{run_seed}",
            model_names=model_names,
            quick_run=quick_run,
            max_rows=max_rows,
            synthetic_fallback=synthetic_fallback,
            seed=int(run_seed),
        )
        run_metrics = run["metrics"].copy()
        run_metrics.insert(0, "seed", int(run_seed))
        metric_frames.append(run_metrics)
        run_calibration = run["calibration_comparison"].copy()
        run_calibration.insert(0, "seed", int(run_seed))
        calibration_frames.append(run_calibration)
        data_sources.add(str(run["data_source"]))
        histories[int(run_seed)] = run["histories"]
        if data_summary is None:
            data_summary = run["split_summary"].copy()
        del run
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    metrics = pd.concat(metric_frames, ignore_index=True)
    metric_columns = [
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "fbeta",
        "brier",
        "ece",
        "nll",
        "threshold",
        "positive_rate",
        "raw_pr_auc",
        "raw_roc_auc",
    ]
    summary = metrics.groupby(["model", "model_key", "split"], as_index=False)[metric_columns].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary = summary.reset_index(drop=True)
    summary["n_seeds"] = len(effective_seeds)
    calibration_comparison = pd.concat(calibration_frames, ignore_index=True)
    metrics.to_csv(destination / "predictive_metrics_all_seeds.csv", index=False)
    summary.to_csv(destination / "predictive_metrics_summary.csv", index=False)
    calibration_comparison.to_csv(destination / "calibration_comparison_all_seeds.csv", index=False)

    validation_summary = summary.query("split == 'validation'").sort_values(
        ["raw_pr_auc_mean", "raw_pr_auc_std"], ascending=[False, True]
    )
    best_row = validation_summary.iloc[0]
    reference_model_key = str(best_row["model_key"])
    reference_seed = int(config["evaluation"].get("reference_seed", effective_seeds[0]))
    if reference_seed not in effective_seeds:
        reference_seed = int(effective_seeds[0])
    reference_directory = destination / f"seed_{reference_seed}"
    reference_metadata = json.loads(
        (reference_directory / "run_metadata.json").read_text(encoding="utf-8")
    )
    with np.load(reference_directory / "predictions.npz") as payload:
        y_validation = payload["y_validation"].copy()
        y_test = payload["y_test"].copy()
        validation_probability = payload[f"validation_{reference_model_key}"].copy()
        test_probability = payload[f"test_{reference_model_key}"].copy()
        validation_raw_probability = payload[f"validation_raw_{reference_model_key}"].copy()
        test_raw_probability = payload[f"test_raw_{reference_model_key}"].copy()
        raw_test_probabilities = {
            model_key: payload[f"test_raw_{model_key}"].copy() for model_key in model_names
        }

    frozen_manifest = {
        "artifact_schema_version": 1,
        "dataset_name": config["dataset"]["name"],
        "data_source": reference_metadata["data_source"],
        "configured_files": reference_metadata.get("configured_files", []),
        "model_key": reference_model_key,
        "model": reference_metadata["display_names"][reference_model_key],
        "selection_basis": "highest mean validation raw PR-AUC",
        "reference_seed": reference_seed,
        "threshold": reference_metadata["thresholds"][reference_model_key],
        "calibration_method": reference_metadata["calibration_methods"][reference_model_key],
        "config_sha256": stable_config_hash(config),
        "split_summary": reference_metadata["split_summary"],
        "feature_names": reference_metadata["feature_names"],
        "environment": reference_metadata["environment"],
        "quick_run": quick_run,
    }
    artifact_path, manifest_path = write_frozen_reference_artifact(
        destination,
        manifest=frozen_manifest,
        y_validation=y_validation,
        y_test=y_test,
        validation_probability=validation_probability,
        test_probability=test_probability,
        validation_raw_probability=validation_raw_probability,
        test_raw_probability=test_raw_probability,
    )

    bootstrap_iterations = int(config["evaluation"].get("predictive_bootstrap_iterations", 300))
    if quick_run:
        bootstrap_iterations = min(bootstrap_iterations, 30)
    bootstrap_rows: list[dict[str, Any]] = []
    for comparator in model_names:
        if comparator == reference_model_key:
            continue
        comparison = paired_bootstrap_pr_auc_difference(
            y_test,
            raw_test_probabilities[reference_model_key],
            raw_test_probabilities[comparator],
            n_bootstrap=bootstrap_iterations,
            seed=reference_seed,
            max_rows=10000 if quick_run else None,
        )
        bootstrap_rows.append(
            {
                "reference_model_key": reference_model_key,
                "comparator_model_key": comparator,
                **comparison,
            }
        )
    predictive_bootstrap = pd.DataFrame(bootstrap_rows)
    predictive_bootstrap.to_csv(destination / "paired_bootstrap_model_differences.csv", index=False)

    metadata = {
        "data_sources": sorted(data_sources),
        "seeds": effective_seeds,
        "quick_run": quick_run,
        "models": list(model_names),
        "selection_metric": "mean validation raw PR-AUC",
        "reference_model_key": reference_model_key,
        "reference_seed": reference_seed,
        "frozen_artifact": artifact_path.name,
        "frozen_manifest": manifest_path.name,
        "config_sha256": stable_config_hash(config),
        "environment": capture_environment(Path(config_path).resolve().parents[1]),
    }
    (destination / "repeated_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {
        "config": config,
        "metrics": metrics,
        "summary": summary,
        "data_summary": data_summary,
        "histories": histories,
        "calibration_comparison": calibration_comparison,
        "predictive_bootstrap": predictive_bootstrap,
        "reference_model_key": reference_model_key,
        "reference_seed": reference_seed,
        "frozen_artifact_path": artifact_path,
        "frozen_manifest_path": manifest_path,
        "output_dir": destination,
        "data_sources": sorted(data_sources),
        "seeds": effective_seeds,
    }
