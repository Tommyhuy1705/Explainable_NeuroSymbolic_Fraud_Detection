"""Reusable experiment orchestration shared by scripts and Kaggle notebooks."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data import (
    load_config,
    load_fraud_dataframe,
    make_synthetic_baf_data,
    make_synthetic_fraud_data,
    prepare_dataset,
)
from src.evaluation import ProbabilityCalibrator, evaluate_binary_predictions, select_threshold
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
) -> tuple[dict[str, float], dict[str, float], float, np.ndarray, np.ndarray]:
    evaluation = config["evaluation"]
    calibrator = ProbabilityCalibrator(str(evaluation.get("calibration", "none")))
    calibrator.fit(validation_probabilities, y_validation)
    calibrated_validation = calibrator.transform(validation_probabilities)
    calibrated_test = calibrator.transform(test_probabilities)
    threshold, _ = select_threshold(
        y_validation,
        calibrated_validation,
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
    return validation_metrics, test_metrics, threshold, calibrated_validation, calibrated_test


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
    thresholds: dict[str, float] = {}
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

        val_metrics, test_metrics, threshold, val_probability, test_probability = _evaluate_probabilities(
            config,
            prepared.y_validation,
            val_probability,
            prepared.y_test,
            test_probability,
        )
        for split_name, metrics in (("validation", val_metrics), ("test", test_metrics)):
            rows.append({"model": display_name, "model_key": model_name, "split": split_name, **metrics})
        models[model_name] = model
        validation_probabilities[model_name] = val_probability
        test_probabilities[model_name] = test_probability
        thresholds[model_name] = threshold

    metrics = pd.DataFrame(rows)
    destination = Path(output_dir or config["project"]["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(destination / "predictive_metrics.csv", index=False)
    np.savez_compressed(
        destination / "predictions.npz",
        y_validation=prepared.y_validation,
        y_test=prepared.y_test,
        **{f"validation_{name}": values for name, values in validation_probabilities.items()},
        **{f"test_{name}": values for name, values in test_probabilities.items()},
    )
    metadata = {
        "data_source": data_source,
        "rows": len(frame),
        "features": prepared.X_train.shape[1],
        "thresholds": thresholds,
        "models": list(model_names),
        "seed": seed,
        "quick_run": quick_run,
    }
    (destination / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "config": config,
        "frame": frame,
        "prepared": prepared,
        "metrics": metrics,
        "models": models,
        "validation_probabilities": validation_probabilities,
        "test_probabilities": test_probabilities,
        "thresholds": thresholds,
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
    configured_seeds = list(seeds or config["evaluation"].get("seed_list", [42, 123, 2026]))
    if not configured_seeds:
        raise ValueError("At least one experiment seed is required")
    effective_seeds = configured_seeds[:1] if quick_run else configured_seeds
    destination = Path(output_dir or config["project"]["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)

    metric_frames: list[pd.DataFrame] = []
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
        data_sources.add(str(run["data_source"]))
        histories[int(run_seed)] = run["histories"]
        if data_summary is None:
            prepared = run["prepared"]
            data_summary = pd.DataFrame(
                {
                    "split": ["train", "validation", "test"],
                    "rows": [len(prepared.y_train), len(prepared.y_validation), len(prepared.y_test)],
                    "fraud_rate": [
                        prepared.y_train.mean(),
                        prepared.y_validation.mean(),
                        prepared.y_test.mean(),
                    ],
                }
            )
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
    metrics.to_csv(destination / "predictive_metrics_all_seeds.csv", index=False)
    summary.to_csv(destination / "predictive_metrics_summary.csv", index=False)
    metadata = {
        "data_sources": sorted(data_sources),
        "seeds": effective_seeds,
        "quick_run": quick_run,
        "models": list(model_names),
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
        "output_dir": destination,
        "data_sources": sorted(data_sources),
        "seeds": effective_seeds,
    }
