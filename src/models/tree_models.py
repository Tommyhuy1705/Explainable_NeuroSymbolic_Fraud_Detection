"""Tree-model factory with deterministic fallbacks for Kaggle and local runs."""

from __future__ import annotations

import importlib.util
from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier


def tree_backend_name(requested: str = "xgboost") -> str:
    requested = requested.lower()
    if requested == "xgboost" and importlib.util.find_spec("xgboost"):
        return "xgboost"
    if requested == "lightgbm" and importlib.util.find_spec("lightgbm"):
        return "lightgbm"
    if importlib.util.find_spec("xgboost"):
        return "xgboost"
    if importlib.util.find_spec("lightgbm"):
        return "lightgbm"
    return "hist_gradient_boosting"


def build_tree_classifier(config: dict[str, Any], random_state: int = 42):
    """Build XGBoost/LightGBM when installed, otherwise use sklearn's HGB."""
    requested = str(config.get("backend", "xgboost"))
    backend = tree_backend_name(requested)

    if backend == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=int(config.get("n_estimators", 500)),
            max_depth=int(config.get("max_depth", 8)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            subsample=float(config.get("subsample", 0.85)),
            colsample_bytree=float(config.get("colsample_bytree", 0.8)),
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=int(config.get("n_estimators", 400)),
            max_depth=int(config.get("max_depth", -1)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            subsample=float(config.get("subsample", 0.85)),
            colsample_bytree=float(config.get("colsample_bytree", 0.85)),
            objective="binary",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    return HistGradientBoostingClassifier(
        learning_rate=float(config.get("learning_rate", 0.08)),
        max_iter=min(int(config.get("n_estimators", 300)), 300),
        max_depth=None if int(config.get("max_depth", -1)) < 0 else int(config["max_depth"]),
        random_state=random_state,
    )
