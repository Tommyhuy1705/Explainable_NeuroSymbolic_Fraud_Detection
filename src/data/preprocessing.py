"""Leakage-aware splitting and tabular preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


@dataclass
class PreparedData:
    """Raw splits plus matrices transformed using train-fitted state only."""

    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    preprocessor: "FraudPreprocessor"


def _validate_split_sizes(split_config: dict[str, Any]) -> tuple[float, float, float]:
    sizes = (
        float(split_config["train_size"]),
        float(split_config["validation_size"]),
        float(split_config["test_size"]),
    )
    if not np.isclose(sum(sizes), 1.0):
        raise ValueError(f"Split sizes must sum to 1.0, got {sizes}")
    if any(size <= 0 for size in sizes):
        raise ValueError(f"All split sizes must be positive, got {sizes}")
    return sizes


def split_dataframe(
    frame: pd.DataFrame,
    target_column: str,
    split_config: dict[str, Any],
    time_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic temporal or stratified train/validation/test splits."""
    train_size, validation_size, _ = _validate_split_sizes(split_config)
    strategy = str(split_config.get("strategy", "stratified")).lower()

    if strategy == "temporal":
        if not time_column or time_column not in frame:
            raise KeyError(f"Temporal split requires time column {time_column!r}")
        ordered = frame.sort_values(time_column, kind="mergesort").reset_index(drop=True)
        train_end = int(len(ordered) * train_size)
        validation_end = int(len(ordered) * (train_size + validation_size))
        train = ordered.iloc[:train_end]
        validation = ordered.iloc[train_end:validation_end]
        test = ordered.iloc[validation_end:]
    elif strategy == "temporal_group":
        if not time_column or time_column not in frame:
            raise KeyError(f"Temporal-group split requires time column {time_column!r}")
        train_groups = set(split_config.get("train_groups", []))
        validation_groups = set(split_config.get("validation_groups", []))
        test_groups = set(split_config.get("test_groups", []))
        if not train_groups or not validation_groups or not test_groups:
            raise ValueError(
                "Temporal-group split requires non-empty train_groups, "
                "validation_groups, and test_groups"
            )
        if (
            train_groups & validation_groups
            or train_groups & test_groups
            or validation_groups & test_groups
        ):
            raise ValueError("Temporal-group assignments must be disjoint")
        observed_groups = set(frame[time_column].dropna().unique().tolist())
        configured_groups = train_groups | validation_groups | test_groups
        missing_groups = configured_groups - observed_groups
        unassigned_groups = observed_groups - configured_groups
        if missing_groups:
            raise ValueError(f"Configured temporal groups are absent from data: {sorted(missing_groups)}")
        if unassigned_groups:
            raise ValueError(f"Observed temporal groups are unassigned: {sorted(unassigned_groups)}")
        ordered = frame.sort_values(time_column, kind="mergesort")
        train = ordered[ordered[time_column].isin(train_groups)]
        validation = ordered[ordered[time_column].isin(validation_groups)]
        test = ordered[ordered[time_column].isin(test_groups)]
    elif strategy == "stratified":
        random_state = int(split_config.get("random_state", 42))
        train, remainder = train_test_split(
            frame,
            train_size=train_size,
            stratify=frame[target_column],
            random_state=random_state,
        )
        relative_validation = validation_size / (1.0 - train_size)
        validation, test = train_test_split(
            remainder,
            train_size=relative_validation,
            stratify=remainder[target_column],
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("A data split is empty; provide more rows or adjust split sizes")
    return train.copy(), validation.copy(), test.copy()


def split_integrity_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    time_column: str,
) -> pd.DataFrame:
    """Summarize temporal boundaries and group overlap for an executed split."""
    frames = {"train": train, "validation": validation, "test": test}
    group_sets = {
        name: set(frame[time_column].dropna().unique().tolist()) for name, frame in frames.items()
    }
    # Named temporal groups such as BAF months are useful audit evidence.  A
    # transaction timestamp such as IEEE-CIS TransactionDT can contain hundreds
    # of thousands of distinct values, so serializing every value would make the
    # run metadata needlessly large.  Bound only the displayed values; overlap
    # checks below still use the complete sets.
    display_groups = sum(len(groups) for groups in group_sets.values()) <= 256
    rows: list[dict[str, Any]] = []
    for name, frame in frames.items():
        other_groups = set().union(*(groups for key, groups in group_sets.items() if key != name))
        groups = group_sets[name]
        rows.append(
            {
                "split": name,
                "rows": len(frame),
                "time_min": frame[time_column].min(),
                "time_max": frame[time_column].max(),
                "time_group_count": len(groups),
                "time_groups": sorted(groups) if display_groups else [],
                "overlap_groups": sorted(groups & other_groups) if display_groups else [],
                "group_disjoint": not bool(groups & other_groups),
            }
        )
    return pd.DataFrame(rows)


class FraudPreprocessor:
    """Select columns and fit imputation/encoding/scaling on training data only."""

    def __init__(
        self,
        max_missing_fraction: float = 0.95,
        max_features: int | None = 120,
        categorical_max_cardinality: int = 500,
        scale_numeric: bool = True,
        drop_columns: list[str] | None = None,
    ) -> None:
        self.max_missing_fraction = max_missing_fraction
        self.max_features = max_features
        self.categorical_max_cardinality = categorical_max_cardinality
        self.scale_numeric = scale_numeric
        self.drop_columns = set(drop_columns or [])
        self.transformer: ColumnTransformer | None = None
        self.selected_features: list[str] = []
        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []

    def _select_features(self, frame: pd.DataFrame) -> list[str]:
        candidates = [column for column in frame.columns if column not in self.drop_columns]
        scored: list[tuple[float, str]] = []
        for column in candidates:
            series = frame[column]
            missing_fraction = float(series.isna().mean())
            if missing_fraction > self.max_missing_fraction or series.nunique(dropna=True) <= 1:
                continue
            if not pd.api.types.is_numeric_dtype(series):
                if series.nunique(dropna=True) > self.categorical_max_cardinality:
                    continue
            scored.append((1.0 - missing_fraction, column))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [column for _, column in scored]
        if self.max_features is not None:
            selected = selected[: int(self.max_features)]
        if not selected:
            raise ValueError("No usable features remain after preprocessing filters")
        return selected

    def fit(self, frame: pd.DataFrame) -> "FraudPreprocessor":
        self.selected_features = self._select_features(frame)
        selected = frame[self.selected_features]
        self.numeric_features = [
            column for column in selected if pd.api.types.is_numeric_dtype(selected[column])
        ]
        self.categorical_features = [
            column for column in selected if column not in self.numeric_features
        ]

        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if self.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(numeric_steps)
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )
        transformers: list[tuple[str, Any, list[str]]] = []
        if self.numeric_features:
            transformers.append(("numeric", numeric_pipeline, self.numeric_features))
        if self.categorical_features:
            transformers.append(("categorical", categorical_pipeline, self.categorical_features))
        self.transformer = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
        self.transformer.fit(selected)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("FraudPreprocessor must be fitted before transform")
        aligned = frame.reindex(columns=self.selected_features)
        matrix = self.transformer.transform(aligned)
        return np.nan_to_num(np.asarray(matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def get_feature_names_out(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]


def prepare_dataset(frame: pd.DataFrame, config: dict[str, Any]) -> PreparedData:
    """Split data first, then fit preprocessing exclusively on the training split."""
    dataset = config["dataset"]
    target = dataset["target_column"]
    id_column = dataset.get("id_column")
    time_column = dataset.get("time_column")
    train, validation, test = split_dataframe(frame, target, config["split"], time_column)

    excluded = [target]
    if id_column:
        excluded.append(id_column)
    preprocessing_config = dict(config["preprocessing"])
    preprocessing_config["drop_columns"] = list(
        set(preprocessing_config.get("drop_columns", [])) | set(excluded)
    )
    preprocessor = FraudPreprocessor(**preprocessing_config)
    X_train = preprocessor.fit_transform(train)
    X_validation = preprocessor.transform(validation)
    X_test = preprocessor.transform(test)

    return PreparedData(
        train_frame=train,
        validation_frame=validation,
        test_frame=test,
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=train[target].to_numpy(dtype=np.int64),
        y_validation=validation[target].to_numpy(dtype=np.int64),
        y_test=test[target].to_numpy(dtype=np.int64),
        feature_names=preprocessor.get_feature_names_out(),
        preprocessor=preprocessor,
    )
