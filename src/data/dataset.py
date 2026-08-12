"""Dataset loading utilities for IEEE-CIS, BAF, and deterministic smoke data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a mapping: {config_path}")
    return config


def _candidate_roots(data_root: str | Path | None) -> list[Path]:
    roots: list[Path] = []
    if data_root is not None:
        roots.append(Path(data_root))
    roots.extend([Path("data/raw"), Path("/kaggle/input")])
    return [root.expanduser() for root in roots if root.exists()]


def resolve_data_file(filename: str, data_root: str | Path | None = None) -> Path:
    """Resolve a dataset file locally or below Kaggle's input mount."""
    direct = Path(filename)
    if direct.exists():
        return direct.resolve()

    for root in _candidate_roots(data_root):
        candidate = root / filename
        if candidate.exists():
            return candidate.resolve()
        matches = list(root.glob(f"**/{filename}"))
        if matches:
            return matches[0].resolve()
    searched = ", ".join(str(path) for path in _candidate_roots(data_root)) or "no existing roots"
    raise FileNotFoundError(f"Could not find {filename!r}. Searched: {searched}")


def _read_csv(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=max_rows, low_memory=False)


def _add_ieee_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "TransactionDT" in result:
        seconds = pd.to_numeric(result["TransactionDT"], errors="coerce")
        result["transaction_hour"] = (seconds // 3600) % 24
        result["transaction_day"] = seconds // 86_400
    if "TransactionAmt" in result:
        amount = pd.to_numeric(result["TransactionAmt"], errors="coerce")
        result["transaction_amount_log"] = np.log1p(amount.clip(lower=0))
    return result


def load_fraud_dataframe(
    config: dict[str, Any],
    data_root: str | Path | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load the configured fraud dataset without fitting any preprocessing state."""
    dataset = config["dataset"]
    name = str(dataset["name"]).lower()
    row_limit = max_rows if max_rows is not None else dataset.get("max_rows")

    if name == "ieee_cis":
        transaction_path = resolve_data_file(dataset["transaction_file"], data_root)
        frame = _read_csv(transaction_path, row_limit)
        identity_name = dataset.get("identity_file")
        if identity_name:
            try:
                identity_path = resolve_data_file(identity_name, data_root)
            except FileNotFoundError:
                identity_path = None
            if identity_path is not None:
                identity = _read_csv(identity_path, None)
                id_column = dataset.get("id_column", "TransactionID")
                frame = frame.merge(identity, how="left", on=id_column, validate="one_to_one")
        frame = _add_ieee_derived_features(frame)
    elif name == "baf":
        data_path = resolve_data_file(dataset["data_file"], data_root)
        frame = _read_csv(data_path, row_limit)
    else:
        raise ValueError(f"Unsupported dataset name: {name}")

    target = dataset["target_column"]
    if target not in frame:
        raise KeyError(f"Target column {target!r} is missing from the loaded dataset")
    frame[target] = pd.to_numeric(frame[target], errors="raise").astype("int8")
    return frame


def make_synthetic_fraud_data(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create deterministic IEEE-like data for tests and notebook smoke runs."""
    rng = np.random.default_rng(seed)
    transaction_amount = rng.lognormal(mean=3.5, sigma=1.0, size=n_rows)
    transaction_dt = np.sort(rng.integers(0, 120 * 86_400, size=n_rows))
    transaction_hour = (transaction_dt // 3600) % 24
    c1 = rng.poisson(2.0, size=n_rows)
    device_type = rng.choice(["desktop", "mobile", None], size=n_rows, p=[0.45, 0.50, 0.05])
    email_domain = rng.choice(["common.com", "mail.net", "risk.example", None], size=n_rows)
    risk_logit = (
        -4.5
        + 0.012 * np.maximum(transaction_amount - 120, 0)
        + 0.55 * (c1 >= 5)
        + 0.65 * np.isin(transaction_hour, [0, 1, 2, 3, 4, 5])
        + 0.80 * (email_domain == "risk.example")
    )
    fraud_probability = 1.0 / (1.0 + np.exp(-np.clip(risk_logit, -20, 20)))
    target = rng.binomial(1, fraud_probability)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(1, n_rows + 1),
            "TransactionDT": transaction_dt,
            "TransactionAmt": transaction_amount,
            "transaction_hour": transaction_hour,
            "transaction_day": transaction_dt // 86_400,
            "transaction_amount_log": np.log1p(transaction_amount),
            "C1": c1,
            "card1": rng.integers(1000, 2000, size=n_rows).astype(str),
            "DeviceType": device_type,
            "P_emaildomain": email_domain,
            "isFraud": target,
        }
    )


def make_synthetic_baf_data(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create deterministic BAF-like data for local and notebook smoke runs."""
    rng = np.random.default_rng(seed)
    month = np.sort(rng.integers(0, 8, size=n_rows))
    velocity = rng.gamma(shape=2.0, scale=300.0, size=n_rows)
    device_emails = rng.poisson(1.2, size=n_rows)
    foreign_request = rng.binomial(1, 0.08, size=n_rows)
    session_length = rng.lognormal(mean=2.5, sigma=0.8, size=n_rows)
    credit_limit = rng.choice([200, 500, 1000, 1500, 2000], size=n_rows)
    age = rng.integers(18, 75, size=n_rows)
    income = np.clip(rng.beta(2.0, 5.0, size=n_rows), 0, 1)
    credit_risk = rng.normal(120, 55, size=n_rows)
    logit = (
        -5.0
        + 0.0015 * np.maximum(velocity - 800, 0)
        + 0.45 * (device_emails >= 3)
        + 0.9 * foreign_request
        + 0.0005 * np.maximum(credit_limit - 1000, 0)
        + 0.012 * np.maximum(credit_risk - 140, 0)
    )
    probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
    target = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "month": month,
            "velocity_6h": velocity,
            "device_distinct_emails_8w": device_emails,
            "foreign_request": foreign_request,
            "session_length_in_minutes": session_length,
            "proposed_credit_limit": credit_limit,
            "customer_age": age,
            "income": income,
            "credit_risk_score": credit_risk,
            "payment_type": rng.choice(["AA", "AB", "AC", "AD", "AE"], size=n_rows),
            "employment_status": rng.choice(["CA", "CB", "CC", "CD", "CE", "CF", "CG"], size=n_rows),
            "fraud_bool": target,
        }
    )
