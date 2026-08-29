"""Leakage-aware ingestion and temporal splitting for thesis stress datasets.

The utilities in this module deliberately keep the extended TransXion v2 and
AMLNet v1.0 benchmarks separate from the two primary thesis benchmarks.  They
load the complete source before any quick sampling, enforce a canonical
schema, construct only past-looking history features, and retain enough
provenance to audit every stress-test result.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5, sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


TRANSXION_V2_TRANSACTION_FILE = "tx.csv"
TRANSXION_V2_SHA256 = "d6c345f07a8d8e26123dba5fe4f6572ef198e04fb94f9b53facb3fef6d197a35"
TRANSXION_V2_SOURCE_REVISION = "53932595c37c23b9f55ea5ddf5984e4d57b88369"
TRANSXION_V2_EXPECTED_BYTES = 245_159_741

AMLNET_V1_FILE = "AMLNet_August 2025.csv"
AMLNET_V1_MD5 = "7668fc7d74c787e07546ce85c6f790b9"
AMLNET_V1_DOI = "10.5281/zenodo.16736515"

TRANSXION_CANONICAL_COLUMNS: Mapping[str, str] = {
    "Timestamp": "event_time",
    "From Bank": "sender_bank_id",
    # Some official-repository documentation/snapshots name both account
    # columns ``Account``; pandas exposes them as ``Account``/``Account.1``.
    # The checksum-pinned file may expose explicit From/To aliases instead, so
    # both schema spellings are supported and recorded in the data manifest.
    "Account": "sender_account_id",
    "Account.1": "receiver_account_id",
    "From Account": "sender_account_id",
    "To Bank": "receiver_bank_id",
    "To Account": "receiver_account_id",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "label",
}

AMLNET_CANONICAL_COLUMNS: Mapping[str, str] = {
    "step": "step",
    "type": "transaction_type",
    "amount": "amount",
    "category": "category",
    "nameOrig": "sender_account_id",
    "nameDest": "receiver_account_id",
    "oldbalanceOrg": "sender_balance_before",
    "newbalanceOrig": "sender_balance_after",
    "laundering_typology": "money_laundering_typology",
    "metadata": "metadata",
    "fraud_probability": "fraud_probability",
    "hour": "reported_hour",
    "day_of_week": "reported_day_of_week",
    "day_of_month": "reported_day_of_month",
    "month": "reported_month",
}

# Fields that encode an outcome, a post-transaction state, or a precomputed
# risk assessment must never be available to the predictor/explainer feature
# pipeline.  ``label`` is added to every dataset-specific denylist below.
AMLNET_LEAKAGE_DENYLIST = frozenset(
    {
        "label",
        "auxiliary_fraud_label",
        "auxiliary_money_laundering_label",
        "money_laundering_typology",
        "fraud_probability",
        "metadata",
        "sender_balance_after",
        "balance_depletion_fraction",
        "reported_hour",
        "reported_day_of_week",
        "reported_day_of_month",
        "reported_month",
        "risk_score",
        "customer_risk_score",
        "isFraud",
        "isMoneyLaundering",
    }
)
TRANSXION_LEAKAGE_DENYLIST = frozenset({"label", "Is Laundering"})

_INTERNAL_COLUMNS = frozenset({"__source_order"})
_LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_AMLNET_DATETIME_PATTERN = re.compile(
    r"datetime\.datetime\(\s*"
    r"(?P<year>\d{4})\s*,\s*(?P<month>\d{1,2})\s*,\s*(?P<day>\d{1,2})"
    r"(?:\s*,\s*(?P<hour>\d{1,2})"
    r"(?:\s*,\s*(?P<minute>\d{1,2})"
    r"(?:\s*,\s*(?P<second>\d{1,2})"
    r"(?:\s*,\s*(?P<microsecond>\d{1,6}))?"
    r")?)?)?\s*\)"
)


@dataclass(frozen=True)
class StressDataset:
    """Canonical stress dataset plus its audit and provenance contract."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    leakage_denylist: tuple[str, ...]
    manifest: dict[str, Any]
    source_paths: dict[str, Path]


@dataclass(frozen=True)
class TemporalStressSplit:
    """Timestamp-disjoint thesis split and four validation-only partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    calibration_fit: pd.DataFrame
    calibration_select: pd.DataFrame
    rule_audit: pd.DataFrame
    policy_select: pd.DataFrame
    manifest: dict[str, Any]


def _dataset_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    candidate = config.get("dataset", config)
    if not isinstance(candidate, Mapping):
        raise TypeError("config['dataset'] must be a mapping")
    return dict(candidate)


def _as_roots(value: str | Path | Iterable[str | Path] | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value).expanduser()]
    return [Path(item).expanduser() for item in value]


def _candidate_roots(
    explicit_roots: str | Path | Iterable[str | Path] | None,
    configured_roots: str | Path | Iterable[str | Path] | None = None,
) -> list[Path]:
    ordered = [
        *_as_roots(explicit_roots),
        *_as_roots(configured_roots),
        Path("data/raw"),
        Path("/kaggle/input"),
    ]
    result: list[Path] = []
    seen: set[str] = set()
    for root in ordered:
        try:
            key = str(root.resolve()).casefold()
        except OSError:
            key = str(root.absolute()).casefold()
        if key not in seen:
            result.append(root)
            seen.add(key)
    return result


def resolve_stress_data_file(
    filename: str | Path,
    data_roots: str | Path | Iterable[str | Path] | None = None,
    *,
    configured_roots: str | Path | Iterable[str | Path] | None = None,
) -> Path:
    """Resolve one exact filename recursively using deterministic root priority.

    The first root containing the requested basename wins.  Multiple recursive
    matches inside that same root are rejected so that a Kaggle input version
    cannot be selected silently.
    """

    requested = Path(filename).expanduser()
    if requested.is_file():
        return requested.resolve()
    basename = requested.name
    if not basename or basename in {".", ".."}:
        raise ValueError(f"Invalid dataset filename: {filename!r}")

    searched: list[str] = []
    for root in _candidate_roots(data_roots, configured_roots):
        searched.append(str(root))
        if root.is_file():
            if root.name == basename:
                return root.resolve()
            continue
        if not root.is_dir():
            continue
        direct = root / requested
        if direct.is_file():
            return direct.resolve()
        matches = sorted(
            (path for path in root.rglob(basename) if path.is_file() and path.name == basename),
            key=lambda path: str(path).casefold(),
        )
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            listed = ", ".join(str(path) for path in matches[:5])
            raise FileExistsError(
                f"Ambiguous exact filename {basename!r} below {root}: {listed}. "
                "Pass a narrower data_roots value."
            )
    raise FileNotFoundError(
        f"Could not find exact dataset file {basename!r}. Searched roots: "
        + (", ".join(searched) if searched else "none")
    )


def reject_git_lfs_pointer(path: str | Path) -> None:
    """Fail before pandas interprets a Git LFS pointer as a one-row CSV."""

    source = Path(path)
    with source.open("rb") as handle:
        prefix = handle.read(256).lstrip(b"\xef\xbb\xbf\x00\t\r\n ")
    if prefix.startswith(_LFS_PREFIX):
        raise ValueError(
            f"{source} is a Git LFS pointer, not dataset content. Download the LFS object "
            "or attach the published dataset as a Kaggle input."
        )


def file_checksum(path: str | Path, algorithm: str) -> str:
    """Compute a memory-bounded MD5 or SHA-256 provenance digest."""

    normalized = algorithm.lower().replace("-", "")
    if normalized == "sha256":
        digest = sha256()
    elif normalized == "md5":
        digest = md5()
    else:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm!r}")
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_contract(
    settings: Mapping[str, Any],
    *,
    default_algorithm: str,
    default_expected: str,
    verify_checksum: bool | None,
) -> tuple[str, str | None, bool]:
    configured = settings.get("checksum")
    algorithm = str(settings.get("checksum_algorithm", default_algorithm))
    expected: str | None = None
    if isinstance(configured, Mapping):
        algorithm = str(configured.get("algorithm", algorithm))
        expected_value = configured.get("value", configured.get("expected"))
        expected = str(expected_value) if expected_value else None
    elif configured:
        expected = str(configured)
    configured_expected = settings.get("expected_checksum")
    if configured_expected:
        expected = str(configured_expected)
    elif expected is None:
        expected = default_expected
    verify = bool(settings.get("verify_checksum", False)) if verify_checksum is None else bool(verify_checksum)
    return algorithm, expected, verify


def _checksum_record(
    path: Path,
    algorithm: str,
    expected: str | None,
    verify: bool,
) -> dict[str, Any]:
    actual = file_checksum(path, algorithm)
    normalized_expected = expected.lower() if expected else None
    verified = normalized_expected is not None and actual.lower() == normalized_expected
    if verify and normalized_expected is None:
        raise ValueError(f"Checksum verification requested for {path}, but no expected checksum was configured")
    if verify and not verified:
        raise ValueError(
            f"Checksum mismatch for {path.name}: expected {normalized_expected}, got {actual.lower()}"
        )
    return {
        "path": str(path),
        "filename": path.name,
        "bytes": int(path.stat().st_size),
        "algorithm": algorithm.lower().replace("-", ""),
        "actual": actual.lower(),
        "expected": normalized_expected,
        "verification_requested": verify,
        "verified": verified,
    }


def _read_csv(path: Path, settings: Mapping[str, Any]) -> pd.DataFrame:
    reject_git_lfs_pointer(path)
    options = settings.get("read_csv", {})
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise TypeError("dataset.read_csv must be a mapping")
    kwargs = {"low_memory": False, **dict(options)}
    # Loading is intentionally complete.  quick_rows is applied only after
    # timestamp parsing, sorting, joins, and causal feature construction.
    kwargs.pop("nrows", None)
    return pd.read_csv(path, **kwargs)


def _effective_column_map(
    raw_columns: Sequence[str],
    defaults: Mapping[str, str],
    custom: Mapping[str, str] | None,
) -> dict[str, str]:
    mapping = dict(defaults)
    if custom:
        for left, right in custom.items():
            # Accept both raw->canonical and canonical->raw forms.  The form
            # that names an observed raw column is unambiguous.
            if left in raw_columns:
                mapping[str(left)] = str(right)
            elif right in raw_columns:
                mapping[str(right)] = str(left)
            else:
                mapping[str(left)] = str(right)
    return mapping


def _rename_and_require(
    frame: pd.DataFrame,
    mapping: Mapping[str, str],
    required: Sequence[str],
) -> pd.DataFrame:
    renamed_targets = [mapping.get(column, column) for column in frame.columns]
    duplicates = sorted({column for column in renamed_targets if renamed_targets.count(column) > 1})
    if duplicates:
        raise ValueError(f"Canonical column mapping creates duplicates: {duplicates}")
    result = frame.rename(columns={key: value for key, value in mapping.items() if key in frame.columns})
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise KeyError(f"Required canonical columns are missing: {missing}")
    return result


def _strict_binary_target(frame: pd.DataFrame, target: str = "label") -> pd.DataFrame:
    result = frame.copy()
    values = pd.to_numeric(result[target], errors="raise")
    if values.isna().any():
        raise ValueError(f"Target {target!r} contains missing values")
    observed_values = set(values.astype(float).unique().tolist())
    if not observed_values.issubset({0.0, 1.0}):
        observed = sorted(observed_values)
        raise ValueError(f"Target {target!r} must be binary 0/1; observed {sorted(observed)}")
    result[target] = values.astype("int8")
    return result


def _stable_time_sort(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "__source_order" not in result:
        result.insert(0, "__source_order", np.arange(len(result), dtype=np.int64))
    if result["event_time"].isna().any():
        raise ValueError("event_time contains missing or unparseable timestamps")
    return result.sort_values(["event_time", "__source_order"], kind="mergesort").reset_index(drop=True)


def _normalize_key(series: pd.Series) -> pd.Series:
    def normalize(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        text = str(value).strip()
        return text if text else pd.NA

    return series.map(normalize).astype("string")


def _prior_count(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    # Source order is only a deterministic tie-break; it is not evidence that
    # one transaction precedes another transaction with the same timestamp.
    # Subtract the within-timestamp position so every row in a timestamp block
    # sees the same state built from strictly earlier timestamps only.
    key_position = frame.groupby(keys, dropna=False, sort=False).cumcount()
    same_time_position = frame.groupby(
        [*keys, "event_time"], dropna=False, sort=False
    ).cumcount()
    return (key_position - same_time_position).astype("int64")


def _prior_sum(frame: pd.DataFrame, keys: list[str], amount: str) -> pd.Series:
    numeric = pd.to_numeric(frame[amount], errors="coerce").fillna(0.0)
    key_grouper = [frame[key] for key in keys]
    timestamp_grouper = [*key_grouper, frame["event_time"]]
    cumulative = numeric.groupby(key_grouper, dropna=False, sort=False).cumsum()
    same_time_cumulative = numeric.groupby(
        timestamp_grouper, dropna=False, sort=False
    ).cumsum()
    return (cumulative - same_time_cumulative).astype("float64")


def _seconds_since_previous(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    key_group = frame.groupby(keys, dropna=False, sort=False)
    previous_row_time = key_group["event_time"].shift(1)
    first_at_timestamp = frame.groupby(
        [*keys, "event_time"], dropna=False, sort=False
    ).cumcount().eq(0)
    # Keep the previous timestamp only at the first row of a timestamp block,
    # then broadcast it through that block within the entity group.  The first
    # timestamp for an entity remains missing by design.
    previous_distinct = previous_row_time.where(first_at_timestamp).groupby(
        [frame[key] for key in keys], dropna=False, sort=False
    ).ffill()
    return (frame["event_time"] - previous_distinct).dt.total_seconds().astype("float64")


def _add_transxion_history(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    sender = ["sender_bank_id", "sender_account_id"]
    receiver = ["receiver_bank_id", "receiver_account_id"]
    pair = [*sender, *receiver]
    result["sender_prior_transaction_count"] = _prior_count(result, sender)
    result["sender_prior_total_amount_paid"] = _prior_sum(result, sender, "amount_paid")
    result["sender_seconds_since_previous"] = _seconds_since_previous(result, sender)
    result["receiver_prior_transaction_count"] = _prior_count(result, receiver)
    result["receiver_prior_total_amount_received"] = _prior_sum(result, receiver, "amount_received")
    result["receiver_seconds_since_previous"] = _seconds_since_previous(result, receiver)
    result["pair_prior_transaction_count"] = _prior_count(result, pair)
    return result


def _add_event_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["event_hour"] = result["event_time"].dt.hour.astype("int8")
    result["event_day_of_week"] = result["event_time"].dt.dayofweek.astype("int8")
    result["is_weekend"] = (result["event_day_of_week"] >= 5).astype("int8")
    return result


def _add_transxion_semantic_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = _add_event_calendar_features(frame)
    paid = pd.to_numeric(result["amount_paid"], errors="raise").astype("float64")
    received = pd.to_numeric(result["amount_received"], errors="raise").astype("float64")
    result["amount_paid_log"] = np.log1p(paid.clip(lower=0.0))
    result["amount_received_log"] = np.log1p(received.clip(lower=0.0))
    denominator = paid.abs() + received.abs() + np.finfo("float64").eps
    result["amount_relative_difference"] = ((paid - received).abs() / denominator).astype("float64")
    result["same_currency"] = (
        result["payment_currency"].astype("string").str.strip().str.upper()
        == result["receiving_currency"].astype("string").str.strip().str.upper()
    ).astype("int8")
    result["cross_bank"] = (
        _normalize_key(result["sender_bank_id"]) != _normalize_key(result["receiver_bank_id"])
    ).astype("int8")
    return result


def _add_amlnet_semantic_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = _add_event_calendar_features(frame)
    amount = pd.to_numeric(result["amount"], errors="raise").astype("float64")
    before = pd.to_numeric(result["sender_balance_before"], errors="coerce").astype("float64")
    result["amount_log"] = np.log1p(amount.clip(lower=0.0))
    denominator = before.abs().clip(lower=np.finfo("float64").eps)
    # Use only information available at transaction scoring time.  Deriving a
    # "depletion" feature from sender_balance_after would reintroduce a denied
    # post-transaction field through a proxy and is therefore prohibited.
    amount_to_balance = (amount / denominator).replace([np.inf, -np.inf], np.nan)
    result["amount_to_sender_balance_fraction"] = amount_to_balance.fillna(0.0).clip(0.0, 10.0)
    return result


def _add_amlnet_history(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    sender = ["sender_account_id"]
    receiver = ["receiver_account_id"]
    pair = ["sender_account_id", "receiver_account_id"]
    result["sender_prior_transaction_count"] = _prior_count(result, sender)
    result["sender_prior_total_amount"] = _prior_sum(result, sender, "amount")
    result["sender_seconds_since_previous"] = _seconds_since_previous(result, sender)
    result["receiver_prior_transaction_count"] = _prior_count(result, receiver)
    result["receiver_prior_total_amount"] = _prior_sum(result, receiver, "amount")
    result["receiver_seconds_since_previous"] = _seconds_since_previous(result, receiver)
    result["pair_prior_transaction_count"] = _prior_count(result, pair)
    return result


def parse_amlnet_metadata_timestamp(metadata: pd.Series) -> tuple[pd.Series, float]:
    """Vectorize AMLNet's ``datetime.datetime(...)`` metadata timestamp.

    No metadata is executed or evaluated.  Every row must match the published
    representation and produce a valid timestamp; partial parse coverage is a
    hard provenance failure.
    """

    extracted = metadata.astype("string").str.extract(_AMLNET_DATETIME_PATTERN, expand=True)
    required = ["year", "month", "day"]
    matched = extracted[required].notna().all(axis=1)
    numeric = extracted.apply(pd.to_numeric, errors="coerce")
    for column in ("hour", "minute", "second", "microsecond"):
        numeric[column] = numeric[column].fillna(0)
    safe_numeric = numeric[["year", "month", "day", "hour", "minute", "second"]].fillna(
        {"year": 1970, "month": 1, "day": 1, "hour": 0, "minute": 0, "second": 0}
    )
    base = pd.to_datetime(
        safe_numeric,
        errors="coerce",
        utc=True,
    )
    parsed = base + pd.to_timedelta(numeric["microsecond"], unit="us")
    valid = matched & parsed.notna()
    coverage = float(valid.mean()) if len(metadata) else 0.0
    if coverage != 1.0:
        examples = metadata.loc[~valid].astype(str).head(3).tolist()
        raise ValueError(
            "AMLNet metadata timestamp parse coverage must be 100%; "
            f"observed {coverage:.6f}. Invalid examples: {examples}"
        )
    return parsed, coverage


def _profile_file_settings(settings: Mapping[str, Any]) -> tuple[bool, list[str]]:
    profiles = settings.get("profiles", {})
    if profiles is None:
        profiles = {}
    if not isinstance(profiles, Mapping):
        raise TypeError("dataset.profiles must be a mapping")
    enabled = bool(settings.get("join_profiles", profiles.get("enabled", False)))
    configured = settings.get("profile_files", profiles.get("files"))
    if configured is None:
        files = ["person.csv", "merchant.csv"]
    elif isinstance(configured, Mapping):
        files = [str(value) for value in configured.values()]
    elif isinstance(configured, (str, Path)):
        files = [str(configured)]
    else:
        files = [str(value) for value in configured]
    return enabled, files


def _join_transxion_profiles(
    frame: pd.DataFrame,
    settings: Mapping[str, Any],
    transaction_path: Path,
    data_roots: str | Path | Iterable[str | Path] | None,
) -> tuple[pd.DataFrame, dict[str, float] | None, dict[str, Path], list[dict[str, Any]]]:
    enabled, filenames = _profile_file_settings(settings)
    if not enabled:
        return frame, None, {}, []

    profile_frames: list[pd.DataFrame] = []
    paths: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    roots = [transaction_path.parent, *_as_roots(data_roots)]
    for filename in filenames:
        path = resolve_stress_data_file(
            filename,
            roots,
            configured_roots=settings.get("data_roots"),
        )
        reject_git_lfs_pointer(path)
        profile = pd.read_csv(path, low_memory=False)
        required = {"bank", "bank_account_number"}
        missing = sorted(required.difference(profile.columns))
        if missing:
            raise KeyError(f"Profile file {path.name} is missing composite-key columns: {missing}")
        profile = profile.copy()
        profile["bank"] = _normalize_key(profile["bank"])
        profile["bank_account_number"] = _normalize_key(profile["bank_account_number"])
        if profile[["bank", "bank_account_number"]].isna().any(axis=None):
            raise ValueError(f"Profile file {path.name} contains missing composite-key values")
        if profile.duplicated(["bank", "bank_account_number"]).any():
            raise ValueError(
                f"Profile file {path.name} must be unique on (bank, bank_account_number)"
            )
        profile["profile_source"] = Path(filename).stem.lower()
        profile_frames.append(profile)
        paths[f"profile_{Path(filename).stem.lower()}"] = path
        records.append(
            {
                "role": "profile",
                "path": str(path),
                "filename": path.name,
                "bytes": int(path.stat().st_size),
                "algorithm": "sha256",
                "actual": file_checksum(path, "sha256"),
            }
        )

    combined = pd.concat(profile_frames, ignore_index=True, sort=False)
    profile_key = ["bank", "bank_account_number"]
    duplicated = combined.duplicated(profile_key, keep=False)
    if duplicated.any():
        examples = combined.loc[duplicated, profile_key].head(5).to_dict("records")
        raise ValueError(
            "Combined TransXion profiles are ambiguous on composite (bank, account) keys: "
            f"{examples}"
        )

    result = frame.copy()
    for column in ("sender_bank_id", "sender_account_id", "receiver_bank_id", "receiver_account_id"):
        result[column] = _normalize_key(result[column])

    coverage: dict[str, float] = {}
    payload = [column for column in combined.columns if column not in profile_key]
    for role in ("sender", "receiver"):
        renamed = combined.rename(
            columns={
                "bank": f"__{role}_profile_bank",
                "bank_account_number": f"__{role}_profile_account",
                **{column: f"{role}_profile_{column}" for column in payload},
            }
        )
        before = len(result)
        result = result.merge(
            renamed,
            how="left",
            left_on=[f"{role}_bank_id", f"{role}_account_id"],
            right_on=[f"__{role}_profile_bank", f"__{role}_profile_account"],
            validate="many_to_one",
            sort=False,
        )
        result = result.drop(columns=[f"__{role}_profile_bank", f"__{role}_profile_account"])
        if len(result) != before:
            raise AssertionError("Profile join changed the transaction grain")
        marker = f"{role}_profile_profile_source"
        coverage[role] = float(result[marker].notna().mean())
    return result, coverage, paths, records


def _temporal_quick_sample(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows < 2:
        raise ValueError("quick_rows must be at least 2 to retain temporal coverage")
    if len(frame) <= max_rows:
        return frame.copy()
    # Evenly spaced positions include both temporal endpoints and therefore
    # retain the complete observed time span.  Sampling happens after history
    # features are calculated on the full source.
    positions = np.linspace(0, len(frame) - 1, num=max_rows, dtype=np.int64)
    return frame.iloc[positions].copy().reset_index(drop=True)


def _schema_fingerprint(frame: pd.DataFrame) -> str:
    schema = [{"column": column, "dtype": str(dtype)} for column, dtype in frame.dtypes.items()]
    return sha256(json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest()


def _time_summary(frame: pd.DataFrame) -> dict[str, str | None]:
    if frame.empty:
        return {"minimum": None, "maximum": None}
    return {
        "minimum": frame["event_time"].min().isoformat(),
        "maximum": frame["event_time"].max().isoformat(),
    }


def _feature_columns(frame: pd.DataFrame, denylist: Iterable[str]) -> tuple[str, ...]:
    denied = set(denylist) | set(_INTERNAL_COLUMNS) | {"event_time"}
    return tuple(column for column in frame.columns if column not in denied and not column.startswith("__"))


def _build_manifest(
    *,
    dataset_name: str,
    role: str,
    source_record: dict[str, Any],
    source_records: list[dict[str, Any]],
    raw_frame: pd.DataFrame,
    full_frame: pd.DataFrame,
    analysis_frame: pd.DataFrame,
    required_columns: Sequence[str],
    feature_columns: Sequence[str],
    denylist: Sequence[str],
    profile_coverage: dict[str, float] | None,
    timestamp_parse_coverage: float,
    quick_rows: int | None,
    source_reference: Mapping[str, Any],
) -> dict[str, Any]:
    target = full_frame["label"]
    quick_enabled = len(analysis_frame) < len(full_frame)
    return {
        "dataset": dataset_name,
        "benchmark_role": role,
        "source_reference": dict(source_reference),
        "source_files": [source_record, *source_records],
        "population": {
            "rows": int(len(full_frame)),
            "positive_count": int(target.sum()),
            "positive_rate": float(target.mean()),
            "time": _time_summary(full_frame),
            "exact_duplicate_rows": int(
                full_frame.drop(columns=list(_INTERNAL_COLUMNS), errors="ignore").duplicated().sum()
            ),
        },
        "analysis_frame": {
            "rows": int(len(analysis_frame)),
            "positive_count": int(analysis_frame["label"].sum()),
            "positive_rate": float(analysis_frame["label"].mean()),
            "time": _time_summary(analysis_frame),
        },
        "schema": {
            "raw_columns": raw_frame.columns.tolist(),
            "canonical_columns": analysis_frame.columns.tolist(),
            "required_columns": list(required_columns),
            "dtypes": {column: str(dtype) for column, dtype in analysis_frame.dtypes.items()},
            "fingerprint_sha256": _schema_fingerprint(analysis_frame),
        },
        "data_quality": {
            "timestamp_parse_coverage": float(timestamp_parse_coverage),
            "timestamp_monotonic": bool(full_frame["event_time"].is_monotonic_increasing),
            "target_missing_count": int(full_frame["label"].isna().sum()),
            "observed_target_values": sorted(int(value) for value in target.unique()),
            "profile_join_coverage": profile_coverage,
        },
        "leakage_contract": {
            "denylist": sorted(set(denylist)),
            "feature_columns": list(feature_columns),
            "history_features_use_labels": False,
            "history_features_computed_before_quick_sampling": True,
        },
        "quick_sample": {
            "enabled": quick_enabled,
            "requested_rows": int(quick_rows) if quick_rows is not None else None,
            "sampling": "deterministic_even_spacing_after_full_temporal_processing" if quick_enabled else None,
            "full_time_span_retained": _time_summary(full_frame) == _time_summary(analysis_frame),
            "not_for_thesis_claims": quick_enabled,
        },
    }


def load_transxion_dataset(
    config: Mapping[str, Any],
    data_roots: str | Path | Iterable[str | Path] | None = None,
    *,
    verify_checksum: bool | None = None,
    quick_rows: int | None = None,
) -> StressDataset:
    """Load checksum-pinned TransXion transactions across documented headers."""

    settings = _dataset_settings(config)
    filename = settings.get(
        "data_file",
        settings.get("transaction_file", settings.get("file", TRANSXION_V2_TRANSACTION_FILE)),
    )
    path = resolve_stress_data_file(filename, data_roots, configured_roots=settings.get("data_roots"))
    reject_git_lfs_pointer(path)
    algorithm, expected, verify = _checksum_contract(
        settings,
        default_algorithm="sha256",
        default_expected=TRANSXION_V2_SHA256,
        verify_checksum=verify_checksum,
    )
    source_record = _checksum_record(path, algorithm, expected, verify)
    raw = _read_csv(path, settings)
    raw_columns = raw.columns.tolist()

    raw_target = str(settings.get("target_column", "Is Laundering"))
    raw_time = str(settings.get("time_column", "Timestamp"))
    defaults = dict(TRANSXION_CANONICAL_COLUMNS)
    defaults[raw_target] = "label"
    defaults[raw_time] = "event_time"
    custom_map = settings.get("column_map", settings.get("rename_columns"))
    if custom_map is not None and not isinstance(custom_map, Mapping):
        raise TypeError("dataset.column_map must be a mapping")
    mapping = _effective_column_map(raw_columns, defaults, custom_map)
    required = list(
        settings.get(
            "required_columns",
            [
                "event_time",
                "sender_bank_id",
                "sender_account_id",
                "receiver_bank_id",
                "receiver_account_id",
                "amount_received",
                "receiving_currency",
                "amount_paid",
                "payment_currency",
                "payment_format",
                "label",
            ],
        )
    )
    frame = _rename_and_require(raw, mapping, required)
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="coerce", utc=True)
    for column in ("amount_received", "amount_paid"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = _strict_binary_target(frame)
    frame = _add_transxion_semantic_features(frame)
    frame = _stable_time_sort(frame)

    frame, profile_coverage, profile_paths, profile_records = _join_transxion_profiles(
        frame, settings, path, data_roots
    )
    if bool(settings.get("add_history_features", True)):
        frame = _add_transxion_history(frame)

    configured_denylist = settings.get("leakage_denylist", [])
    denylist = tuple(sorted(set(TRANSXION_LEAKAGE_DENYLIST) | set(configured_denylist)))
    full_frame = frame
    effective_quick = quick_rows if quick_rows is not None else settings.get("quick_rows")
    analysis = (
        _temporal_quick_sample(full_frame, int(effective_quick))
        if effective_quick is not None
        else full_frame.copy()
    )
    features = _feature_columns(analysis, denylist)
    manifest = _build_manifest(
        dataset_name="transxion_v2",
        role="difficult_aml_generalization_stress_test",
        source_record=source_record,
        source_records=profile_records,
        raw_frame=raw,
        full_frame=full_frame,
        analysis_frame=analysis,
        required_columns=required,
        feature_columns=features,
        denylist=denylist,
        profile_coverage=profile_coverage,
        timestamp_parse_coverage=float(full_frame["event_time"].notna().mean()),
        quick_rows=int(effective_quick) if effective_quick is not None else None,
        source_reference={
            "canonical_dataset_name": settings.get("canonical_dataset_name", "TransXion"),
            "paper_revision": settings.get("paper_revision", "v2"),
            "paper_url": settings.get("source_paper", "https://arxiv.org/abs/2604.17420v2"),
            "repository_url": settings.get(
                "source_repository", "https://github.com/chaos-max/TransXion"
            ),
            "repository_revision": settings.get("source_revision", TRANSXION_V2_SOURCE_REVISION),
            "canonical_file": TRANSXION_V2_TRANSACTION_FILE,
            "canonical_bytes": TRANSXION_V2_EXPECTED_BYTES,
            "license_status": settings.get(
                "license_status", "verify_official_source_terms_before_redistribution"
            ),
            "dataset_v2_release_claimed": False,
        },
    )
    analysis.attrs.update(
        {
            "dataset_name": "transxion_v2",
            "quick_sample": manifest["quick_sample"],
            "data_manifest": manifest,
        }
    )
    return StressDataset(
        frame=analysis,
        feature_columns=features,
        leakage_denylist=denylist,
        manifest=manifest,
        source_paths={"transactions": path, **profile_paths},
    )


def load_amlnet_dataset(
    config: Mapping[str, Any],
    data_roots: str | Path | Iterable[str | Path] | None = None,
    *,
    verify_checksum: bool | None = None,
    quick_rows: int | None = None,
) -> StressDataset:
    """Load AMLNet v1.0 and parse its metadata timestamp without evaluation."""

    settings = _dataset_settings(config)
    filename = settings.get("data_file", settings.get("file", AMLNET_V1_FILE))
    path = resolve_stress_data_file(filename, data_roots, configured_roots=settings.get("data_roots"))
    reject_git_lfs_pointer(path)
    algorithm, expected, verify = _checksum_contract(
        settings,
        default_algorithm="md5",
        default_expected=AMLNET_V1_MD5,
        verify_checksum=verify_checksum,
    )
    source_record = _checksum_record(path, algorithm, expected, verify)
    raw = _read_csv(path, settings)
    raw_columns = raw.columns.tolist()

    raw_target = str(settings.get("target_column", "isMoneyLaundering"))
    raw_metadata = str(settings.get("metadata_column", "metadata"))
    defaults = dict(AMLNET_CANONICAL_COLUMNS)
    defaults["isFraud"] = "auxiliary_fraud_label"
    defaults["isMoneyLaundering"] = "auxiliary_money_laundering_label"
    defaults[raw_target] = "label"
    defaults[raw_metadata] = "metadata"
    custom_map = settings.get("column_map", settings.get("rename_columns"))
    if custom_map is not None and not isinstance(custom_map, Mapping):
        raise TypeError("dataset.column_map must be a mapping")
    mapping = _effective_column_map(raw_columns, defaults, custom_map)
    required = list(
        settings.get(
            "required_columns",
            [
                "step",
                "transaction_type",
                "amount",
                "category",
                "sender_account_id",
                "receiver_account_id",
                "sender_balance_before",
                "sender_balance_after",
                "label",
                "metadata",
            ],
        )
    )
    frame = _rename_and_require(raw, mapping, required)
    frame["event_time"], parse_coverage = parse_amlnet_metadata_timestamp(frame["metadata"])
    frame["amount"] = pd.to_numeric(frame["amount"], errors="raise")
    for column in ("sender_balance_before", "sender_balance_after", "step"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = _strict_binary_target(frame)
    frame = _add_amlnet_semantic_features(frame)
    frame = _stable_time_sort(frame)
    if bool(settings.get("add_history_features", True)):
        frame = _add_amlnet_history(frame)

    configured_denylist = settings.get("leakage_denylist", [])
    denylist = tuple(sorted(set(AMLNET_LEAKAGE_DENYLIST) | set(configured_denylist)))
    full_frame = frame
    effective_quick = quick_rows if quick_rows is not None else settings.get("quick_rows")
    analysis = (
        _temporal_quick_sample(full_frame, int(effective_quick))
        if effective_quick is not None
        else full_frame.copy()
    )
    features = _feature_columns(analysis, denylist)
    manifest = _build_manifest(
        dataset_name="amlnet_v1_0",
        role="saturation_and_limit_stress_test",
        source_record=source_record,
        source_records=[],
        raw_frame=raw,
        full_frame=full_frame,
        analysis_frame=analysis,
        required_columns=[*required, "event_time"],
        feature_columns=features,
        denylist=denylist,
        profile_coverage=None,
        timestamp_parse_coverage=parse_coverage,
        quick_rows=int(effective_quick) if effective_quick is not None else None,
        source_reference={
            "canonical_dataset_name": settings.get("canonical_dataset_name", "AMLNet"),
            "doi": settings.get("source_doi", AMLNET_V1_DOI),
            "record_url": settings.get("source_record", "https://zenodo.org/records/16736515"),
            "version": str(settings.get("version", "1.0")),
            "license": settings.get("license", "CC BY-NC 4.0"),
            "source_version_discrepancy": settings.get(
                "source_version_discrepancy",
                "Zenodo record v1/v1.0 is pinned by exact file and MD5; repository previews may differ.",
            ),
        },
    )
    analysis.attrs.update(
        {
            "dataset_name": "amlnet_v1_0",
            "quick_sample": manifest["quick_sample"],
            "data_manifest": manifest,
        }
    )
    return StressDataset(
        frame=analysis,
        feature_columns=features,
        leakage_denylist=denylist,
        manifest=manifest,
        source_paths={"transactions": path},
    )


def load_stress_dataset(
    config: Mapping[str, Any],
    data_roots: str | Path | Iterable[str | Path] | None = None,
    *,
    verify_checksum: bool | None = None,
    quick_rows: int | None = None,
) -> StressDataset:
    """Dispatch a configured TransXion v2 or AMLNet v1.0 loader."""

    settings = _dataset_settings(config)
    name = (
        str(settings.get("name", ""))
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )
    if name in {"transxion", "transxion_v2", "transxion_2"}:
        return load_transxion_dataset(
            config,
            data_roots,
            verify_checksum=verify_checksum,
            quick_rows=quick_rows,
        )
    if name in {"amlnet", "amlnet_v1", "amlnet_v1_0", "amlnet_1_0"}:
        return load_amlnet_dataset(
            config,
            data_roots,
            verify_checksum=verify_checksum,
            quick_rows=quick_rows,
        )
    raise ValueError(f"Unsupported stress dataset name: {settings.get('name')!r}")


def _safe_boundary(timestamps: pd.Series, desired: int, lower: int, upper: int) -> int:
    values = timestamps.reset_index(drop=True)
    boundaries = np.flatnonzero(values.iloc[1:].to_numpy() != values.iloc[:-1].to_numpy()) + 1
    candidates = boundaries[(boundaries >= lower) & (boundaries <= upper)]
    if len(candidates) == 0:
        raise ValueError("Not enough distinct timestamp groups for leakage-safe temporal partitions")
    distances = np.abs(candidates - int(desired))
    return int(candidates[np.argmin(distances)])


def _assert_both_classes(frame: pd.DataFrame, name: str, target: str) -> None:
    observed = set(pd.to_numeric(frame[target], errors="raise").astype(int).unique().tolist())
    if observed != {0, 1}:
        raise ValueError(
            f"Temporal partition {name!r} must contain both target classes; observed {sorted(observed)}"
        )


def _partition_manifest(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "positive_count": int(frame[target].sum()),
        "positive_rate": float(frame[target].mean()),
        "time": _time_summary(frame),
    }


def temporal_stress_split(
    frame: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
    *,
    time_column: str = "event_time",
    target_column: str = "label",
) -> TemporalStressSplit:
    """Create proposal-compliant timestamp-safe 60/20/20 partitions.

    Validation is split, still chronologically, into independent
    ``calibration_fit``, ``calibration_select``, ``rule_audit``, and
    ``policy_select`` quarters.  Every partition must contain both classes;
    failure is reported rather than silently resampling across time.
    """

    if time_column not in frame or target_column not in frame:
        raise KeyError(f"Expected columns {time_column!r} and {target_column!r}")
    split_config = dict((config or {}).get("split", config or {}))
    train_ratio = float(split_config.get("train_size", 0.60))
    validation_ratio = float(split_config.get("validation_size", 0.20))
    test_ratio = float(split_config.get("test_size", 0.20))
    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
        raise ValueError("train_size + validation_size + test_size must equal 1.0")
    if min(train_ratio, validation_ratio, test_ratio) <= 0:
        raise ValueError("All temporal split ratios must be positive")

    ordered = frame.copy()
    ordered[time_column] = pd.to_datetime(ordered[time_column], errors="coerce", utc=True)
    if ordered[time_column].isna().any():
        raise ValueError(f"{time_column!r} contains missing or invalid timestamps")
    if "__source_order" not in ordered:
        ordered["__source_order"] = np.arange(len(ordered), dtype=np.int64)
    ordered = ordered.sort_values([time_column, "__source_order"], kind="mergesort").reset_index(drop=True)
    if time_column != "event_time":
        ordered["event_time"] = ordered[time_column]
    if target_column != "label":
        ordered["label"] = ordered[target_column]
    ordered = _strict_binary_target(ordered, "label")
    target_column = "label"

    n_rows = len(ordered)
    if n_rows < 40:
        raise ValueError("At least 40 rows are required for eight class-audited temporal partitions")
    train_cut = _safe_boundary(
        ordered["event_time"],
        round(n_rows * train_ratio),
        1,
        n_rows - 2,
    )
    validation_cut = _safe_boundary(
        ordered["event_time"],
        round(n_rows * (train_ratio + validation_ratio)),
        train_cut + 1,
        n_rows - 1,
    )
    train = ordered.iloc[:train_cut].copy()
    validation = ordered.iloc[train_cut:validation_cut].copy()
    test = ordered.iloc[validation_cut:].copy()

    validation_times = validation["event_time"].reset_index(drop=True)
    validation_size = len(validation)
    subcuts: list[int] = []
    lower = 1
    for fraction in (0.25, 0.50, 0.75):
        remaining_cuts = 3 - len(subcuts)
        upper = validation_size - remaining_cuts
        cut = _safe_boundary(validation_times, round(validation_size * fraction), lower, upper)
        subcuts.append(cut)
        lower = cut + 1
    calibration_fit = validation.iloc[: subcuts[0]].copy()
    calibration_select = validation.iloc[subcuts[0] : subcuts[1]].copy()
    rule_audit = validation.iloc[subcuts[1] : subcuts[2]].copy()
    policy_select = validation.iloc[subcuts[2] :].copy()

    partitions = {
        "train": train,
        "validation": validation,
        "test": test,
        "calibration_fit": calibration_fit,
        "calibration_select": calibration_select,
        "rule_audit": rule_audit,
        "policy_select": policy_select,
    }
    for name, partition in partitions.items():
        _assert_both_classes(partition, name, target_column)

    chronology_pairs = [
        ("train", train, "validation", validation),
        ("validation", validation, "test", test),
        ("calibration_fit", calibration_fit, "calibration_select", calibration_select),
        ("calibration_select", calibration_select, "rule_audit", rule_audit),
        ("rule_audit", rule_audit, "policy_select", policy_select),
    ]
    for left_name, left, right_name, right in chronology_pairs:
        if not left["event_time"].max() < right["event_time"].min():
            raise AssertionError(
                f"Equal timestamps cross {left_name}/{right_name}; temporal boundary is not leakage-safe"
            )

    manifest = {
        "strategy": "timestamp_safe_temporal",
        "requested_ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "boundary_policy": "nearest row-count boundary moved between distinct timestamps",
        "validation_policy": "four chronological timestamp-disjoint quarters",
        "both_classes_asserted": True,
        "partitions": {
            name: _partition_manifest(partition, target_column)
            for name, partition in partitions.items()
        },
        "source_order_tie_break": "stable original row order",
    }
    return TemporalStressSplit(
        train=train,
        validation=validation,
        test=test,
        calibration_fit=calibration_fit,
        calibration_select=calibration_select,
        rule_audit=rule_audit,
        policy_select=policy_select,
        manifest=manifest,
    )


def make_synthetic_transxion_test_fixture(
    root: str | Path,
    *,
    n_rows: int = 400,
    seed: int = 42,
) -> dict[str, Path]:
    """Write a deterministic TransXion-shaped fixture for tests only."""

    if n_rows < 100:
        raise ValueError("The test fixture requires at least 100 rows")
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    account_numbers = [f"A{index:06d}" for index in range(12)]
    banks = [str(1000 + index // 4) for index in range(12)]
    account_index = np.arange(n_rows) % len(account_numbers)
    receiver_index = (account_index + 3) % len(account_numbers)
    event_time = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(
        np.arange(n_rows) // 2, unit="min"
    )
    amount_paid = np.round(rng.lognormal(3.0, 0.7, n_rows), 2)
    transactions = pd.DataFrame(
        {
            "Timestamp": event_time.strftime("%Y-%m-%d %H:%M:%S"),
            "From Bank": np.asarray(banks)[account_index],
            "From Account": np.asarray(account_numbers)[account_index],
            "To Bank": np.asarray(banks)[receiver_index],
            "To Account": np.asarray(account_numbers)[receiver_index],
            "Amount Received": np.round(amount_paid * 0.98, 2),
            "Receiving Currency": np.where(account_index % 2 == 0, "USD", "EUR"),
            "Amount Paid": amount_paid,
            "Payment Currency": np.where(account_index % 2 == 0, "USD", "EUR"),
            "Payment Format": np.where(account_index % 3 == 0, "Wire", "Card"),
            "Is Laundering": ((np.arange(n_rows) % 7) == 0).astype("int8"),
        }
    )
    person = pd.DataFrame(
        {
            "person_id": [f"P{index:04d}" for index in range(6)],
            "bank_account_number": account_numbers[:6],
            "bank": banks[:6],
            "person_age": np.arange(25, 31),
            "person_education": ["college"] * 6,
            "person_gender": ["F", "M"] * 3,
            "person_marital_status": ["single"] * 6,
            "person_occupation": ["analyst"] * 6,
        }
    )
    merchant = pd.DataFrame(
        {
            "merchant_id": [f"M{index:04d}" for index in range(6)],
            "bank_account_number": account_numbers[6:],
            "bank": banks[6:],
            "description": ["test merchant"] * 6,
            "type": ["retail"] * 6,
            "registered_capital": np.arange(1_000, 7_000, 1_000),
            "industry": ["services"] * 6,
            "operating_status": ["active"] * 6,
            "establishment_date": ["2020-01-01"] * 6,
            "legal_representative_id": [f"L{index:04d}" for index in range(6)],
        }
    )
    paths = {
        "transactions": destination / TRANSXION_V2_TRANSACTION_FILE,
        "person": destination / "person.csv",
        "merchant": destination / "merchant.csv",
    }
    transactions.to_csv(paths["transactions"], index=False)
    person.to_csv(paths["person"], index=False)
    merchant.to_csv(paths["merchant"], index=False)
    return paths


def make_synthetic_amlnet_test_fixture(
    root: str | Path,
    *,
    n_rows: int = 400,
    seed: int = 42,
) -> Path:
    """Write a deterministic AMLNet-v1-shaped fixture for tests only."""

    if n_rows < 100:
        raise ValueError("The test fixture requires at least 100 rows")
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    timestamps = pd.Timestamp("2025-02-01") + pd.to_timedelta(np.arange(n_rows) // 2, unit="min")
    metadata = [
        (
            "{'timestamp': datetime.datetime("
            f"{timestamp.year}, {timestamp.month}, {timestamp.day}, {timestamp.hour}, "
            f"{timestamp.minute}, {timestamp.second}, {index % 1_000_000}), "
            "'location': {'country': 'Australia'}, "
            "'risk_indicators': {'risk_score': 99.0}}"
        )
        for index, timestamp in enumerate(timestamps)
    ]
    amount = np.round(rng.lognormal(4.0, 0.8, n_rows), 6)
    frame = pd.DataFrame(
        {
            "step": np.arange(n_rows) // 20,
            "type": np.where(np.arange(n_rows) % 2 == 0, "TRANSFER", "DEBIT"),
            "amount": amount,
            "category": np.where(np.arange(n_rows) % 3 == 0, "Other", "Retail"),
            "nameOrig": [f"C{index % 17:04d}" for index in range(n_rows)],
            "nameDest": [f"C{(index + 5) % 19:04d}" for index in range(n_rows)],
            "oldbalanceOrg": 10_000.0 + np.arange(n_rows),
            "newbalanceOrig": 10_000.0 + np.arange(n_rows) - amount,
            "isFraud": ((np.arange(n_rows) % 11) == 0).astype("int8"),
            "isMoneyLaundering": ((np.arange(n_rows) % 7) == 0).astype("int8"),
            "laundering_typology": np.where(np.arange(n_rows) % 7 == 0, "fan_out", "normal"),
            "metadata": metadata,
            "fraud_probability": np.where(np.arange(n_rows) % 11 == 0, 0.99, 0.01),
            "hour": timestamps.hour,
            "day_of_week": timestamps.dayofweek,
            "day_of_month": timestamps.day,
            "month": timestamps.month,
        }
    )
    path = destination / AMLNET_V1_FILE
    frame.to_csv(path, index=False)
    return path
