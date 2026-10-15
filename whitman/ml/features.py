
from __future__ import annotations

import json
from typing import Any

import pandas as pd

NUMERIC_FEATURES = (
    "total_count",
    "non_empty_count",
    "missing_count",
    "missing_ratio",
    "unique_count",
    "unique_ratio",
)

CATEGORICAL_FEATURES = (
    "inferred_type",
    "profile_role_hint",
)


def _clean_text(value: Any) -> str:
    """Convert a value to normalized, readable text."""
    if value is None:
        return ""

    text = str(value).strip()
    return " ".join(text.split())


def _sample_values(column: dict[str, Any]) -> list[str]:
    """Extract sample values from either profiler output or manual data."""
    values = column.get("sample_values", [])

    if not values:
        values = column.get("frequent_values", [])

    result: list[str] = []

    for value in values:
        if isinstance(value, dict):
            value = value.get("value")

        text = _clean_text(value)
        if text and text not in result:
            result.append(text)

    return result


def build_column_features(
    column: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert one profiler column record into ML-ready features.

    Expected input is a column entry from profile_products(), optionally
    enriched with additional metadata.

    The returned dictionary intentionally excludes the target label.
    """
    name = _clean_text(column.get("name")).lower()
    inferred_type = _clean_text(
        column.get("inferred_type", "unknown")
    ).lower()

    role_hint = _clean_text(
        column.get("role", "unknown")
    ).lower()

    samples = _sample_values(column)

    # Text is intended for TF-IDF or a local sentence-embedding model.
    text_parts = [
        f"column name: {name.replace('_', ' ')}",
        f"inferred type: {inferred_type}",
        f"profile role hint: {role_hint.replace('_', ' ')}",
    ]

    if samples:
        text_parts.append(
            "sample values: " + " ; ".join(samples[:8])
        )

    numeric: dict[str, float] = {}

    for feature in NUMERIC_FEATURES:
        value = column.get(feature, 0)

        try:
            numeric[feature] = float(value or 0)
        except (TypeError, ValueError):
            numeric[feature] = 0.0

    # Ratios are expected to be between zero and one.
    for feature in ("missing_ratio", "unique_ratio"):
        numeric[feature] = min(1.0, max(0.0, numeric[feature]))

    return {
        "text": " | ".join(text_parts),
        "column_name": name,
        "inferred_type": inferred_type or "unknown",
        "profile_role_hint": role_hint or "unknown",
        **numeric,
    }


def build_feature_frame(
    columns: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build one feature row per profiled column."""
    if not columns:
        return pd.DataFrame(
            columns=[
                "text",
                "column_name",
                *CATEGORICAL_FEATURES,
                *NUMERIC_FEATURES,
            ]
        )

    rows = [build_column_features(column) for column in columns]
    frame = pd.DataFrame(rows)

    for feature in NUMERIC_FEATURES:
        frame[feature] = pd.to_numeric(
            frame[feature], errors="coerce"
        ).fillna(0.0)

    for feature in CATEGORICAL_FEATURES:
        frame[feature] = frame[feature].fillna("unknown").astype(str)

    frame["text"] = frame["text"].fillna("")
    frame["column_name"] = frame["column_name"].fillna("")

    return frame


def profile_to_json(profile: dict[str, Any]) -> str:
    """Serialize a profiler result for storage or debugging."""
    return json.dumps(profile, ensure_ascii=False, default=str)
