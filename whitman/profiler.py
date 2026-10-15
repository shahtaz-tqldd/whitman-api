from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd


def normalize_column_name(name: Any) -> str:
    """Convert a column name to a consistent identifier."""
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_scalar(value: Any) -> Any:
    """Convert pandas/numpy values into ordinary Python values."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass

    if isinstance(value, str):
        value = value.strip()
        return value or None

    return value


def infer_value_type(values: list[Any]) -> str:
    """Infer a preliminary type from observed non-empty values."""
    if not values:
        return "unknown"

    if all(isinstance(v, bool) for v in values):
        return "boolean"

    numeric_count = 0
    for value in values:
        if isinstance(value, bool):
            continue

        if isinstance(value, (int, float)):
            numeric_count += 1
        elif isinstance(value, str):
            candidate = value.replace(",", "").strip()
            try:
                float(candidate)
                numeric_count += 1
            except ValueError:
                pass

    if numeric_count / len(values) >= 0.9:
        return "numeric"

    unique_count = len({str(value).lower() for value in values})
    unique_ratio = unique_count / len(values)

    if unique_count <= 30 or unique_ratio <= 0.2:
        return "categorical"

    return "text"


def infer_column_role(name: str, value_type: str) -> str:
    """Suggest a role based on column naming, not a fixed attribute schema."""
    name = normalize_column_name(name)

    if re.search(r"(^|_)(id|sku|upc|ean|barcode)(_|$)", name):
        return "identifier"

    if re.search(r"(^|_)(category|product_type|item_type)(_|$)", name):
        return "category"

    if re.search(r"(^|_)(title|product_name|name)(_|$)", name):
        return "product_name"

    if re.search(r"(^|_)(description|details|summary)(_|$)", name):
        return "description"

    if value_type == "unknown":
        return "unknown"

    return "candidate_attribute"


def profile_products(
    rows: list[dict[str, Any]],
    *,
    sample_size: int = 8,
) -> dict[str, Any]:
    """Profile product rows to identify candidate fields and data quality."""
    if not rows:
        raise ValueError("No product records were provided.")

    normalized_rows: list[dict[str, Any]] = []

    for row in rows:
        normalized: dict[str, Any] = {}

        for raw_key, raw_value in row.items():
            if raw_key is None:
                continue

            key = normalize_column_name(raw_key)
            if not key:
                continue

            value = normalize_scalar(raw_value)

            # Do not silently overwrite two source columns that normalize
            # to the same name.
            if key in normalized:
                raise ValueError(
                    f"Duplicate column after normalization: {key!r}"
                )

            normalized[key] = value

        normalized_rows.append(normalized)

    column_names = list(
        dict.fromkeys(
            key
            for row in normalized_rows
            for key in row
        )
    )

    columns: list[dict[str, Any]] = []

    for name in column_names:
        values = [row.get(name) for row in normalized_rows]
        non_empty = [value for value in values if value is not None]

        frequencies = Counter(str(value) for value in non_empty)
        unique_count = len(frequencies)

        value_type = infer_value_type(non_empty)

        columns.append(
            {
                "name": name,
                "role": infer_column_role(name, value_type),
                "inferred_type": value_type,
                "total_count": len(values),
                "non_empty_count": len(non_empty),
                "missing_count": len(values) - len(non_empty),
                "missing_ratio": round(
                    (len(values) - len(non_empty)) / len(values), 4
                ),
                "unique_count": unique_count,
                "unique_ratio": round(
                    unique_count / len(non_empty), 4
                ) if non_empty else 0.0,
                "sample_values": list(frequencies.keys())[:sample_size],
                "frequent_values": [
                    {"value": value, "count": count}
                    for value, count in frequencies.most_common(
                        sample_size
                    )
                ],
            }
        )

    return {
        "total_products": len(normalized_rows),
        "total_columns": len(column_names),
        "columns": columns,
    }
