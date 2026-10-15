from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from whitman.embeddings import embed_texts

CATEGORY_CANDIDATES = (
    "category",
    "product_category",
    "product_type",
    "item_type",
)

SUBCATEGORY_CANDIDATES = (
    "subcategory",
    "sub_category",
    "product_subcategory",
    "product_sub_category",
    "item_subcategory",
    "item_sub_category",
)

IDENTIFIER_NAMES = {
    "id",
    "product_id",
    "item_id",
    "sku",
    "barcode",
    "gtin",
    "ean",
    "upc",
    "uuid",
}

COMMERCIAL_NAMES = {
    "price",
    "cost",
    "amount",
    "msrp",
    "sale_price",
    "selling_price",
    "list_price",
    "compare_at_price",
    "discount",
    "tax",
    "shipping_cost",
}

INVENTORY_NAMES = {
    "stock",
    "stock_quantity",
    "inventory",
    "inventory_quantity",
    "quantity_in_stock",
    "available_quantity",
    "available_stock",
    "reserved_quantity",
}

UNIT_PATTERNS = (
    ("mm", re.compile(r"\bmm\b", re.IGNORECASE)),
    ("cm", re.compile(r"\bcm\b", re.IGNORECASE)),
    ("m", re.compile(r"(?<![a-z])m\b", re.IGNORECASE)),
    ("in", re.compile(r"\b(in|inch|inches)\b", re.IGNORECASE)),
    ("ft", re.compile(r"\b(ft|feet)\b", re.IGNORECASE)),
    ("kg", re.compile(r"\bkg\b", re.IGNORECASE)),
    ("g", re.compile(r"(?<![a-z])g\b", re.IGNORECASE)),
    ("ml", re.compile(r"\bml\b", re.IGNORECASE)),
    ("l", re.compile(r"(?<![a-z])l\b", re.IGNORECASE)),
    ("gb", re.compile(r"\bgb\b", re.IGNORECASE)),
    ("tb", re.compile(r"\btb\b", re.IGNORECASE)),
    ("hz", re.compile(r"\bhz\b", re.IGNORECASE)),
    ("w", re.compile(r"(?<![a-z])w\b", re.IGNORECASE)),
    ("v", re.compile(r"(?<![a-z])v\b", re.IGNORECASE)),
)


def _normalize_name(name: str) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    return name.strip("_").lower()


def _normalized_profile_columns(
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Normalize profile column names to match normalized input row keys."""
    columns: list[dict[str, Any]] = []

    for column in profile.get("columns", []):
        if not column.get("name"):
            continue

        normalized = dict(column)
        normalized["name"] = _normalize_name(str(column["name"]))

        if normalized["name"]:
            columns.append(normalized)

    return columns


def _hierarchy_columns(
    profile_columns: list[dict[str, Any]],
) -> tuple[str | None, str | None]:
    """Return primary category and optional subcategory source columns."""
    names = {column["name"] for column in profile_columns}

    # Prefer explicit category columns, then product-type columns.
    category_column = next(
        (name for name in CATEGORY_CANDIDATES if name in names),
        None,
    )

    if category_column is None:
        category_column = next(
            (
                column["name"]
                for column in profile_columns
                if column.get("role") == "category"
            ),
            None,
        )

    subcategory_column = next(
        (name for name in SUBCATEGORY_CANDIDATES if name in names),
        None,
    )

    # Never use the same source column for both hierarchy levels.
    if subcategory_column == category_column:
        subcategory_column = None

    return category_column, subcategory_column


def _infer_unit(values: list[Any]) -> str | None:
    """Return a unit only when observed text provides clear evidence."""
    text_values = [
        str(value).strip()
        for value in values
        if value is not None
    ]

    found_units: Counter[str] = Counter()

    for value in text_values:
        for unit, pattern in UNIT_PATTERNS:
            if pattern.search(value):
                found_units[unit] += 1
                break

    if not found_units:
        return None

    unit, count = found_units.most_common(1)[0]

    if count / len(text_values) >= 0.5:
        return unit

    return None


def _values_for_column(
    rows: list[dict[str, Any]],
    column_name: str,
) -> list[Any]:
    return [
        row.get(column_name)
        for row in rows
        if row.get(column_name) is not None
        and str(row[column_name]).strip()
    ]


def _category_name(value: Any) -> str:
    if value is None or not str(value).strip():
        return "uncategorized"

    return re.sub(r"\s+", " ", str(value).strip()).lower()


def _field_bucket(
    name: str,
    role: str,
    predicted_role: str | None = None,
) -> str | None:
    """Classify fields using deterministic rules before optional ML hints."""
    tokens = set(name.split("_"))

    # Deterministic rules always take priority.
    if role == "identifier" or name in IDENTIFIER_NAMES:
        return None

    if (
        name.endswith("_id")
        or name.endswith("_sku")
        or name.endswith("_barcode")
    ):
        return None

    if name in INVENTORY_NAMES or (
        tokens & {"stock", "inventory"}
        or {"available", "quantity"}.issubset(tokens)
    ):
        return "inventory_fields"

    if name in COMMERCIAL_NAMES or (
        tokens & {"price", "cost", "msrp", "discount", "tax"}
    ):
        return "commercial_fields"

    # Do not let ML turn hierarchy or descriptive fields into attributes.
    if role in {"category", "product_name", "description", "unknown"}:
        return None

    # Only recognized ML labels can influence the bucket.
    ml_buckets = {
        "attribute": "attributes",
        "commercial_field": "commercial_fields",
        "inventory_field": "inventory_fields",
    }
    if predicted_role in ml_buckets:
        return ml_buckets[predicted_role]

    # Preserve existing behavior when ML has no usable prediction.
    if role == "candidate_attribute":
        return "attributes"

    return None


def _build_field_schema(
    rows: list[dict[str, Any]],
    profile_by_name: dict[str, dict[str, Any]],
    field_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Build attribute metadata using only the supplied group's rows."""
    fields: dict[str, dict[str, Any]] = {}

    for name in field_names:
        values = _values_for_column(rows, name)

        if not values:
            continue

        unique_values = sorted(
            {str(value).strip() for value in values}
        )

        profile_column = profile_by_name.get(name, {})
        inferred_type = profile_column.get("inferred_type", "unknown")
        unit = _infer_unit(values)

        if unit:
            field_type = "measurement"
        elif inferred_type == "numeric":
            field_type = "numeric"
        elif inferred_type == "boolean":
            field_type = "boolean"
        elif inferred_type == "categorical":
            field_type = "categorical"
        else:
            field_type = "text"

        fields[name] = {
            "type": field_type,
            "unit": unit,
            "product_count_with_value": len(values),
            "unique_value_count": len(unique_values),
            "sample_values": unique_values[:20],
            "source": "observed_input_column",
            "status": "proposed",
        }

    return fields


def _build_alias_suggestions(
    attribute_names: list[str],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Suggest similar field names without automatically merging them."""
    names = sorted(set(attribute_names))

    if len(names) < 2:
        return []

    vectors = np.asarray(embed_texts(names), dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[0] != len(names):
        raise ValueError("Embedding model returned an invalid matrix.")

    # embed_texts() normalizes vectors, so dot product gives cosine similarity.
    similarities = vectors @ vectors.T

    suggestions: list[dict[str, Any]] = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            score = float(similarities[i, j])

            if score >= threshold:
                suggestions.append(
                    {
                        "attribute_a": names[i],
                        "attribute_b": names[j],
                        "similarity": round(score, 4),
                        "requires_review": True,
                    }
                )

    return sorted(
        suggestions,
        key=lambda item: item["similarity"],
        reverse=True,
    )



def discover_schema(
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    *,
    alias_similarity_threshold: float = 0.78,
    use_ml: bool = False,
    ml_artifact_path: str | None = None,
) -> dict[str, Any]:

    """Discover a hierarchical product schema from profile data and embeddings."""
    if not rows:
        raise ValueError("Cannot discover a schema from empty product data.")

    profile_columns = _normalized_profile_columns(profile)

    if not profile_columns:
        raise ValueError("Profile does not contain any columns.")

    category_column, subcategory_column = _hierarchy_columns(
        profile_columns
    )

    profile_by_name = {
        column["name"]: column
        for column in profile_columns
    }

    column_roles = {
        column["name"]: column.get("role", "unknown")
        for column in profile_columns
    }


    ml_predictions_by_name: dict[str, dict[str, Any]] = {}
    ml_diagnostics: dict[str, Any] = {
        "enabled": use_ml,
        "status": "disabled",
        "predictions": [],
    }

    if use_ml:
        try:
            from whitman.ml.predict import (
                predict_column_roles,
            )

            prediction_kwargs = {}
            if ml_artifact_path is not None:
                prediction_kwargs["artifact_path"] = ml_artifact_path

            predictions = predict_column_roles(
                profile_columns,
                **prediction_kwargs,
            )

            ml_diagnostics["status"] = "available"
            ml_diagnostics["predictions"] = predictions

            for prediction in predictions:
                name = _normalize_name(
                    str(prediction.get("column_name", ""))
                )
                if name:
                    ml_predictions_by_name[name] = prediction

        except Exception as exc:
            # Discovery must remain usable if the optional model fails.
            ml_diagnostics["status"] = "unavailable"
            ml_diagnostics["error"] = (
                f"{type(exc).__name__}: {exc}"
            )


    # Normalize input keys consistently with the profiler.
    normalized_rows: list[dict[str, Any]] = []

    for row in rows:
        normalized_rows.append(
            {
                _normalize_name(str(key)): value
                for key, value in row.items()
                if key is not None and _normalize_name(str(key))
            }
        )

    # Hierarchy fields should never be emitted as ordinary attributes.
    hierarchy_names = {
        name
        for name in (category_column, subcategory_column)
        if name is not None
    }

    # Determine the destination bucket for every non-hierarchy field.
    fields_by_bucket: dict[str, list[str]] = {
        "attributes": [],
        "commercial_fields": [],
        "inventory_fields": [],
    }

    
    for name, role in column_roles.items():
        if name in hierarchy_names:
            continue

        prediction = ml_predictions_by_name.get(name)
        predicted_role = None

        if prediction is not None:
            confidence = float(prediction.get("confidence", 0.0))
            needs_review = prediction.get("needs_review", True)

            # Only apply predictions from a sufficiently mature,
            # confident model. Otherwise retain deterministic behavior.
            if not needs_review and confidence >= 0.60:
                predicted_role = prediction.get("predicted_role")

        bucket = _field_bucket(
            name,
            role,
            predicted_role=predicted_role,
        )

        if bucket:
            fields_by_bucket[bucket].append(name)


    for bucket in fields_by_bucket:
        fields_by_bucket[bucket].sort()

    all_field_names = sorted(
        {
            name
            for names in fields_by_bucket.values()
            for name in names
        }
    )

    # First group by primary category.
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in normalized_rows:
        category = (
            _category_name(row.get(category_column))
            if category_column
            else "uncategorized"
        )
        grouped_rows[category].append(row)

    categories: list[dict[str, Any]] = []
    subcategory_count = 0

    for category_name, category_rows in sorted(grouped_rows.items()):
        category_schema: dict[str, Any] = {
            "name": category_name,
            "product_count": len(category_rows),
            "attributes": {},
            "commercial_fields": {},
            "inventory_fields": {},
            "subcategories": [],
        }

        if subcategory_column:
            # Calculate each child schema from its own products.
            grouped_subcategories: dict[
                str, list[dict[str, Any]]
            ] = defaultdict(list)

            for row in category_rows:
                subcategory_name = _category_name(
                    row.get(subcategory_column)
                )
                grouped_subcategories[subcategory_name].append(row)

            for subcategory_name, subcategory_rows in sorted(
                grouped_subcategories.items()
            ):
                subcategory_schema = {
                    "name": subcategory_name,
                    "product_count": len(subcategory_rows),
                    "attributes": _build_field_schema(
                        subcategory_rows,
                        profile_by_name,
                        fields_by_bucket["attributes"],
                    ),
                    "commercial_fields": _build_field_schema(
                        subcategory_rows,
                        profile_by_name,
                        fields_by_bucket["commercial_fields"],
                    ),
                    "inventory_fields": _build_field_schema(
                        subcategory_rows,
                        profile_by_name,
                        fields_by_bucket["inventory_fields"],
                    ),
                }

                category_schema["subcategories"].append(
                    subcategory_schema
                )
                subcategory_count += 1
        else:
            # Datasets without a subcategory column still work as before.
            category_schema["attributes"] = _build_field_schema(
                category_rows,
                profile_by_name,
                fields_by_bucket["attributes"],
            )
            category_schema["commercial_fields"] = _build_field_schema(
                category_rows,
                profile_by_name,
                fields_by_bucket["commercial_fields"],
            )
            category_schema["inventory_fields"] = _build_field_schema(
                category_rows,
                profile_by_name,
                fields_by_bucket["inventory_fields"],
            )

        categories.append(category_schema)

    alias_suggestions = _build_alias_suggestions(
        all_field_names,
        threshold=alias_similarity_threshold,
    )

    return {
        "total_products": len(normalized_rows),
        "category_count": len(categories),
        "subcategory_count": subcategory_count,
        "category_source_column": category_column,
        "subcategory_source_column": subcategory_column,
        "categories": categories,
        "alias_suggestions": alias_suggestions,
        "discovery_method": "profile_plus_local_embeddings",
        "ml_role_discovery": ml_diagnostics,
    }
