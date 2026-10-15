import csv
import io
import json
import re
from typing import Any

import pandas as pd

CATEGORY_KEYS = ("category", "product_category", "product_type", "item_type", "type")

ATTRIBUTE_PLURALS = {
    "color": "colors",
    "colour": "colors",
    "size": "sizes",
    "material": "materials",
    "brand": "brands",
    "gender": "genders",
    "style": "styles",
    "pattern": "patterns",
    "fit": "fits",
    "sleeve": "sleeves",
    "neckline": "necklines",
    "occasion": "occasions",
}

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}


def normalize_key(key: str) -> str:
    return re.sub(r"[\s\-]+", "_", key.strip().lower())


def normalize_value(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def parse_upload(filename: str, data: bytes) -> list[dict[str, Any]]:
    """Parse an uploaded csv/xlsx/json product file into a list of row dicts."""
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{extension}'. Use csv, xlsx or json.")

    if extension == ".csv":
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        return [{normalize_key(k): v for k, v in row.items() if k} for row in reader]

    if extension in {".xlsx", ".xls"}:
        frame = pd.read_excel(io.BytesIO(data))
        return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")

    payload = json.loads(data.decode("utf-8"))
    if isinstance(payload, dict):
        for key in ("products", "items", "data", "records"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("JSON payload must be a list of product objects.")
    return [{normalize_key(k): v for k, v in row.items() if k} for row in payload]


def build_catalog(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn unstructured product rows into a structured catalog.

    Placeholder for the ML stage: groups rows by category and aggregates the
    unique values of recognized attribute columns per category.
    """
    categories: dict[str, dict[str, set[str]]] = {}
    total_rows = 0

    for row in rows:
        total_rows += 1
        category = next(
            (
                value
                for key in CATEGORY_KEYS
                if (value := normalize_value(row.get(key)))
            ),
            "uncategorized",
        )
        category = category.lower()
        attributes = categories.setdefault(category, {})
        for key in ATTRIBUTE_PLURALS:
            value = normalize_value(row.get(key))
            if value:
                attributes.setdefault(ATTRIBUTE_PLURALS[key], set()).add(value.lower())

    catalog = [
        {"category": category, **{key: sorted(values) for key, values in sorted(attrs.items())}}
        for category, attrs in sorted(categories.items())
    ]
    return {"total_products": total_rows, "categories": catalog}
