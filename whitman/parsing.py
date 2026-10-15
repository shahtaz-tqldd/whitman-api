import csv
import io
import json
import re
from typing import Any

import pandas as pd

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
