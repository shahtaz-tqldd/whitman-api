from typing import Any

from whitman.discovery import discover_schema
from whitman.profiler import profile_products


def build_catalog(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the full catalog pipeline: profile products, then discover the schema."""
    profile = profile_products(rows)
    return discover_schema(rows, profile)
