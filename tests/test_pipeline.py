import numpy as np
import pytest

from whitman import discovery
from whitman.pipeline import build_catalog


def _catalog(rows, monkeypatch):
    # Keep tests deterministic and independent of the local embedding model.
    monkeypatch.setattr(
        discovery,
        "embed_texts",
        lambda texts: np.eye(len(texts), dtype=np.float32),
    )
    return build_catalog(rows)


def test_build_catalog_groups_rows_by_category(monkeypatch) -> None:
    rows = [
        {"category": "Shirts", "color": "Red", "size": "XL"},
        {"category": "shirts", "color": "Blue", "size": "LG"},
        {"category": "Shirts", "color": "red", "size": "XL"},
        {"category": "Pants", "color": "Black", "size": "M"},
    ]
    catalog = _catalog(rows, monkeypatch)

    assert catalog["total_products"] == 4
    assert catalog["category_count"] == 2
    assert [category["name"] for category in catalog["categories"]] == ["pants", "shirts"]

    shirts = catalog["categories"][1]
    assert shirts["product_count"] == 3
    assert shirts["attributes"]["color"]["sample_values"] == ["Blue", "Red", "red"]
    assert shirts["attributes"]["size"]["sample_values"] == ["LG", "XL"]


def test_build_catalog_defaults_to_uncategorized(monkeypatch) -> None:
    catalog = _catalog([{"name": "mystery item"}], monkeypatch)

    assert catalog["total_products"] == 1
    assert catalog["category_source_column"] is None
    assert [category["name"] for category in catalog["categories"]] == ["uncategorized"]


def test_build_catalog_uses_product_type_as_category(monkeypatch) -> None:
    rows = [{"product_type": "shirt"}, {"product_type": "pants"}]
    catalog = _catalog(rows, monkeypatch)

    assert catalog["category_source_column"] == "product_type"
    assert [category["name"] for category in catalog["categories"]] == ["pants", "shirt"]


def test_build_catalog_empty_input() -> None:
    with pytest.raises(ValueError, match="No product records"):
        build_catalog([])
