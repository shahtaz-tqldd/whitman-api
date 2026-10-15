import numpy as np
import pytest

from whitman import discovery
from whitman.discovery import discover_schema
from whitman.profiler import profile_products


def _discover(rows, monkeypatch):
    # Keep tests deterministic and independent of the local embedding model.
    monkeypatch.setattr(
        discovery, "embed_texts",
        lambda texts: np.eye(len(texts), dtype=np.float32),
    )
    return discover_schema(rows, profile_products(rows), alias_similarity_threshold=0.78)


def _category_map(result):
    return {category["name"]: category for category in result["categories"]}


def test_discovery_builds_category_subcategory_hierarchy(monkeypatch):
    rows = [
        {"category": "Women", "subcategory": "Tops", "color": "Blue",
         "price": 20, "stock_quantity": 4, "product_id": "p1"},
        {"category": "Women", "subcategory": "Tops", "color": "Red",
         "price": 25, "stock_quantity": 2, "product_id": "p2"},
        {"category": "Men", "subcategory": "Tops", "color": "Black",
         "price": 30, "stock_quantity": 5, "product_id": "p3"},
        {"category": "Men", "subcategory": "Bottoms", "color": "Blue",
         "price": 40, "stock_quantity": 1, "product_id": "p4"},
    ]
    result = _discover(rows, monkeypatch)
    categories = _category_map(result)

    assert result["total_products"] == 4
    assert result["category_count"] == 2
    # Tops occurs under both parent categories, so it forms two nested groups.
    assert result["subcategory_count"] == 3
    assert result["category_source_column"] == "category"
    assert result["subcategory_source_column"] == "subcategory"
    assert set(categories) == {"men", "women"}

    women_tops = next(
        item for item in categories["women"]["subcategories"]
        if item["name"] == "tops"
    )
    assert women_tops["product_count"] == 2
    assert women_tops["attributes"]["color"]["sample_values"] == ["Blue", "Red"]
    assert "product_id" not in women_tops["attributes"]
    assert "price" in women_tops["commercial_fields"]
    assert "stock_quantity" in women_tops["inventory_fields"]


def test_discovery_supports_flat_category_schema(monkeypatch):
    rows = [
        {"category": "Accessories", "color": "Black", "price": 10},
        {"category": "Accessories", "color": "Brown", "price": 12},
        {"category": "Footwear", "color": "White", "price": 30},
    ]
    result = _discover(rows, monkeypatch)
    categories = _category_map(result)

    assert result["category_count"] == 2
    assert result["subcategory_count"] == 0
    assert result["subcategory_source_column"] is None
    assert categories["accessories"]["attributes"]["color"]["sample_values"] == ["Black", "Brown"]
    assert "price" in categories["accessories"]["commercial_fields"]
    assert categories["accessories"]["subcategories"] == []


def test_discovery_recognizes_product_type_as_category(monkeypatch):
    rows = [
        {"product_type": "Shoes", "color": "Black"},
        {"product_type": "Shirts", "color": "White"},
    ]
    result = _discover(rows, monkeypatch)

    assert result["category_source_column"] == "product_type"
    assert result["category_count"] == 2
    assert {item["name"] for item in result["categories"]} == {"shirts", "shoes"}


def test_discovery_normalizes_hierarchy_values_and_missing_values(monkeypatch):
    rows = [
        {"category": "  Women  ", "subcategory": "Tops", "color": "Blue"},
        {"category": "Women", "subcategory": None, "color": None},
        {"category": None, "subcategory": "Accessories", "color": "Gold"},
    ]
    result = _discover(rows, monkeypatch)
    categories = _category_map(result)

    assert set(categories) == {"women", "uncategorized"}
    women_subcategories = {item["name"]: item for item in categories["women"]["subcategories"]}
    assert set(women_subcategories) == {"tops", "uncategorized"}
    assert women_subcategories["tops"]["attributes"]["color"]["sample_values"] == ["Blue"]


def test_discovery_does_not_emit_identifiers_as_attributes(monkeypatch):
    rows = [
        {"category": "Home", "product_id": "p1", "variant_id": "v1",
         "sku": "SKU-1", "material": "Wood"},
        {"category": "Home", "product_id": "p2", "variant_id": "v2",
         "sku": "SKU-2", "material": "Metal"},
    ]
    result = _discover(rows, monkeypatch)
    home = _category_map(result)["home"]

    assert set(home["attributes"]) == {"material"}


def test_discovery_classifies_commercial_and_inventory_fields(monkeypatch):
    rows = [
        {"category": "Food", "price": 2.5, "discount": 0.1, "stock_quantity": 10},
        {"category": "Food", "price": 3.0, "discount": 0.0, "stock_quantity": 5},
    ]
    result = _discover(rows, monkeypatch)
    food = _category_map(result)["food"]

    assert {"price", "discount"} <= set(food["commercial_fields"])
    assert "stock_quantity" in food["inventory_fields"]
    assert "price" not in food["attributes"]
    assert "stock_quantity" not in food["attributes"]


def test_discovery_rejects_empty_rows():
    with pytest.raises(ValueError, match="empty product data"):
        discover_schema([], {"columns": [{"name": "category", "role": "category"}]})


def test_discovery_rejects_profile_without_columns():
    with pytest.raises(ValueError, match="does not contain any columns"):
        discover_schema([{"category": "Home"}], {"columns": []})


def test_alias_suggestions_are_review_only(monkeypatch):
    rows = [
        {"category": "Home", "colour": "Blue", "color": "Blue"},
        {"category": "Home", "colour": "Red", "color": "Red"},
    ]
    monkeypatch.setattr(
        discovery, "embed_texts",
        lambda texts: np.ones((len(texts), 3), dtype=np.float32),
    )
    result = discover_schema(rows, profile_products(rows), alias_similarity_threshold=0.78)

    assert len(result["alias_suggestions"]) == 1
    suggestion = result["alias_suggestions"][0]
    assert {suggestion["attribute_a"], suggestion["attribute_b"]} == {"color", "colour"}
    assert suggestion["requires_review"] is True