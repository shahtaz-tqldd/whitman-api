import pytest

from whitman.profiler import (
    infer_column_role,
    infer_value_type,
    normalize_column_name,
    normalize_scalar,
    profile_products,
)


def test_normalize_column_name():
    assert normalize_column_name(" Product Name ") == "product_name"
    assert normalize_column_name("SKU / ID") == "sku_id"


def test_normalize_scalar_handles_missing_and_strings():
    assert normalize_scalar(None) is None
    assert normalize_scalar("  Blue  ") == "Blue"
    assert normalize_scalar("   ") is None


def test_infer_value_type():
    assert infer_value_type([]) == "unknown"
    assert infer_value_type([True, False]) == "boolean"
    assert infer_value_type([10, 20.5, "30"]) == "numeric"
    assert infer_value_type(["Men", "Women", "Men"]) == "categorical"
    assert infer_value_type(["long descriptive text " + str(i) for i in range(40)]) == "text"


def test_infer_column_role():
    assert infer_column_role("product_id", "numeric") == "identifier"
    assert infer_column_role("product_type", "categorical") == "category"
    assert infer_column_role("product_name", "text") == "product_name"
    assert infer_column_role("description", "text") == "description"
    assert infer_column_role("color", "categorical") == "candidate_attribute"


def test_profile_products_reports_counts_and_missing_values():
    rows = [
        {"Product Name": "Blue Shirt", "Color": "Blue", "Price": 20},
        {"Product Name": "Red Shirt", "Color": None, "Price": 25},
        {"Product Name": "Green Shirt", "Color": "Green", "Price": None},
    ]
    profile = profile_products(rows)
    columns = {column["name"]: column for column in profile["columns"]}

    assert profile["total_products"] == 3
    assert profile["total_columns"] == 3
    assert columns["product_name"]["non_empty_count"] == 3
    assert columns["color"]["missing_count"] == 1
    assert columns["color"]["missing_ratio"] == pytest.approx(1 / 3, abs=0.0001)
    assert columns["price"]["missing_count"] == 1


def test_profile_products_rejects_empty_rows():
    with pytest.raises(ValueError, match="No product records"):
        profile_products([])


def test_profile_products_rejects_duplicate_normalized_names():
    with pytest.raises(ValueError, match="Duplicate column after normalization"):
        profile_products([{"Product Name": "Shirt", "product-name": "Trouser"}])