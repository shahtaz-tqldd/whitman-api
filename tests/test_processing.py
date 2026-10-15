import json

import pytest

from app.modules.processing.services.base import build_catalog, parse_upload


def test_parse_csv() -> None:
    data = b"Category,Color,Size\nShirts,Red,XL\nShirts,Blue,LG\nPants,Black,M\n"
    rows = parse_upload("products.csv", data)
    assert rows == [
        {"category": "Shirts", "color": "Red", "size": "XL"},
        {"category": "Shirts", "color": "Blue", "size": "LG"},
        {"category": "Pants", "color": "Black", "size": "M"},
    ]


def test_parse_json_list() -> None:
    data = json.dumps(
        [{"product_type": "shirt", "colour": "red", "size": "xl"}]
    ).encode()
    rows = parse_upload("products.json", data)
    assert rows == [{"product_type": "shirt", "colour": "red", "size": "xl"}]


def test_parse_json_wrapped_dict() -> None:
    data = json.dumps({"products": [{"category": "shirt"}]}).encode()
    rows = parse_upload("products.json", data)
    assert rows == [{"category": "shirt"}]


def test_parse_unsupported_extension() -> None:
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_upload("products.txt", b"data")


def test_build_catalog_groups_and_aggregates() -> None:
    rows = [
        {"category": "Shirts", "color": "Red", "size": "XL"},
        {"category": "shirts", "color": "Blue", "size": "LG"},
        {"category": "Shirts", "color": "red", "size": "XL"},
        {"category": "Pants", "color": "Black", "size": "M"},
    ]
    catalog = build_catalog(rows)
    assert catalog["total_products"] == 4
    assert catalog["categories"] == [
        {"category": "pants", "colors": ["black"], "sizes": ["m"]},
        {"category": "shirts", "colors": ["blue", "red"], "sizes": ["lg", "xl"]},
    ]


def test_build_catalog_defaults_to_uncategorized() -> None:
    catalog = build_catalog([{"name": "mystery item"}])
    assert catalog == {"total_products": 1, "categories": [{"category": "uncategorized"}]}


def test_build_catalog_uses_category_aliases() -> None:
    rows = [{"product_type": "shirt"}, {"item_type": "pants"}, {"type": "socks"}]
    categories = [entry["category"] for entry in build_catalog(rows)["categories"]]
    assert categories == ["pants", "shirt", "socks"]


def test_build_catalog_empty_input() -> None:
    assert build_catalog([]) == {"total_products": 0, "categories": []}
