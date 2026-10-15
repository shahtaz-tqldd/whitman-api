import json

import pytest

from whitman.parsing import parse_upload


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


def test_parse_json_rejects_non_object_items() -> None:
    with pytest.raises(ValueError, match="list of product objects"):
        parse_upload("products.json", b'["shirt", "pants"]')


def test_parse_unsupported_extension() -> None:
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_upload("products.txt", b"data")
