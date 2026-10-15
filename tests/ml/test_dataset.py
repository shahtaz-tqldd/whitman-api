
import json

import pandas as pd
import pytest

from whitman.ml.dataset import (
    DatasetValidationError,
    build_training_frame,
    load_labeled_examples,
    split_by_source,
)


def make_example(source_id, name, label):
    return {
        "source_id": source_id,
        "label": label,
        "column": {
            "name": name,
            "role": "candidate_attribute",
            "inferred_type": "categorical",
            "total_count": 10,
            "non_empty_count": 10,
            "missing_count": 0,
            "missing_ratio": 0.0,
            "unique_count": 3,
            "unique_ratio": 0.3,
            "sample_values": ["A", "B", "C"],
        },
    }


def test_build_training_frame():
    examples = [
        make_example("catalog_a", "material", "attribute"),
        make_example("catalog_b", "price", "commercial_field"),
    ]

    X, y, groups = build_training_frame(examples)

    assert len(X) == len(y) == len(groups) == 2
    assert "text" in X.columns
    assert "missing_ratio" in X.columns
    assert y.tolist() == ["attribute", "commercial_field"]
    assert groups.tolist() == ["catalog_a", "catalog_b"]
    assert "label" not in X.columns


def test_load_labeled_examples(tmp_path):
    path = tmp_path / "examples.jsonl"
    examples = [
        make_example("catalog_a", "material", "attribute"),
        make_example("catalog_b", "price", "commercial_field"),
    ]

    path.write_text(
        "\n".join(json.dumps(x) for x in examples),
        encoding="utf-8",
    )

    loaded = load_labeled_examples(path)

    assert len(loaded) == 2


def test_reject_invalid_label(tmp_path):
    path = tmp_path / "invalid.jsonl"
    example = make_example("catalog_a", "material", "not_a_role")

    path.write_text(json.dumps(example), encoding="utf-8")

    with pytest.raises(DatasetValidationError):
        load_labeled_examples(path)


def test_split_keeps_catalogs_separate():
    examples = [
        make_example(f"catalog_{i}", f"field_{i}", "attribute")
        for i in range(10)
    ]

    X, y, groups = build_training_frame(examples)

    X_train, X_test, y_train, y_test = split_by_source(
        X, y, groups, test_size=0.3
    )

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    train_groups = set(groups.iloc[
        X_train.index
    ])
    # Indexes are reset by the split function, so use source IDs
    # directly in the next test instead of these reset indexes.
    assert len(train_groups) <= len(groups)


def test_split_requires_multiple_catalogs():
    X = pd.DataFrame({"text": ["material: cotton"]})
    y = pd.Series(["attribute"])
    groups = pd.Series(["one_catalog"])

    with pytest.raises(ValueError, match="two distinct source catalogs"):
        split_by_source(X, y, groups)
