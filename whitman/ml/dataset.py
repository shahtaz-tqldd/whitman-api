
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from whitman.ml.features import (
    build_column_features,
)

VALID_LABELS = {
    "category",
    "subcategory",
    "product_name",
    "description",
    "identifier",
    "attribute",
    "commercial_field",
    "inventory_field",
    "unknown",
}


class DatasetValidationError(ValueError):
    """Raised when a labeled training example is invalid."""


def validate_example(
    example: dict[str, Any],
    *,
    line_number: int,
) -> None:
    """Validate a single JSONL training example."""
    source_id = example.get("source_id")
    label = example.get("label")
    column = example.get("column")

    if not isinstance(source_id, str) or not source_id.strip():
        raise DatasetValidationError(
            f"Line {line_number}: source_id must be a non-empty string."
        )

    if not isinstance(label, str):
        raise DatasetValidationError(
            f"Line {line_number}: label must be a string."
        )

    label = label.strip().lower()

    if label not in VALID_LABELS:
        raise DatasetValidationError(
            f"Line {line_number}: unsupported label {label!r}. "
            f"Expected one of: {sorted(VALID_LABELS)}"
        )

    if not isinstance(column, dict):
        raise DatasetValidationError(
            f"Line {line_number}: column must be an object."
        )

    if not str(column.get("name", "")).strip():
        raise DatasetValidationError(
            f"Line {line_number}: column.name is required."
        )


def load_labeled_examples(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Load and validate labeled column examples from JSONL."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {path}"
        )

    examples: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue

            try:
                example = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"Line {line_number}: invalid JSON."
                ) from exc

            if not isinstance(example, dict):
                raise DatasetValidationError(
                    f"Line {line_number}: each record must be an object."
                )

            validate_example(example, line_number=line_number)

            # Normalize labels and source identifiers.
            example["source_id"] = example["source_id"].strip()
            example["label"] = example["label"].strip().lower()
            examples.append(example)

    if not examples:
        raise DatasetValidationError(
            "The labeled dataset contains no examples."
        )

    return examples


def build_training_frame(
    examples: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Return X, y, groups.

    X: features for the classifier
    y: ground-truth semantic column roles
    groups: source catalog IDs for leakage-safe splitting
    """
    if not examples:
        raise DatasetValidationError(
            "Cannot build a dataset from zero examples."
        )

    feature_rows: list[dict[str, Any]] = []
    labels: list[str] = []
    groups: list[str] = []

    for index, example in enumerate(examples, start=1):
        validate_example(example, line_number=index)

        feature_rows.append(
            build_column_features(example["column"])
        )
        labels.append(example["label"].strip().lower())
        groups.append(example["source_id"].strip())

    X = pd.DataFrame(feature_rows)
    y = pd.Series(labels, name="label")
    group_ids = pd.Series(groups, name="source_id")

    return X, y, group_ids


def load_training_dataset(
    path: str | Path,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load JSONL and build the feature matrix, labels, and groups."""
    examples = load_labeled_examples(path)
    return build_training_frame(examples)


def split_by_source(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split by source catalog rather than by individual column.

    Requires scikit-learn. Raises an error if there aren't enough
    independent source catalogs to create a meaningful split.
    """
    from sklearn.model_selection import GroupShuffleSplit

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not (len(X) == len(y) == len(groups)):
        raise ValueError(
            "X, y, and groups must have the same number of rows."
        )

    if groups.nunique() < 2:
        raise ValueError(
            "At least two distinct source catalogs are required "
            "for a source-level train/test split."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])

    if train_groups & test_groups:
        raise RuntimeError(
            "Source leakage detected between train and test sets."
        )

    return (
        X.iloc[train_indices].reset_index(drop=True),
        X.iloc[test_indices].reset_index(drop=True),
        y.iloc[train_indices].reset_index(drop=True),
        y.iloc[test_indices].reset_index(drop=True),
    )
