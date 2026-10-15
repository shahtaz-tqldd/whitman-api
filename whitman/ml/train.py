
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from whitman.ml.dataset import load_training_dataset
from whitman.ml.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)

DEFAULT_ARTIFACT = (
    Path(__file__).parent / "artifacts" / "column_role_classifier.joblib"
)


def build_classifier() -> Pipeline:
    """Build the initial CPU-friendly column-role classifier."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=20_000,
                    sublinear_tf=True,
                ),
                "text",
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
            (
                "numeric",
                StandardScaler(with_mean=False),
                list(NUMERIC_FEATURES),
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("features", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _find_valid_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    test_size: float,
    random_state: int,
) -> tuple[list[int], list[int]]:
    """
    Find a catalog-separated split with at least two training classes
    and at least one class represented in both train and test.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if groups.nunique() < 2:
        raise ValueError(
            "At least two distinct source catalogs are required."
        )

    splitter = GroupShuffleSplit(
        n_splits=min(100, max(20, groups.nunique() * 5)),
        test_size=test_size,
        random_state=random_state,
    )

    all_classes = set(y)
    if len(all_classes) < 2:
        raise ValueError(
            "Training requires at least two distinct target labels."
        )

    for train_idx, test_idx in splitter.split(X, y, groups):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])

        if train_groups & test_groups:
            continue

        train_classes = set(y.iloc[train_idx])
        test_classes = set(y.iloc[test_idx])

        if len(train_classes) < 2:
            continue

        if not train_classes.intersection(test_classes):
            continue

        return train_idx.tolist(), test_idx.tolist()

    raise ValueError(
        "Could not find a useful catalog-level train/test split. "
        "Add more independently labeled source catalogs, check label "
        "coverage, or adjust test_size. Do not split columns randomly "
        "just to make training succeed."
    )


def train_classifier(
    dataset_path: str | Path,
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Evaluate on held-out catalogs, then fit and save the final model.

    Returns a report containing holdout metrics and artifact location.
    """
    X, y, groups = load_training_dataset(dataset_path)

    train_idx, test_idx = _find_valid_split(
        X,
        y,
        groups,
        test_size=test_size,
        random_state=random_state,
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    evaluation_model = build_classifier()
    evaluation_model.fit(X_train, y_train)

    predictions = evaluation_model.predict(X_test)
    labels = sorted(set(y))

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_test,
                predictions,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": report,
        "train_examples": len(train_idx),
        "test_examples": len(test_idx),
        "train_catalogs": sorted(set(groups.iloc[train_idx])),
        "test_catalogs": sorted(set(groups.iloc[test_idx])),
        "labels": labels,
    }

    # Refit on all labeled examples for subsequent inference.
    final_model = build_classifier()
    final_model.fit(X, y)

    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    
    joblib.dump(
        {
            "model": final_model,
            "labels": labels,
            "feature_version": 1,
            "training_examples": int(len(y)),
            "source_catalog_count": int(groups.nunique()),
            "class_counts": {
                str(label): int(count)
                for label, count in y.value_counts().items()
            },
        },
        artifact_path,
    )


    metrics["artifact_path"] = str(artifact_path)

    report_path = artifact_path.with_suffix(".metrics.json")
    report_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics["report_path"] = str(report_path)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the local product column-role classifier."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the labeled JSONL dataset.",
    )
    parser.add_argument(
        "--artifact",
        default=str(DEFAULT_ARTIFACT),
        help="Path where the trained classifier will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of source catalogs reserved for evaluation.",
    )
    args = parser.parse_args()

    result = train_classifier(
        args.dataset,
        artifact_path=args.artifact,
        test_size=args.test_size,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
