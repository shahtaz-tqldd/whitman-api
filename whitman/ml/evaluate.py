
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit

from whitman.ml.dataset import (
    build_training_frame,
    load_labeled_examples,
)
from whitman.ml.train import build_classifier

DEFAULT_CONFIDENCE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80)


def _json_safe(value: Any) -> Any:
    """Convert NumPy/Pandas values into JSON-serializable Python values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value


def _select_catalog_split(
    y: pd.Series,
    groups: pd.Series,
    *,
    test_size: float,
    random_state: int,
    attempts: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Find a catalog-level split with usable training and test labels."""
    splitter = GroupShuffleSplit(
        n_splits=attempts,
        test_size=test_size,
        random_state=random_state,
    )

    y_values = np.asarray(y)
    group_values = np.asarray(groups)

    best_split = None
    best_score = -1

    for train_idx, test_idx in splitter.split(
        np.zeros(len(y_values)),
        y_values,
        groups=group_values,
    ):
        train_labels = set(y_values[train_idx])
        test_labels = set(y_values[test_idx])
        shared_labels = train_labels & test_labels

        # A classifier needs at least two training classes. The test set
        # needs at least one class represented during training.
        if len(train_labels) < 2 or not shared_labels:
            continue

        # Prefer splits that evaluate more examples and more known classes.
        score = len(test_idx) * 100 + len(shared_labels)

        if score > best_score:
            best_split = (train_idx, test_idx)
            best_score = score

    if best_split is None:
        raise ValueError(
            "Could not find a viable catalog-level split. Add more labeled "
            "catalogs and examples across the role classes."
        )

    return best_split


def _evaluate_confidence_threshold(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    labels: list[str],
    threshold: float,
) -> dict[str, Any]:
    """Evaluate accuracy and coverage for predictions above a threshold."""
    accepted = confidence >= threshold
    accepted_count = int(accepted.sum())
    total_count = len(y_true)

    result: dict[str, Any] = {
        "threshold": threshold,
        "total_examples": total_count,
        "accepted_count": accepted_count,
        "review_count": total_count - accepted_count,
        "coverage": (
            round(accepted_count / total_count, 4)
            if total_count
            else 0.0
        ),
        "review_rate": (
            round((total_count - accepted_count) / total_count, 4)
            if total_count
            else 0.0
        ),
    }

    if accepted_count:
        result.update(
            {
                "accuracy": round(
                    float(
                        accuracy_score(
                            y_true[accepted],
                            y_pred[accepted],
                        )
                    ),
                    4,
                ),
                "macro_f1": round(
                    float(
                        f1_score(
                            y_true[accepted],
                            y_pred[accepted],
                            labels=labels,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    4,
                ),
            }
        )
    else:
        result.update({"accuracy": None, "macro_f1": None})

    return result


def evaluate_classifier(
    dataset_path: str | Path,
    *,
    report_path: str | Path | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
    confidence_thresholds: Sequence[float] = (
        DEFAULT_CONFIDENCE_THRESHOLDS
    ),
) -> dict[str, Any]:
    """
    Evaluate role classification on catalogs excluded from training.

    The final production model is not loaded or modified. A fresh model is
    fitted only on the training catalogs for this evaluation.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    thresholds = sorted(set(float(t) for t in confidence_thresholds))
    if any(t < 0 or t > 1 for t in thresholds):
        raise ValueError("Confidence thresholds must be between 0 and 1.")

    examples = load_labeled_examples(dataset_path)
    X, y, groups = build_training_frame(examples)

    if len(X) < 2:
        raise ValueError("At least two labeled examples are required.")

    if groups.nunique() < 2:
        raise ValueError(
            "At least two distinct source catalogs are required for "
            "catalog-level evaluation."
        )

    train_idx, test_idx = _select_catalog_split(
        y,
        groups,
        test_size=test_size,
        random_state=random_state,
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    # Defensive check: no source catalog may leak across the split.
    if set(train_groups) & set(test_groups):
        raise RuntimeError("Catalog leakage detected in evaluation split.")

    model = build_classifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    confidence = probabilities.max(axis=1)

    labels = sorted(str(label) for label in y.unique())
    train_labels = set(str(label) for label in y_train.unique())
    test_labels = set(str(label) for label in y_test.unique())

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    result: dict[str, Any] = {
        "dataset": {
            "path": str(dataset_path),
            "total_examples": int(len(X)),
            "source_catalog_count": int(groups.nunique()),
            "class_counts": {
                str(label): int(count)
                for label, count in y.value_counts().items()
            },
        },
        "split": {
            "strategy": "GroupShuffleSplit by source catalog",
            "test_size_requested": test_size,
            "random_state": random_state,
            "training_examples": int(len(train_idx)),
            "test_examples": int(len(test_idx)),
            "training_catalog_count": int(train_groups.nunique()),
            "test_catalog_count": int(test_groups.nunique()),
            "training_catalogs": sorted(
                str(group) for group in train_groups.unique()
            ),
            "test_catalogs": sorted(
                str(group) for group in test_groups.unique()
            ),
            "catalog_overlap": False,
            "labels_not_seen_in_training": sorted(
                test_labels - train_labels
            ),
        },
        "overall": {
            "accuracy": round(
                float(accuracy_score(y_test, predictions)), 4
            ),
            "macro_f1": round(
                float(
                    f1_score(
                        y_test,
                        predictions,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                ),
                4,
            ),
            "weighted_f1": round(
                float(
                    f1_score(
                        y_test,
                        predictions,
                        labels=labels,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                4,
            ),
            "per_class": report,
            "confusion_matrix": {
                "labels": labels,
                "matrix": confusion_matrix(
                    y_test,
                    predictions,
                    labels=labels,
                ).tolist(),
            },
        },
        "confidence_analysis": [
            _evaluate_confidence_threshold(
                y_true=np.asarray(y_test),
                y_pred=np.asarray(predictions),
                confidence=confidence,
                labels=labels,
                threshold=threshold,
            )
            for threshold in thresholds
        ],
        "notes": [
            "Confidence is the classifier's maximum predicted probability; "
            "it is not necessarily calibrated.",
            "Threshold metrics describe selective prediction on this test "
            "split, not guaranteed production performance.",
            "Labels absent from training cannot be learned by this model.",
        ],
    }

    result = _json_safe(result)

    if report_path is not None:
        output_path = Path(report_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return result
