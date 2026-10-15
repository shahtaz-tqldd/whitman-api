
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from whitman.ml.features import build_feature_frame
from whitman.ml.train import DEFAULT_ARTIFACT

MIN_CONFIDENCE = 0.60
MIN_TRAINING_EXAMPLES = 50
MIN_SOURCE_CATALOGS = 5


def load_classifier(
    artifact_path: str | Path = DEFAULT_ARTIFACT,
) -> dict[str, Any]:
    """Load a previously trained local classifier artifact."""
    path = Path(artifact_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Classifier artifact not found: {path}. "
            "Train the classifier before running predictions."
        )

    artifact = joblib.load(path)

    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(
            f"Invalid classifier artifact: {path}"
        )

    return artifact


def _model_maturity_reasons(
    artifact: dict[str, Any],
) -> list[str]:
    """Flag artifacts whose training coverage is insufficient."""
    reasons: list[str] = []

    training_examples = artifact.get("training_examples")
    source_catalogs = artifact.get("source_catalog_count")
    class_counts = artifact.get("class_counts")

    if training_examples is None:
        reasons.append("training_metadata_missing")
    elif training_examples < MIN_TRAINING_EXAMPLES:
        reasons.append("insufficient_training_examples")

    if source_catalogs is None:
        reasons.append("source_catalog_metadata_missing")
    elif source_catalogs < MIN_SOURCE_CATALOGS:
        reasons.append("insufficient_source_catalogs")

    if not isinstance(class_counts, dict):
        reasons.append("class_coverage_unknown")
    else:
        sparse_classes = [
            label
            for label, count in class_counts.items()
            if count < 5
        ]
        if sparse_classes:
            reasons.append("sparse_training_classes")

    return reasons


def predict_column_roles(
    columns: list[dict[str, Any]],
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    min_confidence: float = MIN_CONFIDENCE,
) -> list[dict[str, Any]]:
    """
    Predict a semantic role for each profiled column.

    `confidence` is the classifier's maximum predicted probability,
    not a calibrated probability of correctness.
    """
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1.")

    if not columns:
        return []

    artifact = load_classifier(artifact_path)
    model = artifact["model"]

    features = build_feature_frame(columns)

    probabilities = model.predict_proba(features)
    classes = list(model.classes_)

    maturity_reasons = _model_maturity_reasons(artifact)
    results: list[dict[str, Any]] = []

    for column, row_probabilities in zip(
        columns, probabilities, strict=True
    ):
        ranked_indices = sorted(
            range(len(classes)),
            key=lambda i: float(row_probabilities[i]),
            reverse=True,
        )

        best_index = ranked_indices[0]
        predicted_role = str(classes[best_index])
        confidence = float(row_probabilities[best_index])

        reasons = list(maturity_reasons)

        if confidence < min_confidence:
            reasons.append("low_prediction_confidence")

        results.append(
            {
                "column_name": str(column.get("name", "")),
                "predicted_role": predicted_role,
                "confidence": round(confidence, 4),
                "alternatives": [
                    {
                        "role": str(classes[index]),
                        "probability": round(
                            float(row_probabilities[index]), 4
                        ),
                    }
                    for index in ranked_indices[1:4]
                ],
                "needs_review": bool(reasons),
                "review_reasons": reasons,
            }
        )

    return results


def predict_column_role(
    column: dict[str, Any],
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    min_confidence: float = MIN_CONFIDENCE,
) -> dict[str, Any]:
    """Predict the role of a single profiled column."""
    return predict_column_roles(
        [column],
        artifact_path=artifact_path,
        min_confidence=min_confidence,
    )[0]
