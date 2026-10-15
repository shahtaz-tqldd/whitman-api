
import json

import pytest

from whitman.ml.predict import (
    predict_column_role,
    predict_column_roles,
)
from whitman.ml.train import train_classifier


def make_example(source_id, name, label, value_type, samples):
    return {
        "source_id": source_id,
        "label": label,
        "column": {
            "name": name,
            "role": "candidate_attribute",
            "inferred_type": value_type,
            "total_count": 100,
            "non_empty_count": 100,
            "missing_count": 0,
            "missing_ratio": 0.0,
            "unique_count": len(samples),
            "unique_ratio": len(samples) / 100,
            "sample_values": samples,
        },
    }


@pytest.fixture
def trained_artifact(tmp_path):
    dataset_path = tmp_path / "examples.jsonl"
    artifact_path = tmp_path / "classifier.joblib"

    examples = []

    for i in range(8):
        examples.append(
            make_example(
                f"catalog_{i}",
                "material",
                "attribute",
                "categorical",
                ["Cotton", "Linen", "Wool"],
            )
        )
        examples.append(
            make_example(
                f"catalog_{i}",
                "selling_price",
                "commercial_field",
                "numeric",
                ["19.99", "29.99", "49.99"],
            )
        )

    dataset_path.write_text(
        "\n".join(json.dumps(x) for x in examples),
        encoding="utf-8",
    )

    train_classifier(
        dataset_path,
        artifact_path=artifact_path,
        test_size=0.25,
    )

    return artifact_path


def test_predict_single_column(trained_artifact):
    result = predict_column_role(
        {
            "name": "material",
            "role": "candidate_attribute",
            "inferred_type": "categorical",
            "total_count": 100,
            "non_empty_count": 100,
            "missing_count": 0,
            "missing_ratio": 0.0,
            "unique_count": 3,
            "unique_ratio": 0.03,
            "sample_values": ["Cotton", "Linen", "Wool"],
        },
        artifact_path=trained_artifact,
    )

    assert result["column_name"] == "material"
    assert result["predicted_role"] in {
        "attribute",
        "commercial_field",
    }
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["needs_review"], bool)
    assert isinstance(result["review_reasons"], list)


def test_predict_multiple_columns(trained_artifact):
    results = predict_column_roles(
        [
            {
                "name": "material",
                "inferred_type": "categorical",
                "sample_values": ["Cotton", "Linen"],
            },
            {
                "name": "selling_price",
                "inferred_type": "numeric",
                "sample_values": ["19.99", "29.99"],
            },
        ],
        artifact_path=trained_artifact,
    )

    assert len(results) == 2


def test_empty_prediction_does_not_load_model():
    assert predict_column_roles([]) == []


def test_invalid_confidence_threshold():
    with pytest.raises(ValueError, match="min_confidence"):
        predict_column_roles([], min_confidence=1.5)
