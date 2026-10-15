
import json

import joblib

from whitman.ml.features import build_feature_frame
from whitman.ml.train import train_classifier


def test_train_classifier_saves_artifact_and_metrics(tmp_path):
    dataset_path = tmp_path / "examples.jsonl"
    artifact_path = tmp_path / "artifacts" / "classifier.joblib"

    examples = []

    for i in range(8):
        examples.extend(
            [
                {
                    "source_id": f"catalog_{i}",
                    "label": "attribute",
                    "column": {
                        "name": "material",
                        "role": "candidate_attribute",
                        "inferred_type": "categorical",
                        "total_count": 100,
                        "non_empty_count": 95,
                        "missing_count": 5,
                        "missing_ratio": 0.05,
                        "unique_count": 4,
                        "unique_ratio": 4 / 95,
                        "sample_values": ["Cotton", "Linen", "Wool"],
                    },
                },
                {
                    "source_id": f"catalog_{i}",
                    "label": "commercial_field",
                    "column": {
                        "name": "selling_price",
                        "role": "candidate_attribute",
                        "inferred_type": "numeric",
                        "total_count": 100,
                        "non_empty_count": 100,
                        "missing_count": 0,
                        "missing_ratio": 0.0,
                        "unique_count": 60,
                        "unique_ratio": 0.6,
                        "sample_values": ["19.99", "29.99", "49.99"],
                    },
                },
            ]
        )

    dataset_path.write_text(
        "\n".join(json.dumps(item) for item in examples),
        encoding="utf-8",
    )

    result = train_classifier(
        dataset_path,
        artifact_path=artifact_path,
        test_size=0.25,
        random_state=42,
    )

    assert artifact_path.exists()
    assert artifact_path.with_suffix(".metrics.json").exists()
    assert result["train_examples"] > 0
    assert result["test_examples"] > 0
    assert result["train_examples"] + result["test_examples"] == len(examples)

    assert set(result["train_catalogs"]).isdisjoint(
        result["test_catalogs"]
    )

    artifact = joblib.load(artifact_path)

    assert "model" in artifact
    assert set(artifact["labels"]) == {
        "attribute",
        "commercial_field",
    }

    predictions = artifact["model"].predict(
        build_feature_frame(
            [
                {
                    "name": "material",
                    "role": "candidate_attribute",
                    "inferred_type": "categorical",
                    "sample_values": ["Cotton", "Linen"],
                }
            ]
        )
    )

    assert len(predictions) == 1
