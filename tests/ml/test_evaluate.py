
import numpy as np
import pandas as pd
import pytest

from whitman.ml import evaluate


class FakeClassifier:
    def fit(self, X, y):
        self.classes_ = np.array(sorted(y.unique()))
        return self

    def predict_proba(self, X):
        probabilities = np.tile(
            [0.9, 0.1],
            (len(X), 1),
        )
        return probabilities

    def predict(self, X):
        return np.repeat(self.classes_[0], len(X))


def test_evaluation_keeps_catalogs_separate(monkeypatch, tmp_path):
    examples = [{"unused": True}]
    monkeypatch.setattr(
        evaluate,
        "load_labeled_examples",
        lambda _path: examples,
    )

    X = pd.DataFrame(
        {
            "column_name": [
                "name_a", "price_a",
                "name_b", "price_b",
                "name_c", "price_c",
                "name_d", "price_d",
            ],
            "inferred_type": [
                "text", "numeric",
                "text", "numeric",
                "text", "numeric",
                "text", "numeric",
            ],
        }
    )
    y = pd.Series(
        [
            "product_name", "commercial_field",
            "product_name", "commercial_field",
            "product_name", "commercial_field",
            "product_name", "commercial_field",
        ]
    )
    groups = pd.Series(["a", "a", "b", "b", "c", "c", "d", "d"])

    monkeypatch.setattr(
        evaluate,
        "build_training_frame",
        lambda _examples: (X, y, groups),
    )
    monkeypatch.setattr(
        evaluate,
        "build_classifier",
        FakeClassifier,
    )

    result = evaluate.evaluate_classifier(
        "unused.jsonl",
        report_path=tmp_path / "report.json",
        random_state=42,
    )

    assert result["split"]["catalog_overlap"] is False
    assert not (
        set(result["split"]["training_catalogs"])
        & set(result["split"]["test_catalogs"])
    )
    assert len(result["confidence_analysis"]) == 4
    assert (tmp_path / "report.json").exists()


def test_evaluation_requires_multiple_catalogs(monkeypatch):
    monkeypatch.setattr(
        evaluate,
        "load_labeled_examples",
        lambda _path: [{"unused": True}],
    )

    X = pd.DataFrame({"column_name": ["a", "b"]})
    y = pd.Series(["attribute", "commercial_field"])
    groups = pd.Series(["one", "one"])

    monkeypatch.setattr(
        evaluate,
        "build_training_frame",
        lambda _examples: (X, y, groups),
    )

    with pytest.raises(ValueError, match="distinct source catalogs"):
        evaluate.evaluate_classifier("unused.jsonl")
