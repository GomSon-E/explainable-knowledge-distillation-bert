import csv
import json

import pytest

from explainable_kd.common.metrics import classification_metrics, save_metrics


def test_classification_metrics_returns_required_macro_scores():
    result = classification_metrics(
        y_true=[0, 0, 1, 1, 2, 2],
        y_pred=[0, 1, 1, 1, 2, 0],
    )

    assert result == pytest.approx(
        {
            "accuracy": 4 / 6,
            "precision_macro": (0.5 + 2 / 3 + 1.0) / 3,
            "recall_macro": (0.5 + 1.0 + 0.5) / 3,
            "f1_macro": (0.5 + 0.8 + 2 / 3) / 3,
        }
    )


def test_save_metrics_writes_matching_json_and_csv(tmp_path):
    metrics = {"accuracy": 0.75, "f1_macro": 0.7}
    json_path = tmp_path / "nested/metrics.json"
    csv_path = tmp_path / "nested/metrics.csv"

    save_metrics(metrics, json_path, csv_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == metrics
    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert next(csv.DictReader(handle)) == {
            "accuracy": "0.75",
            "f1_macro": "0.7",
        }
