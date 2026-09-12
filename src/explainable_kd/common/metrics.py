"""Classification metrics and machine-readable metric serialization."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> dict[str, float]:
    """Return the classification metrics required by the experiment protocol."""

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def save_metrics(
    metrics: Mapping[str, object], json_path: str | Path, csv_path: str | Path
) -> None:
    """Write one metrics record to matching JSON and CSV files."""

    json_path = Path(json_path)
    csv_path = Path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(dict(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
