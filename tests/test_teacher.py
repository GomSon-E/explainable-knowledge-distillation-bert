from types import SimpleNamespace

import torch
from datasets import Dataset, DatasetDict
from torch import nn

from explainable_kd.common.config import load_config
from explainable_kd.training.teacher import _collate_batch, teacher_experiment_id, train_teacher


def test_teacher_experiment_is_fixed_to_the_canonical_12_layer_condition(tmp_path):
    config = load_config("configs/base.yaml", artifact_root=tmp_path)

    assert teacher_experiment_id(config) == "teacher_d12_supervised"


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(10, 6)

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        logits = self.projection(input_ids.float())
        return SimpleNamespace(logits=logits, loss=nn.functional.cross_entropy(logits, labels))

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path / "pytorch_model.bin")


class TinyTokenizer:
    def save_pretrained(self, path):
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_teacher_smoke_persists_metrics_and_best_last_checkpoints(tmp_path):
    config = load_config("configs/base.yaml", "configs/smoke.yaml", artifact_root=tmp_path)
    split = Dataset.from_dict(
        {
            "input_ids": [[1] * 10, [2] * 10, [3] * 10, [4] * 10],
            "attention_mask": [[1] * 10] * 4,
            "label": [0, 1, 2, 3],
        }
    )
    prepared = SimpleNamespace(dataset=DatasetDict({"train": split, "validation": split, "test": split}))

    result = train_teacher(config, model=TinyTeacher(), tokenizer=TinyTokenizer(), prepared_data=prepared)

    assert len(result["history"]) == 1
    assert set(result["test"]) == {"loss", "accuracy", "precision_macro", "recall_macro", "f1_macro"}
    assert (tmp_path / "checkpoints/teacher_d12_supervised/seed_42/best/pytorch_model.bin").exists()
    assert (tmp_path / "checkpoints/teacher_d12_supervised/seed_42/last/training_state.pt").exists()
    assert (tmp_path / "metrics/teacher_d12_supervised/seed_42/metrics.json").exists()
    assert result["efficiency"]["latency_scope"] == "classification_only"


def test_collator_converts_dataset_rows_without_torch_formatter_dependencies():
    batch = _collate_batch(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "label": 0},
            {"input_ids": [3, 4], "attention_mask": [1, 1], "label": 1},
        ]
    )

    assert batch["input_ids"].shape == (2, 2)
    assert batch["labels"].tolist() == [0, 1]
