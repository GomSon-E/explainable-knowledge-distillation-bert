from types import SimpleNamespace

import torch
from datasets import Dataset, DatasetDict
from torch import nn

from explainable_kd.common.config import load_config
from explainable_kd.training.baseline import baseline_experiment_id, train_baseline
import explainable_kd.training.baseline as baseline_module
from explainable_kd.training.kd import kd_loss, train_kd
from explainable_kd.training.teacher import _collate_batch, teacher_experiment_id, train_teacher


def test_teacher_experiment_is_fixed_to_the_canonical_12_layer_condition(tmp_path):
    config = load_config("configs/base.yaml", artifact_root=tmp_path)

    assert teacher_experiment_id(config) == "teacher_d12_supervised"


def test_baseline_experiment_ids_include_depth_and_method():
    assert [baseline_experiment_id(depth) for depth in (10, 8, 6)] == [
        "student_d10_baseline",
        "student_d8_baseline",
        "student_d6_baseline",
    ]


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


def test_baseline_smoke_saves_depth_specific_artifacts_without_teacher(tmp_path):
    config = load_config("configs/base.yaml", "configs/smoke.yaml", artifact_root=tmp_path)
    split = Dataset.from_dict(
        {
            "input_ids": [[1] * 10, [2] * 10, [3] * 10, [4] * 10],
            "attention_mask": [[1] * 10] * 4,
            "label": [0, 1, 2, 3],
        }
    )
    prepared = SimpleNamespace(dataset=DatasetDict({"train": split, "validation": split, "test": split}))

    result = train_baseline(
        config,
        depth=6,
        model=TinyTeacher(),
        tokenizer=TinyTokenizer(),
        prepared_data=prepared,
    )

    assert result["experiment_id"] == "student_d6_baseline"
    assert result["teacher_ref"] is None
    assert (tmp_path / "checkpoints/student_d6_baseline/seed_42/best/pytorch_model.bin").exists()
    assert (tmp_path / "metrics/student_d6_baseline/seed_42/metrics.json").exists()


def test_baseline_model_passes_num_labels_only_through_config(monkeypatch, tmp_path):
    config = load_config("configs/base.yaml", artifact_root=tmp_path)
    fake_config = SimpleNamespace(num_hidden_layers=12)
    captured = {}

    monkeypatch.setattr(
        baseline_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: fake_config,
    )

    def fake_model_loader(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(config=fake_config)

    monkeypatch.setattr(
        baseline_module.AutoModelForSequenceClassification,
        "from_pretrained",
        fake_model_loader,
    )

    baseline_module.build_baseline_model(config, 6)

    assert fake_config.num_hidden_layers == 6
    assert "num_labels" not in captured


def test_kd_loss_uses_temperature_and_has_expected_components():
    student = torch.tensor([[2.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[0.0, 2.0]])

    loss = kd_loss(student, teacher, temperature=2.0)
    loss.backward()

    expected = torch.nn.functional.kl_div(
        torch.log_softmax(student / 2.0, dim=-1),
        torch.softmax(teacher / 2.0, dim=-1),
        reduction="batchmean",
    ) * 4.0
    assert torch.allclose(loss.detach(), expected)
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_kd_smoke_freezes_teacher_and_saves_depth_specific_artifacts(tmp_path):
    config = load_config("configs/base.yaml", "configs/smoke.yaml", artifact_root=tmp_path)
    split = Dataset.from_dict(
        {
            "input_ids": [[1] * 10, [2] * 10, [3] * 10, [4] * 10],
            "attention_mask": [[1] * 10] * 4,
            "label": [0, 1, 2, 3],
        }
    )
    prepared = SimpleNamespace(dataset=DatasetDict({"train": split, "validation": split, "test": split}))
    teacher = TinyTeacher()
    student = TinyTeacher()

    result = train_kd(
        config,
        depth=6,
        teacher=teacher,
        model=student,
        tokenizer=TinyTokenizer(),
        prepared_data=prepared,
    )

    assert result["experiment_id"] == "student_d6_kd"
    assert result["teacher_ref"] == "teacher_d12_supervised"
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert {"train_task_loss", "train_kd_loss"} <= set(result["history"][0])
    assert (tmp_path / "checkpoints/student_d6_kd/seed_42/best/pytorch_model.bin").exists()
    assert (tmp_path / "metrics/student_d6_kd/seed_42/metrics.json").exists()
