from types import SimpleNamespace

import torch
from datasets import Dataset, DatasetDict
from torch import nn

from explainable_kd.common.config import load_config
from explainable_kd.training.ig_kd import ig_explanation_loss, train_ig_kd


class TinyIGKDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.classifier = nn.Linear(4, 6)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        values = inputs_embeds if inputs_embeds is not None else self.embedding(input_ids)
        pooled = (values * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        return SimpleNamespace(logits=self.classifier(pooled))

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path / "pytorch_model.bin")


class TinyTokenizer:
    pad_token_id = 0
    all_special_ids = [0]

    def save_pretrained(self, path):
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_ig_explanation_loss_has_student_gradient():
    student = torch.tensor([[1.0, 2.0]], requires_grad=True)
    teacher = torch.tensor([[2.0, 1.0]])

    loss = ig_explanation_loss(student, teacher, [True, True])
    loss.backward()

    assert loss.item() > 0
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert torch.count_nonzero(student.grad) > 0


def test_ig_kd_smoke_saves_depth_specific_artifacts_and_loss_components(tmp_path):
    config = load_config("configs/base.yaml", "configs/smoke.yaml", artifact_root=tmp_path)
    split = Dataset.from_dict(
        {
            "input_ids": [[1] * 10, [2] * 10, [3] * 10, [4] * 10],
            "attention_mask": [[1] * 10] * 4,
            "label": [0, 1, 2, 3],
        }
    )
    prepared = SimpleNamespace(dataset=DatasetDict({"train": split, "validation": split, "test": split}))

    result = train_ig_kd(
        config,
        depth=6,
        teacher=TinyIGKDModel(),
        model=TinyIGKDModel(),
        tokenizer=TinyTokenizer(),
        prepared_data=prepared,
    )

    assert result["experiment_id"] == "student_d6_ig_kd"
    assert result["teacher_ref"] == "teacher_d12_supervised"
    assert {"train_task_loss", "train_kd_loss", "train_ig_loss", "train_loss"} <= set(result["history"][0])
    assert (tmp_path / "checkpoints/student_d6_ig_kd/seed_42/best/pytorch_model.bin").exists()
    assert (tmp_path / "metrics/student_d6_ig_kd/seed_42/metrics.json").exists()
