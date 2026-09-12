from types import SimpleNamespace

import torch
from datasets import Dataset, DatasetDict
from torch import nn

from explainable_kd.common.config import load_config
from explainable_kd.training.lrp_kd import lrp_explanation_loss, train_lrp_kd
from explainable_kd.xai.lrp import lrp_token_relevance_tensor


class TinyLRPKDModel(nn.Module):
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


def test_lrp_attribution_graph_reaches_student_parameters():
    model = TinyLRPKDModel()
    batch = {"input_ids": torch.tensor([[1, 3, 4, 0]]), "attention_mask": torch.tensor([[1, 1, 1, 0]])}
    scores, valid_masks, _ = lrp_token_relevance_tensor(model, batch, 1, TinyTokenizer(), create_graph=True)
    loss = lrp_explanation_loss(scores[0], torch.zeros_like(scores[0]), valid_masks[0])
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_lrp_explanation_loss_reaches_student():
    student = torch.tensor([1.0, 2.0], requires_grad=True)
    teacher = torch.tensor([2.0, 1.0])
    loss = lrp_explanation_loss(student, teacher, [True, True])
    loss.backward()
    assert loss.item() > 0
    assert torch.count_nonzero(student.grad) > 0
    assert torch.isfinite(student.grad).all()


def test_lrp_kd_smoke_saves_depth_specific_artifacts(tmp_path):
    config = load_config("configs/base.yaml", "configs/smoke.yaml", artifact_root=tmp_path)
    split = Dataset.from_dict({"input_ids": [[1] * 10, [2] * 10, [3] * 10, [4] * 10], "attention_mask": [[1] * 10] * 4, "label": [0, 1, 2, 3]})
    prepared = SimpleNamespace(dataset=DatasetDict({"train": split, "validation": split, "test": split}))
    result = train_lrp_kd(config, depth=6, teacher=TinyLRPKDModel(), model=TinyLRPKDModel(), tokenizer=TinyTokenizer(), prepared_data=prepared)
    assert result["experiment_id"] == "student_d6_lrp_kd"
    assert {"train_task_loss", "train_kd_loss", "train_lrp_loss", "train_loss"} <= set(result["history"][0])
    assert (tmp_path / "checkpoints/student_d6_lrp_kd/seed_42/best/pytorch_model.bin").exists()
