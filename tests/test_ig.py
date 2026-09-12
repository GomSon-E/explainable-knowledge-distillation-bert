from types import SimpleNamespace

import torch
from torch import nn

from explainable_kd.xai.ig import cosine_similarity, integrated_gradients


class TinyIGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.classifier = nn.Linear(4, 2)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        values = inputs_embeds if inputs_embeds is not None else self.embedding(input_ids)
        pooled = (values * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        return SimpleNamespace(logits=self.classifier(pooled))


class TinyTokenizer:
    pad_token_id = 0
    all_special_ids = [0, 1, 2]

    def convert_ids_to_tokens(self, ids):
        return [f"tok-{value}" for value in ids]


def test_integrated_gradients_masks_special_and_padding_tokens():
    model = TinyIGModel()
    batch = {
        "input_ids": torch.tensor([[1, 3, 4, 2, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
    }

    result = integrated_gradients(model, batch, target_class=1, tokenizer=TinyTokenizer(), steps=4)

    assert result["tokens"] == ["tok-1", "tok-3", "tok-4", "tok-2", "tok-0"]
    assert result["valid_mask"] == [False, True, True, False, False]
    assert result["target_class"] == 1
    assert len(result["scores"]) == 5
    assert result["scores"][0] == 0.0
    assert result["scores"][4] == 0.0
    assert "predicted_class" in result


def test_cosine_similarity_uses_only_valid_aligned_tokens():
    assert cosine_similarity([99.0, 1.0, 0.0], [5.0, 2.0, 9.0], [False, True, False]) == 1.0
