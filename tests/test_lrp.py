from types import SimpleNamespace

import torch
from torch import nn

from explainable_kd.xai.lrp import lrp_token_relevance


class TinyLRPModel(nn.Module):
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


def test_lrp_uses_same_target_and_masks_special_padding():
    result = lrp_token_relevance(
        TinyLRPModel(),
        {"input_ids": torch.tensor([[1, 3, 4, 2, 0]]), "attention_mask": torch.tensor([[1, 1, 1, 1, 0]])},
        target_class=1,
        tokenizer=TinyTokenizer(),
        epsilon=1e-6,
    )

    assert result["target_class"] == 1
    assert result["valid_mask"] == [False, True, True, False, False]
    assert result["scores"][0] == 0.0
    assert result["scores"][4] == 0.0
    assert result["rules"]["attention"]
    assert result["rules"]["residual"]
    assert result["rules"]["layer_norm"]
