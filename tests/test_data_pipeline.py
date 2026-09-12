from dataclasses import replace

import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from explainable_kd.common.checkpoint import ArtifactPaths
from explainable_kd.common.config import load_config
from explainable_kd.data.pipeline import CLASS_NAMES, prepare_dataset, smoke_batch


class TinyTokenizer:
    name_or_path = "tiny-test-tokenizer"

    def __call__(
        self,
        texts,
        *,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
    ):
        if isinstance(texts, str):
            texts = [texts]
        input_ids = [[101, *range(1000, 1000 + len(text.split())), 102] for text in texts]
        if truncation:
            input_ids = [ids[:max_length] for ids in input_ids]
        if padding == "max_length":
            input_ids = [ids + [0] * (max_length - len(ids)) for ids in input_ids]
        attention_mask = [[int(token != 0) for token in ids] for ids in input_ids]
        token_type_ids = [[0] * len(ids) for ids in input_ids]
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        if return_tensors == "pt":
            output = {key: torch.tensor(value) for key, value in output.items()}
        return output


def make_trec_fixture():
    features = Features(
        {
            "text": Value("string"),
            "coarse_label": ClassLabel(names=list(CLASS_NAMES)),
            "fine_label": Value("int64"),
        }
    )
    train_labels = list(range(6)) * 3
    test_labels = list(range(6)) * 2
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": [f"short question {i}" for i in range(len(train_labels))],
                    "coarse_label": train_labels,
                    "fine_label": [0] * len(train_labels),
                },
                features=features,
            ),
            "test": Dataset.from_dict(
                {
                    "text": [f"test question {i}" for i in range(len(test_labels))],
                    "coarse_label": test_labels,
                    "fine_label": [0] * len(test_labels),
                },
                features=features,
            ),
        }
    )


def test_prepare_dataset_keeps_label_mapping_and_deterministic_splits(tmp_path):
    config = load_config("configs/base.yaml", artifact_root=tmp_path)
    config = replace(
        config,
        data=replace(config.data, validation_size=1 / 3),
        paths=ArtifactPaths(tmp_path),
    )

    first = prepare_dataset(config, raw_dataset=make_trec_fixture(), tokenizer=TinyTokenizer())
    second = prepare_dataset(config, raw_dataset=make_trec_fixture(), tokenizer=TinyTokenizer())

    assert tuple(first.manifest.label_names) == CLASS_NAMES
    assert set(first.dataset) == {"train", "validation", "test"}
    assert first.manifest.split_example_ids == second.manifest.split_example_ids
    assert first.manifest.data_fingerprint == second.manifest.data_fingerprint
    assert all(len(row) == 10 for row in first.dataset["train"]["input_ids"])


def test_smoke_batch_is_a_rectangular_bert_input():
    batch = smoke_batch(
        ["Who invented the telephone?", "Where is Seoul?"],
        TinyTokenizer(),
        max_length=10,
    )

    assert set(batch) >= {"input_ids", "attention_mask", "token_type_ids"}
    assert all(value.shape == (2, 10) for value in batch.values())
    assert all(value.dtype == torch.int64 for value in batch.values())
