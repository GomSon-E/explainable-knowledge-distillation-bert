"""Deterministic TREC-6 loading, splitting, tokenization, and smoke checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from explainable_kd.common.config import ExperimentConfig


CLASS_NAMES = ("ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM")
LABEL_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class DataManifest:
    dataset_name: str
    dataset_revision: str
    tokenizer_name: str
    tokenizer_revision: str
    max_length: int
    short_input_policy: str
    validation_size: float
    seed: int
    label_names: tuple[str, ...]
    split_example_ids: Mapping[str, list[str]]
    split_counts: Mapping[str, int]
    data_fingerprint: str
    created_at_utc: str


@dataclass(frozen=True)
class PreparedData:
    dataset: DatasetDict
    manifest: DataManifest


def prepare_dataset(
    config: ExperimentConfig,
    *,
    raw_dataset: DatasetDict | None = None,
    tokenizer: PreTrainedTokenizerBase | Any | None = None,
    save: bool = True,
) -> PreparedData:
    """Prepare the one deterministic dataset used by every experiment."""

    data_config = config.data
    raw_dataset = raw_dataset or load_dataset(
        data_config.dataset_name,
        revision=data_config.dataset_revision,
    )
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        data_config.tokenizer_name,
        revision=data_config.tokenizer_revision,
        use_fast=True,
    )
    _validate_raw_splits(raw_dataset)

    train_parts = [_normalize_labels(raw_dataset["train"], "train")]
    if "validation" in raw_dataset:
        train_parts.append(_normalize_labels(raw_dataset["validation"], "validation"))
    train = concatenate_datasets(train_parts) if len(train_parts) > 1 else train_parts[0]
    test = _normalize_labels(raw_dataset["test"], "test")
    if data_config.short_input_policy == "filter":
        train = _filter_short_inputs(train, tokenizer, data_config.max_length)
        test = _filter_short_inputs(test, tokenizer, data_config.max_length)

    split = train.train_test_split(
        test_size=data_config.validation_size,
        seed=config.runtime.seed,
        stratify_by_column="label",
    )
    datasets = DatasetDict(
        {"train": split["train"], "validation": split["test"], "test": test}
    )
    datasets = _limit_samples(datasets, config)

    split_example_ids = {
        name: list(dataset["example_id"]) for name, dataset in datasets.items()
    }
    fingerprint_payload = {
        "dataset_name": data_config.dataset_name,
        "dataset_revision": data_config.dataset_revision,
        "tokenizer_name": data_config.tokenizer_name,
        "tokenizer_revision": data_config.tokenizer_revision,
        "max_length": data_config.max_length,
        "short_input_policy": data_config.short_input_policy,
        "validation_size": data_config.validation_size,
        "seed": config.runtime.seed,
        "label_names": CLASS_NAMES,
        "split_example_ids": split_example_ids,
    }
    data_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    processed_path = config.paths.processed(data_fingerprint)

    if save and processed_path.exists():
        tokenized = load_from_disk(str(processed_path))
    else:
        tokenized = datasets.map(
            lambda batch: tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=data_config.max_length,
            ),
            batched=True,
            desc="Tokenizing TREC-6",
        )
        if save:
            config.paths.ensure_base_directories()
            tokenized.save_to_disk(str(processed_path))

    manifest = DataManifest(
        **fingerprint_payload,
        split_counts={name: len(dataset) for name, dataset in tokenized.items()},
        data_fingerprint=data_fingerprint,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    if save:
        _write_json(config.paths.split_manifest, asdict(manifest))
    return PreparedData(dataset=tokenized, manifest=manifest)


def smoke_batch(
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase | Any,
    *,
    max_length: int,
) -> dict[str, Any]:
    """Turn real question text into the rectangular tensors BERT consumes."""

    return dict(
        tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    )


def run_smoke_test(config: ExperimentConfig, batch_size: int = 4) -> dict[str, Any]:
    """Download TREC-6 and build one real BERT tokenizer input batch."""

    tokenizer = AutoTokenizer.from_pretrained(
        config.data.tokenizer_name,
        revision=config.data.tokenizer_revision,
        use_fast=True,
    )
    prepared = prepare_dataset(config, tokenizer=tokenizer)
    texts = prepared.dataset["train"].select(range(batch_size))["text"]
    batch = smoke_batch(texts, tokenizer, max_length=config.data.max_length)
    result = {
        "status": "passed",
        "dataset": config.data.dataset_name,
        "dataset_revision": config.data.dataset_revision,
        "tokenizer": config.data.tokenizer_name,
        "tokenizer_revision": config.data.tokenizer_revision,
        "batch_size": batch_size,
        "sequence_length": config.data.max_length,
        "tensor_shapes": {key: list(value.shape) for key, value in batch.items()},
        "tensor_dtypes": {key: str(value.dtype) for key, value in batch.items()},
        "split_counts": dict(prepared.manifest.split_counts),
        "data_fingerprint": prepared.manifest.data_fingerprint,
        "artifact_root": str(config.paths.root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(config.paths.root / "metrics/smoke_test.json", result)
    return result


def _normalize_labels(dataset: Dataset, source_split: str) -> Dataset:
    feature = dataset.features.get("coarse_label")
    source_names = tuple(feature.names) if isinstance(feature, ClassLabel) else None
    observed_names = (
        set(source_names) if source_names is not None else set(dataset.unique("coarse_label"))
    )
    if observed_names != set(CLASS_NAMES):
        raise ValueError(f"unexpected TREC coarse labels: {sorted(observed_names)}")

    def normalize(batch: Mapping[str, list[Any]], indices: list[int]) -> dict[str, Any]:
        names = [
            source_names[label] if source_names is not None else label
            for label in batch["coarse_label"]
        ]
        return {
            "text": batch["text"],
            "label": [LABEL_TO_ID[name] for name in names],
            "example_id": [f"{source_split}:{index}" for index in indices],
        }

    normalized = dataset.map(
        normalize,
        batched=True,
        with_indices=True,
        remove_columns=dataset.column_names,
        desc=f"Mapping {source_split} labels",
    )
    return normalized.cast_column("label", ClassLabel(names=list(CLASS_NAMES)))


def _filter_short_inputs(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase | Any,
    max_length: int,
) -> Dataset:
    return dataset.filter(
        lambda batch: [
            len(input_ids) <= max_length
            for input_ids in tokenizer(
                batch["text"], padding=False, truncation=False
            )["input_ids"]
        ],
        batched=True,
        desc=f"Filtering inputs longer than {max_length} tokens",
    )


def _limit_samples(dataset: DatasetDict, config: ExperimentConfig) -> DatasetDict:
    limits = {
        "train": config.data.max_train_samples,
        "validation": config.data.max_validation_samples,
        "test": config.data.max_test_samples,
    }
    return DatasetDict(
        {
            name: split.select(range(min(limit, len(split)))) if limit else split
            for name, split in dataset.items()
            for limit in [limits[name]]
        }
    )


def _validate_raw_splits(dataset: DatasetDict) -> None:
    missing = {"train", "test"} - set(dataset)
    if missing:
        raise ValueError(f"TREC dataset is missing splits: {sorted(missing)}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
