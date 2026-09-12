"""Integrated Gradients extraction for saved Teacher and Student checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from explainable_kd.common.runtime import resolve_device
from explainable_kd.data.pipeline import prepare_dataset
from explainable_kd.training.teacher import TEACHER_ID, _to_device


IG_EXPERIMENTS = (
    TEACHER_ID,
    "student_d10_baseline",
    "student_d8_baseline",
    "student_d6_baseline",
    "student_d10_kd",
    "student_d8_kd",
    "student_d6_kd",
)


def integrated_gradients(model, batch: Mapping[str, torch.Tensor], target_class: int, tokenizer: Any, steps: int = 16) -> dict[str, Any]:
    """Return signed embedding IG scores, masking special tokens and padding."""
    if steps < 1:
        raise ValueError("steps must be positive")
    scores, valid_mask, logits = integrated_gradients_tensor(
        model, batch, target_class, tokenizer, steps=steps, create_graph=False
    )
    ids = batch["input_ids"][0].detach().cpu().tolist()
    return {
        "tokens": tokenizer.convert_ids_to_tokens(ids),
        "scores": [float(score) if valid else 0.0 for score, valid in zip(scores[0].detach().cpu().tolist(), valid_mask[0])],
        "valid_mask": valid_mask[0],
        "target_class": int(target_class),
        "predicted_class": int(logits.argmax(-1)[0].item()),
    }


def integrated_gradients_tensor(model, batch: Mapping[str, torch.Tensor], target_class, tokenizer: Any, *, steps: int = 16, create_graph: bool = False):
    """Return tensor IG scores; ``create_graph=True`` keeps the Student gradient path."""
    was_training = model.training
    model.eval()
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(input_ids)
    baseline = embedding_layer(torch.full_like(input_ids, tokenizer.pad_token_id))
    total_gradients = torch.zeros_like(embeddings)
    for alpha in torch.linspace(0.0, 1.0, steps + 1, device=embeddings.device)[1:]:
        interpolated = (baseline + alpha * (embeddings - baseline)).detach().requires_grad_(True)
        model_inputs = {key: value for key, value in batch.items() if key not in {"input_ids", "labels"}}
        output = model(inputs_embeds=interpolated, **model_inputs)
        logits = output.logits if hasattr(output, "logits") else output["logits"]
        targets = target_class if torch.is_tensor(target_class) else torch.full((logits.shape[0],), target_class, device=logits.device, dtype=torch.long)
        selected = logits.gather(1, targets.view(-1, 1)).sum()
        gradients = torch.autograd.grad(selected, interpolated, create_graph=create_graph)[0]
        total_gradients += gradients
    scores = ((embeddings - baseline) * total_gradients / steps).sum(dim=-1)

    with torch.no_grad():
        output = model(**{key: value for key, value in batch.items() if key != "labels"})
        logits = output.logits if hasattr(output, "logits") else output["logits"]
    special_ids = set(tokenizer.all_special_ids)
    valid_mask = [
        [bool(mask) and token_id not in special_ids for token_id, mask in zip(ids, masks)]
        for ids, masks in zip(input_ids.detach().cpu().tolist(), attention_mask.detach().cpu().tolist())
    ]
    if was_training:
        model.train()
    return scores, valid_mask, logits


def cosine_similarity(first: list[float], second: list[float], valid_mask: list[bool]) -> float:
    """Compute cosine similarity over aligned, non-special, non-padding scores."""
    if len(first) != len(second) or len(first) != len(valid_mask):
        raise ValueError("attribution arrays must have equal lengths")
    left = torch.tensor([a for a, valid in zip(first, valid_mask) if valid], dtype=torch.float32)
    right = torch.tensor([b for b, valid in zip(second, valid_mask) if valid], dtype=torch.float32)
    if not len(left) or torch.linalg.vector_norm(left) == 0 or torch.linalg.vector_norm(right) == 0:
        return 0.0
    return float(torch.nn.functional.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())


def extract_ig(config, max_examples: int | None = None, *, experiment_ids=IG_EXPERIMENTS) -> dict[str, Any]:
    """Extract IG for the same test questions across Teacher, baselines, and KD Students."""
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name, revision=config.data.tokenizer_revision, use_fast=True)
    prepared = prepare_dataset(config, tokenizer=tokenizer)
    device = resolve_device(config.runtime.device)
    max_examples = max_examples or config.xai.max_examples
    if max_examples < 1:
        raise ValueError("max_examples must be positive")
    models = {}
    for experiment_id in experiment_ids:
        path = config.paths.checkpoint(experiment_id, config.runtime.seed, "best")
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found for {experiment_id}: {path}")
        models[experiment_id] = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(device).eval()

    rows_by_experiment = {experiment_id: [] for experiment_id in experiment_ids}
    for row in prepared.dataset["test"].select(range(min(max_examples, len(prepared.dataset["test"])) )):
        batch = _row_to_batch(row, device)
        target_class = int(row["label"])
        for experiment_id, model in models.items():
            record = integrated_gradients(model, batch, target_class, tokenizer, config.xai.ig_steps)
            record["experiment_id"] = experiment_id
            record["example_id"] = row["example_id"]
            rows_by_experiment[experiment_id].append(record)

    for experiment_id, rows in rows_by_experiment.items():
        path = config.paths.root / "attributions" / "ig" / experiment_id / f"seed_{config.runtime.seed}" / "examples.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    teacher_rows = {row["example_id"]: row for row in rows_by_experiment[TEACHER_ID]}
    similarity = {}
    for experiment_id in experiment_ids[1:]:
        values = [cosine_similarity(teacher_rows[row["example_id"]]["scores"], row["scores"], teacher_rows[row["example_id"]]["valid_mask"]) for row in rows_by_experiment[experiment_id]]
        similarity[experiment_id] = {"values": values, "mean_cosine_similarity": sum(values) / max(1, len(values)), "teacher_ref": TEACHER_ID}
    similarity_path = config.paths.root / "metrics" / "ig_similarity" / f"seed_{config.runtime.seed}.json"
    similarity_path.parent.mkdir(parents=True, exist_ok=True)
    similarity_path.write_text(json.dumps(similarity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure_paths = _write_comparison_figures(config, rows_by_experiment)
    return {"status": "passed", "method": "integrated_gradients", "device": str(device), "examples": len(teacher_rows), "experiments": list(experiment_ids), "attribution_paths": {key: str(config.paths.root / "attributions" / "ig" / key / f"seed_{config.runtime.seed}" / "examples.jsonl") for key in experiment_ids}, "similarity_path": str(similarity_path), "figure_paths": figure_paths}


def _row_to_batch(row: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("input_ids", "attention_mask", "token_type_ids")
    return {key: torch.tensor([row[key]], dtype=torch.long, device=device) for key in keys if key in row}


def _write_comparison_figures(config, rows_by_experiment):
    import matplotlib.pyplot as plt

    figure_root = config.paths.root / "figures" / "token_importance" / "ig"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, teacher_row in enumerate(rows_by_experiment[TEACHER_ID]):
        labels = ["Teacher" if experiment_id == TEACHER_ID else experiment_id.removeprefix("student_") for experiment_id in rows_by_experiment]
        experiment_ids = tuple(rows_by_experiment)
        figure, axes = plt.subplots(len(experiment_ids), 1, figsize=(max(8, len(teacher_row["tokens"]) * 0.35), 10), sharex=True)
        axes = [axes] if len(experiment_ids) == 1 else axes
        for axis, experiment_id, label in zip(axes, experiment_ids, labels):
            row = rows_by_experiment[experiment_id][index]
            valid = row["valid_mask"]
            tokens = [token for token, is_valid in zip(row["tokens"], valid) if is_valid]
            scores = [score for score, is_valid in zip(row["scores"], valid) if is_valid]
            colors = ["#2b6cb0" if score >= 0 else "#c53030" for score in scores]
            axis.bar(range(len(tokens)), scores, color=colors)
            axis.set_ylabel(label, rotation=0, ha="right", va="center")
            axis.axhline(0, color="black", linewidth=0.5)
        valid_tokens = [token for token, is_valid in zip(teacher_row["tokens"], teacher_row["valid_mask"]) if is_valid]
        axes[-1].set_xticks(range(len(valid_tokens)), valid_tokens, rotation=60, ha="right")
        figure.tight_layout()
        path = figure_root / f"{str(teacher_row['example_id']).replace(':', '_')}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))
    return paths
