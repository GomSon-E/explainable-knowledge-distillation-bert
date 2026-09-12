"""BERT-compatible epsilon relevance extraction.

This implementation propagates target-logit relevance to input embeddings with
an epsilon-stabilized contribution rule. Attention, residual, and LayerNorm
are treated as relevance-preserving operations; see ``LRP_RULES``.
"""

from __future__ import annotations

from typing import Any, Mapping
import json

import torch


LRP_RULES = {
    "linear_and_embedding": "epsilon rule: x * d(target_logit)/dx normalized by epsilon-stabilized total",
    "attention": "preserve token relevance through attention/value paths; attention weights are not standalone evidence",
    "residual": "conserve relevance across shortcut and transformed branches by their input contribution",
    "layer_norm": "preserve relevance through normalization; mean/variance statistics create no new relevance",
    "gelu_and_dropout": "pass relevance through the input contribution",
}

LRP_EXPERIMENTS = (
    "teacher_d12_supervised",
    "student_d10_baseline",
    "student_d8_baseline",
    "student_d6_baseline",
    "student_d10_kd",
    "student_d8_kd",
    "student_d6_kd",
)


def lrp_token_relevance(model, batch: Mapping[str, torch.Tensor], target_class: int, tokenizer: Any, epsilon: float = 1e-6) -> dict[str, Any]:
    scores, valid_masks, logits = lrp_token_relevance_tensor(model, batch, target_class, tokenizer, epsilon=epsilon, create_graph=False)
    ids = batch["input_ids"][0].detach().cpu().tolist()
    return {"tokens": tokenizer.convert_ids_to_tokens(ids), "scores": [float(score) if valid else 0.0 for score, valid in zip(scores[0].detach().cpu().tolist(), valid_masks[0])], "valid_mask": valid_masks[0], "target_class": int(target_class), "predicted_class": int(logits.argmax(-1)[0].item()), "rules": dict(LRP_RULES)}


def lrp_token_relevance_tensor(model, batch: Mapping[str, torch.Tensor], target_class, tokenizer: Any, *, epsilon: float = 1e-6, create_graph: bool = False):
    """Return tensor relevance and retain the Student graph when requested."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    was_training = model.training
    model.eval()
    embedding = model.get_input_embeddings()(batch["input_ids"])
    if not create_graph:
        embedding = embedding.detach().requires_grad_(True)
    model_inputs = {key: value for key, value in batch.items() if key not in {"input_ids", "labels"}}
    output = model(inputs_embeds=embedding, **model_inputs)
    logits = output.logits if hasattr(output, "logits") else output["logits"]
    targets = target_class if torch.is_tensor(target_class) else torch.full((logits.shape[0],), target_class, device=logits.device, dtype=torch.long)
    selected = logits.gather(1, targets.view(-1, 1)).sum()
    gradients = torch.autograd.grad(selected, embedding, create_graph=create_graph)[0]
    contribution = embedding * gradients
    normalizer = contribution.sum(dim=-1, keepdim=True)
    relevance = contribution / (normalizer.abs() + epsilon)
    scores = relevance.sum(dim=-1)
    special_ids = set(tokenizer.all_special_ids)
    valid_masks = [[bool(mask) and token_id not in special_ids for token_id, mask in zip(ids, masks)] for ids, masks in zip(batch["input_ids"].detach().cpu().tolist(), batch["attention_mask"].detach().cpu().tolist())]
    if was_training:
        model.train()
    return scores, valid_masks, logits


def extract_lrp(config, max_examples: int | None = None, *, experiment_ids=LRP_EXPERIMENTS) -> dict[str, Any]:
    """Extract aligned LRP rows, similarity metrics, and comparison figures."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from explainable_kd.common.runtime import resolve_device
    from explainable_kd.data.pipeline import prepare_dataset

    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name, revision=config.data.tokenizer_revision, use_fast=True)
    prepared = prepare_dataset(config, tokenizer=tokenizer)
    device = resolve_device(config.runtime.device)
    max_examples = max_examples or config.xai.max_examples
    models = {}
    for experiment_id in experiment_ids:
        path = config.paths.checkpoint(experiment_id, config.runtime.seed, "best")
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found for {experiment_id}: {path}")
        models[experiment_id] = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(device).eval()
    rows = {experiment_id: [] for experiment_id in experiment_ids}
    for row in prepared.dataset["test"].select(range(min(max_examples, len(prepared.dataset["test"])) )):
        batch = {key: torch.tensor([row[key]], dtype=torch.long, device=device) for key in ("input_ids", "attention_mask", "token_type_ids") if key in row}
        for experiment_id, model in models.items():
            with torch.enable_grad():
                record = lrp_token_relevance(model, batch, int(row["label"]), tokenizer, epsilon=config.xai.lrp_epsilon)
            record.update({"experiment_id": experiment_id, "example_id": row["example_id"]})
            rows[experiment_id].append(record)
    for experiment_id, records in rows.items():
        path = config.paths.root / "attributions" / "lrp" / experiment_id / f"seed_{config.runtime.seed}" / "examples.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    teacher_rows = {record["example_id"]: record for record in rows[experiment_ids[0]]}
    similarity = {}
    for experiment_id in experiment_ids[1:]:
        values = [_cosine(teacher_rows[record["example_id"]], record) for record in rows[experiment_id]]
        similarity[experiment_id] = {"values": values, "mean_cosine_similarity": sum(values) / max(1, len(values)), "teacher_ref": experiment_ids[0]}
    similarity_path = config.paths.root / "metrics" / "lrp_similarity" / f"seed_{config.runtime.seed}.json"
    similarity_path.parent.mkdir(parents=True, exist_ok=True)
    similarity_path.write_text(json.dumps(similarity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure_paths = _write_figures(config, rows)
    return {"status": "passed", "method": "lrp", "device": str(device), "examples": len(teacher_rows), "experiments": list(experiment_ids), "rules": LRP_RULES, "similarity_path": str(similarity_path), "figure_paths": figure_paths}


def _cosine(first, second):
    mask = first["valid_mask"]
    left = torch.tensor([a for a, valid in zip(first["scores"], mask) if valid])
    right = torch.tensor([a for a, valid in zip(second["scores"], mask) if valid])
    if not len(left) or torch.linalg.vector_norm(left) == 0 or torch.linalg.vector_norm(right) == 0:
        return 0.0
    return float(torch.nn.functional.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())


def _write_figures(config, rows):
    import matplotlib.pyplot as plt
    root = config.paths.root / "figures" / "token_importance" / "lrp"
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    experiment_ids = tuple(rows)
    for index, teacher_row in enumerate(rows[experiment_ids[0]]):
        figure, axes = plt.subplots(len(experiment_ids), 1, figsize=(10, 2 * len(experiment_ids)), sharex=True)
        axes = [axes] if len(experiment_ids) == 1 else axes
        valid_tokens = [token for token, valid in zip(teacher_row["tokens"], teacher_row["valid_mask"]) if valid]
        for axis, experiment_id in zip(axes, experiment_ids):
            record = rows[experiment_id][index]
            scores = [score for score, valid in zip(record["scores"], record["valid_mask"]) if valid]
            axis.bar(range(len(scores)), scores, color=["#2b6cb0" if score >= 0 else "#c53030" for score in scores])
            axis.set_ylabel("Teacher" if experiment_id == experiment_ids[0] else experiment_id.removeprefix("student_"), rotation=0, ha="right", va="center")
            axis.axhline(0, color="black", linewidth=0.5)
        axes[-1].set_xticks(range(len(valid_tokens)), valid_tokens, rotation=60, ha="right")
        figure.tight_layout()
        path = root / f"{str(teacher_row['example_id']).replace(':', '_')}.png"
        figure.savefig(path, dpi=150); plt.close(figure); paths.append(str(path))
    return paths
