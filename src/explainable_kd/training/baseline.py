"""Labels-only supervised baselines for 10-, 8-, and 6-layer BERT."""

from __future__ import annotations

import csv
import json
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from explainable_kd.common.config import ExperimentConfig
from explainable_kd.common.runtime import resolve_device
from explainable_kd.data.pipeline import prepare_dataset
from explainable_kd.training.teacher import (
    LABEL_NAMES,
    _load_model_weights,
    _loaders,
    _save_checkpoint,
    _to_device,
    evaluate_model,
    measure_efficiency,
)


BASELINE_DEPTHS = (10, 8, 6)


def baseline_experiment_id(depth: int) -> str:
    if depth not in BASELINE_DEPTHS:
        raise ValueError(f"baseline depth must be one of {BASELINE_DEPTHS}")
    return f"student_d{depth}_baseline"


def train_baselines(config: ExperimentConfig) -> dict[str, Any]:
    """Train all three labels-only baseline models independently."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.data.tokenizer_name,
        revision=config.data.tokenizer_revision,
        use_fast=True,
    )
    prepared_data = prepare_dataset(config, tokenizer=tokenizer)
    return {
        baseline_experiment_id(depth): train_baseline(
            config, depth=depth, tokenizer=tokenizer, prepared_data=prepared_data
        )
        for depth in BASELINE_DEPTHS
    }


def train_baseline(
    config: ExperimentConfig,
    *,
    depth: int,
    model=None,
    tokenizer=None,
    prepared_data=None,
) -> dict[str, Any]:
    """Train one labels-only baseline without loading a Teacher."""
    experiment_id = baseline_experiment_id(depth)
    device = resolve_device(config.runtime.device)
    config.paths.ensure_base_directories()
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        config.data.tokenizer_name,
        revision=config.data.tokenizer_revision,
        use_fast=True,
    )
    prepared_data = prepared_data or prepare_dataset(config, tokenizer=tokenizer)
    model = model or build_baseline_model(config, depth)
    model.to(device)
    train_loader, validation_loader, test_loader = _loaders(prepared_data.dataset, config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader) // config.training.gradient_accumulation_steps)
    total_steps = config.training.max_steps or config.training.epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(total_steps * config.training.warmup_ratio),
        max(1, total_steps),
    )
    paths = {
        slot: config.paths.checkpoint(experiment_id, config.runtime.seed, slot)
        for slot in ("best", "last")
    }
    history, start_epoch, best_f1, global_step = _resume(
        config, model, optimizer, scheduler, paths, device
    )

    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        train_loss, steps_seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            outputs = model(**_to_device(batch, device))
            loss = outputs.loss / config.training.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            train_loss += loss.item() * config.training.gradient_accumulation_steps
            steps_seen = step + 1
            if config.training.max_steps and global_step >= config.training.max_steps:
                break

        validation = evaluate_model(model, validation_loader, device)
        improved = validation["f1_macro"] > best_f1
        if improved:
            best_f1 = validation["f1_macro"]
        history.append({"epoch": epoch + 1, "train_loss": train_loss / max(1, steps_seen), **validation})
        _save_checkpoint(model, tokenizer, optimizer, scheduler, paths["last"], epoch + 1, best_f1, history, global_step)
        if improved:
            _save_checkpoint(model, tokenizer, None, None, paths["best"], epoch + 1, best_f1, history, global_step)
        if config.training.max_steps and global_step >= config.training.max_steps:
            break

    _load_model_weights(model, paths["best"])
    test = evaluate_model(model, test_loader, device)
    efficiency = measure_efficiency(model, test_loader, device, config.training.inference_repeats, paths["best"])
    result = {
        "experiment_id": experiment_id,
        "role": "student",
        "depth": depth,
        "method": "baseline",
        "teacher_ref": None,
        "seed": config.runtime.seed,
        "config_hash": config.config_hash,
        "history": history,
        "best_validation_f1_macro": best_f1,
        "test": test,
        "efficiency": efficiency,
        "artifacts": {"best_checkpoint": str(paths["best"]), "last_checkpoint": str(paths["last"])},
    }
    _write_result(config.paths, experiment_id, config.runtime.seed, result, history)
    return result


def build_baseline_model(config: ExperimentConfig, depth: int):
    """Build BERT with only encoder depth changed from the pretrained config."""
    baseline_config = AutoConfig.from_pretrained(
        config.model["pretrained_name"], revision=config.data.tokenizer_revision
    )
    baseline_config.num_hidden_layers = depth
    baseline_config.num_labels = config.model["num_labels"]
    baseline_config.id2label = dict(enumerate(LABEL_NAMES))
    baseline_config.label2id = {name: index for index, name in enumerate(LABEL_NAMES)}
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model["pretrained_name"],
        revision=config.data.tokenizer_revision,
        config=baseline_config,
        ignore_mismatched_sizes=False,
    )
    if model.config.num_hidden_layers != depth:
        raise ValueError(f"expected {depth} BERT layers, got {model.config.num_hidden_layers}")
    return model


def _resume(config, model, optimizer, scheduler, paths, device):
    state_path = paths["last"] / "training_state.pt"
    if not config.runtime.resume or not state_path.exists():
        return [], 0, float("-inf"), 0
    _load_model_weights(model, paths["last"])
    state = torch.load(state_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state.get("history", []), state["epoch"], state.get("best_f1", float("-inf")), state.get("global_step", 0)


def _write_result(paths, experiment_id, seed, result, history):
    json_path = paths.metrics_json(experiment_id, seed)
    csv_path = paths.metrics_csv(experiment_id, seed)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in history for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
