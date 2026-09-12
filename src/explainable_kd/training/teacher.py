"""12-layer supervised Teacher fine-tuning for the Colab GPU runner."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from explainable_kd.common.checkpoint import ArtifactPaths
from explainable_kd.common.config import ExperimentConfig
from explainable_kd.common.metrics import classification_metrics
from explainable_kd.common.runtime import describe_device, resolve_device
from explainable_kd.data.pipeline import prepare_dataset

TEACHER_ID = "teacher_d12_supervised"
LABEL_NAMES = ("ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM")


def teacher_experiment_id(config: ExperimentConfig) -> str:
    del config
    return TEACHER_ID


def train_teacher(config: ExperimentConfig, *, model=None, tokenizer=None, prepared_data=None) -> dict[str, Any]:
    """Fine-tune, resume, evaluate, and persist the canonical Teacher."""
    device = resolve_device(config.runtime.device)
    config.paths.ensure_base_directories()
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.model["pretrained_name"], revision=config.data.tokenizer_revision)
    prepared_data = prepared_data or prepare_dataset(config, tokenizer=tokenizer)
    model = model or _load_model(config)
    model.to(device)
    train_loader, validation_loader, test_loader = _loaders(prepared_data.dataset, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // config.training.gradient_accumulation_steps)
    total_steps = config.training.max_steps or config.training.epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.training.warmup_ratio), max(1, total_steps))
    paths = _artifact_paths(config.paths, config.runtime.seed)
    history, start_epoch, best_f1, global_step = _resume_if_available(config, model, optimizer, scheduler, paths, device)

    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        train_loss, steps_seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch = _to_device(batch, device)
            loss = _forward_loss(model, batch) / config.training.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
            train_loss += loss.item() * config.training.gradient_accumulation_steps
            steps_seen = step + 1
            if config.training.max_steps and global_step >= config.training.max_steps:
                break
        validation = evaluate_model(model, validation_loader, device)
        record = {"epoch": epoch + 1, "train_loss": train_loss / max(1, steps_seen), **validation}
        history.append(record)
        _save_checkpoint(model, tokenizer, optimizer, scheduler, paths["last"], epoch + 1, best_f1, history, global_step)
        if validation["f1_macro"] > best_f1:
            best_f1 = validation["f1_macro"]
            _save_checkpoint(model, tokenizer, None, None, paths["best"], epoch + 1, best_f1, history, global_step)
        if config.training.max_steps and global_step >= config.training.max_steps:
            break

    _load_model_weights(model, paths["best"])
    test = evaluate_model(model, test_loader, device)
    efficiency = measure_efficiency(model, test_loader, device, config.training.inference_repeats, paths["best"])
    result = {"experiment_id": TEACHER_ID, "role": "teacher", "depth": 12, "method": "supervised", "seed": config.runtime.seed, "config_hash": config.config_hash, "device": describe_device(device), "history": history, "best_validation_f1_macro": best_f1, "test": test, "efficiency": efficiency, "artifacts": {"best_checkpoint": str(paths["best"]), "last_checkpoint": str(paths["last"])}}
    _write_result(config.paths, config.runtime.seed, result, history)
    return result


def evaluate_model(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval(); loss_fn = CrossEntropyLoss(); losses, labels, predictions = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device); outputs = model(**batch)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            labels.extend(batch["labels"].cpu().tolist()); predictions.extend(logits.argmax(-1).cpu().tolist()); losses.append(loss_fn(logits, batch["labels"]).item())
    return {"loss": sum(losses) / max(1, len(losses)), **classification_metrics(labels, predictions)}


def measure_efficiency(model, loader, device, repeats: int, checkpoint: Path) -> dict[str, float | int | str]:
    model.eval(); batch = _to_device(next(iter(loader)), device)
    with torch.no_grad():
        for _ in range(min(3, max(1, repeats))): model(**batch)
        _sync(device); started = time.perf_counter()
        for _ in range(max(1, repeats)): model(**batch)
        _sync(device)
    return {"parameter_count": int(sum(p.numel() for p in model.parameters())), "model_file_size_mb": round(_directory_size(checkpoint) / 1024**2, 4), "classification_inference_ms_per_batch": (time.perf_counter() - started) * 1000 / max(1, repeats), "timed_batch_size": int(batch["input_ids"].shape[0]), "latency_scope": "classification_only"}


def _loaders(dataset, config):
    loaders = []
    for name in ("train", "validation", "test"):
        loaders.append(DataLoader(dataset[name], batch_size=config.training.batch_size, shuffle=name == "train", collate_fn=_collate_batch))
    return loaders


def _collate_batch(rows):
    keys = ("input_ids", "attention_mask", "token_type_ids")
    batch = {key: torch.tensor([row[key] for row in rows], dtype=torch.long) for key in keys if key in rows[0]}
    batch["labels"] = torch.tensor([row["label"] for row in rows], dtype=torch.long)
    return batch


def _forward_loss(model, batch):
    outputs = model(**batch)
    if hasattr(outputs, "loss") and outputs.loss is not None: return outputs.loss
    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
    return CrossEntropyLoss()(logits, batch["labels"])


def _to_device(batch: Mapping[str, torch.Tensor], device): return {key: value.to(device) for key, value in batch.items()}


def _load_model(config):
    model = AutoModelForSequenceClassification.from_pretrained(config.model["pretrained_name"], revision=config.data.tokenizer_revision, num_labels=config.model["num_labels"], id2label=dict(enumerate(LABEL_NAMES)), label2id={name: i for i, name in enumerate(LABEL_NAMES)})
    if getattr(model.config, "num_hidden_layers", 12) != 12: raise ValueError("Teacher must use the 12-layer BERT architecture")
    return model


def _artifact_paths(paths: ArtifactPaths, seed: int): return {slot: paths.checkpoint(TEACHER_ID, seed, slot) for slot in ("best", "last")}


def _save_checkpoint(model, tokenizer, optimizer, scheduler, path, epoch, best_f1, history, global_step):
    path.mkdir(parents=True, exist_ok=True); model.save_pretrained(path); tokenizer.save_pretrained(path)
    if optimizer is not None: torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "best_f1": best_f1, "history": history, "global_step": global_step}, path / "training_state.pt")


def _resume_if_available(config, model, optimizer, scheduler, paths, device):
    state_path = paths["last"] / "training_state.pt"
    if not config.runtime.resume or not state_path.exists(): return [], 0, float("-inf"), 0
    _load_model_weights(model, paths["last"]); state = torch.load(state_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
    return state.get("history", []), state["epoch"], state.get("best_f1", float("-inf")), state.get("global_step", 0)


def _load_model_weights(model, path):
    if (path / "pytorch_model.bin").exists(): model.load_state_dict(torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=False))
    elif (path / "model.safetensors").exists():
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(path / "model.safetensors")))


def _write_result(paths, seed, result, history):
    json_path = paths.metrics_json(TEACHER_ID, seed)
    csv_path = paths.metrics_csv(TEACHER_ID, seed)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in history for key in row}); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(history)


def _directory_size(path: Path) -> int: return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _sync(device):
    if device.type == "cuda": torch.cuda.synchronize(device)
