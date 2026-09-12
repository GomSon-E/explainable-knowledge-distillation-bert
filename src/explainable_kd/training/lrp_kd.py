"""LRP explanation distillation from the fixed 12-layer Teacher."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from explainable_kd.common.runtime import resolve_device
from explainable_kd.data.pipeline import prepare_dataset
from explainable_kd.training.baseline import BASELINE_DEPTHS, _resume, _write_result, baseline_experiment_id, build_baseline_model
from explainable_kd.training.kd import _evaluate_kd, _load_teacher, kd_loss
from explainable_kd.training.teacher import TEACHER_ID, _load_model_weights, _loaders, _save_checkpoint, _to_device, evaluate_model, measure_efficiency
from explainable_kd.xai.lrp import extract_lrp, lrp_token_relevance_tensor


def lrp_explanation_loss(student_scores: torch.Tensor, teacher_scores: torch.Tensor, valid_mask) -> torch.Tensor:
    student_scores = student_scores.reshape(-1)
    teacher_scores = teacher_scores.detach().reshape(-1)
    mask = torch.as_tensor(valid_mask, device=student_scores.device, dtype=torch.bool).reshape(-1)
    if not torch.any(mask):
        return student_scores.sum() * 0.0
    return torch.nn.functional.mse_loss(student_scores[mask], teacher_scores[mask])


def train_lrp_kd(config, *, depth=None, teacher=None, model=None, tokenizer=None, prepared_data=None) -> dict[str, Any]:
    if depth is not None:
        return train_lrp_kd_student(config, depth=depth, teacher=teacher, model=model, tokenizer=tokenizer, prepared_data=prepared_data)
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.data.tokenizer_name, revision=config.data.tokenizer_revision, use_fast=True)
    prepared_data = prepared_data or prepare_dataset(config, tokenizer=tokenizer)
    device = resolve_device(config.runtime.device)
    teacher = (teacher or _load_teacher(config)).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    results = {}
    for student_depth in BASELINE_DEPTHS:
        experiment_id = baseline_experiment_id(student_depth).replace("_baseline", "_lrp_kd")
        results[experiment_id] = train_lrp_kd_student(config, depth=student_depth, teacher=teacher, tokenizer=tokenizer, prepared_data=prepared_data)
    results["lrp_artifacts"] = extract_lrp(config, config.xai.max_examples, experiment_ids=(TEACHER_ID, "student_d10_lrp_kd", "student_d8_lrp_kd", "student_d6_lrp_kd"))
    return results


def train_lrp_kd_student(config, *, depth: int, teacher, model=None, tokenizer=None, prepared_data=None) -> dict[str, Any]:
    if depth not in BASELINE_DEPTHS:
        raise ValueError(f"LRP KD depth must be one of {BASELINE_DEPTHS}")
    experiment_id = baseline_experiment_id(depth).replace("_baseline", "_lrp_kd")
    device = resolve_device(config.runtime.device)
    config.paths.ensure_base_directories()
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.data.tokenizer_name, revision=config.data.tokenizer_revision, use_fast=True)
    prepared_data = prepared_data or prepare_dataset(config, tokenizer=tokenizer)
    model = model or build_baseline_model(config, depth)
    model.to(device); teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    train_loader, validation_loader, test_loader = _loaders(prepared_data.dataset, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // config.training.gradient_accumulation_steps)
    total_steps = config.training.max_steps or config.training.epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config.training.warmup_ratio), max(1, total_steps))
    paths = {slot: config.paths.checkpoint(experiment_id, config.runtime.seed, slot) for slot in ("best", "last")}
    history, start_epoch, best_f1, global_step = _resume(config, model, optimizer, scheduler, paths, device)
    for epoch in range(start_epoch, config.training.epochs):
        model.train(); task_total, kd_total, lrp_total, steps_seen = 0.0, 0.0, 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch = _to_device(batch, device)
            with torch.no_grad():
                teacher_logits = _logits(teacher, batch)
            student_logits = _logits(model, batch)
            task = CrossEntropyLoss()(student_logits, batch["labels"])
            distill = kd_loss(student_logits, teacher_logits, config.distillation.temperature)
            with torch.enable_grad():
                teacher_scores, valid_masks, _ = lrp_token_relevance_tensor(teacher, batch, batch["labels"], tokenizer, epsilon=config.xai.lrp_epsilon, create_graph=False)
                student_scores, _, _ = lrp_token_relevance_tensor(model, batch, batch["labels"], tokenizer, epsilon=config.xai.lrp_epsilon, create_graph=True)
            explanation = torch.stack([lrp_explanation_loss(student_scores[index], teacher_scores[index], valid_masks[index]) for index in range(student_scores.shape[0])]).mean()
            loss = (config.distillation.task_loss_weight * task + config.distillation.kd_loss_weight * distill + config.distillation.lrp_loss_weight * explanation) / config.training.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
            task_total += task.item(); kd_total += distill.item(); lrp_total += explanation.item(); steps_seen = step + 1
            if config.training.max_steps and global_step >= config.training.max_steps:
                break
        validation = _evaluate_kd(model, teacher, validation_loader, device, config)
        task_mean = task_total / max(1, steps_seen); kd_mean = kd_total / max(1, steps_seen); lrp_mean = lrp_total / max(1, steps_seen)
        record = {"epoch": epoch + 1, "train_task_loss": task_mean, "train_kd_loss": kd_mean, "train_lrp_loss": lrp_mean, "train_loss": config.distillation.task_loss_weight * task_mean + config.distillation.kd_loss_weight * kd_mean + config.distillation.lrp_loss_weight * lrp_mean, **validation}
        history.append(record)
        improved = validation["f1_macro"] > best_f1
        if improved: best_f1 = validation["f1_macro"]
        _save_checkpoint(model, tokenizer, optimizer, scheduler, paths["last"], epoch + 1, best_f1, history, global_step)
        if improved: _save_checkpoint(model, tokenizer, None, None, paths["best"], epoch + 1, best_f1, history, global_step)
        if config.training.max_steps and global_step >= config.training.max_steps:
            break
    _load_model_weights(model, paths["best"])
    test = evaluate_model(model, test_loader, device)
    efficiency = measure_efficiency(model, test_loader, device, config.training.inference_repeats, paths["best"])
    result = {"experiment_id": experiment_id, "role": "student", "depth": depth, "method": "lrp_kd", "teacher_ref": TEACHER_ID, "seed": config.runtime.seed, "config_hash": config.config_hash, "distillation": {"temperature": config.distillation.temperature, "task_loss_weight": config.distillation.task_loss_weight, "kd_loss_weight": config.distillation.kd_loss_weight, "lrp_loss_weight": config.distillation.lrp_loss_weight}, "lrp_rules": "BERT-compatible epsilon relevance; see explainable_kd.xai.lrp.LRP_RULES", "history": history, "best_validation_f1_macro": best_f1, "test": test, "efficiency": efficiency, "artifacts": {"best_checkpoint": str(paths["best"]), "last_checkpoint": str(paths["last"])} }
    _write_result(config.paths, experiment_id, config.runtime.seed, result, history)
    return result


def _logits(model, batch):
    output = model(**batch)
    return output.logits if hasattr(output, "logits") else output["logits"]
