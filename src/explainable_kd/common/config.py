"""Typed YAML configuration and canonical experiment registry validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from explainable_kd.common.checkpoint import ArtifactPaths


CANONICAL_TEACHER_ID = "teacher_d12_supervised"
Depth = Literal[6, 8, 10, 12]
Method = Literal["supervised", "baseline", "kd", "ig_kd", "lrp_kd"]
Role = Literal["teacher", "student"]


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    seed: int
    deterministic: bool
    resume: bool
    smoke_test: bool


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str
    dataset_revision: str
    tokenizer_name: str
    tokenizer_revision: str
    max_length: int
    short_input_policy: str
    validation_size: float
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 3
    max_steps: int | None = None
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    inference_repeats: int = 20


@dataclass(frozen=True)
class DistillationConfig:
    temperature: float = 2.0
    task_loss_weight: float = 1.0
    kd_loss_weight: float = 1.0
    ig_loss_weight: float = 1.0


@dataclass(frozen=True)
class XAIConfig:
    max_examples: int = 1
    ig_steps: int = 16


@dataclass(frozen=True)
class ExperimentConfig:
    project_name: str
    runtime: RuntimeConfig
    data: DataConfig
    model: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    training: TrainingConfig
    distillation: DistillationConfig
    xai: XAIConfig
    paths: ArtifactPaths
    config_hash: str


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    role: Role
    depth: Depth
    method: Method
    teacher_ref: str | None


def load_config(
    base_path: str | Path,
    overlay_path: str | Path | None = None,
    *,
    artifact_root: str | Path | None = None,
) -> ExperimentConfig:
    """Load base YAML, apply an optional overlay, and resolve artifact paths."""

    base_path = Path(base_path).expanduser().resolve()
    raw = _read_yaml(base_path)
    if overlay_path is not None:
        raw = _deep_merge(raw, _read_yaml(Path(overlay_path).expanduser().resolve()))

    project_root = base_path.parent.parent
    configured_root = artifact_root or raw["project"]["artifact_root"]
    configured_root = Path(configured_root).expanduser()
    if not configured_root.is_absolute():
        configured_root = project_root / configured_root
    paths = ArtifactPaths(configured_root)

    runtime = RuntimeConfig(**raw["runtime"])
    data_values = {
        key: value
        for key, value in raw["data"].items()
        if key in DataConfig.__dataclass_fields__
    }
    data = DataConfig(**data_values)
    training = TrainingConfig(**raw.get("training", {}))
    distillation = DistillationConfig(**raw.get("distillation", {}))
    xai_values = {
        key: value
        for key, value in raw.get("xai", {}).items()
        if key in XAIConfig.__dataclass_fields__
    }
    xai = XAIConfig(**xai_values)
    _validate_config(runtime, data)

    hash_input = dict(raw)
    hash_input["project"] = dict(raw["project"], artifact_root=str(paths.root))
    config_hash = hashlib.sha256(
        json.dumps(hash_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ExperimentConfig(
        project_name=raw["project"]["name"],
        runtime=runtime,
        data=data,
        model=raw["model"],
        evaluation=raw["evaluation"],
        training=training,
        distillation=distillation,
        xai=xai,
        paths=paths,
        config_hash=config_hash,
    )


def load_experiment_registry(path: str | Path) -> tuple[ExperimentSpec, ...]:
    """Read and validate the immutable 13-condition experiment registry."""

    raw = _read_yaml(Path(path))
    specs = tuple(
        ExperimentSpec(**item) for item in [*raw.get("teacher", []), *raw.get("students", [])]
    )
    ids = [spec.experiment_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate experiment_id in registry")
    if len(specs) != 13:
        raise ValueError(f"registry must contain 13 experiments, found {len(specs)}")

    teachers = [spec for spec in specs if spec.role == "teacher"]
    if teachers != [
        ExperimentSpec(CANONICAL_TEACHER_ID, "teacher", 12, "supervised", None)
    ]:
        raise ValueError("registry must contain only the canonical 12-layer Teacher")

    expected_students = {
        (depth, method)
        for depth in (10, 8, 6)
        for method in ("baseline", "kd", "ig_kd", "lrp_kd")
    }
    actual_students = {(spec.depth, spec.method) for spec in specs if spec.role == "student"}
    if actual_students != expected_students:
        raise ValueError("registry must contain baseline, KD, IG KD, and LRP KD per depth")
    for spec in specs:
        expected_teacher = None if spec.method in {"supervised", "baseline"} else CANONICAL_TEACHER_ID
        if spec.teacher_ref != expected_teacher:
            raise ValueError(f"invalid teacher_ref for {spec.experiment_id}")
    return specs


def get_experiment(experiment_id: str, registry_path: str | Path) -> ExperimentSpec:
    for spec in load_experiment_registry(registry_path):
        if spec.experiment_id == experiment_id:
            return spec
    raise KeyError(f"unknown experiment_id: {experiment_id}")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_config(runtime: RuntimeConfig, data: DataConfig) -> None:
    if runtime.seed < 0:
        raise ValueError("seed must be non-negative")
    if data.max_length != 512:
        raise ValueError("project protocol requires data.max_length=512")
    if data.short_input_policy not in {"filter", "truncate"}:
        raise ValueError("short_input_policy must be 'filter' or 'truncate'")
    if not 0 < data.validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
