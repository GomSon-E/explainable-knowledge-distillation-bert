"""Collision-safe artifact paths shared by every experiment stage."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_SAFE_ID = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class ArtifactPaths:
    """Resolve every generated artifact below one configurable root."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def split_manifest(self) -> Path:
        return self.data / "split_manifest.json"

    def processed(self, fingerprint: str) -> Path:
        self._validate_component(fingerprint, "data fingerprint")
        return self.data / "processed" / fingerprint

    def checkpoint(self, experiment_id: str, seed: int, slot: str) -> Path:
        self._validate_run(experiment_id, seed)
        if slot not in {"best", "last"}:
            raise ValueError("checkpoint slot must be 'best' or 'last'")
        return self.root / "checkpoints" / experiment_id / f"seed_{seed}" / slot

    def metrics_json(self, experiment_id: str, seed: int) -> Path:
        return self._metrics_dir(experiment_id, seed) / "metrics.json"

    def metrics_csv(self, experiment_id: str, seed: int) -> Path:
        return self._metrics_dir(experiment_id, seed) / "metrics.csv"

    @property
    def plots(self) -> Path:
        return self.root / "figures"

    def ensure_base_directories(self) -> None:
        for path in (
            self.data,
            self.root / "checkpoints",
            self.root / "metrics",
            self.root / "logs",
            self.root / "figures",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _metrics_dir(self, experiment_id: str, seed: int) -> Path:
        self._validate_run(experiment_id, seed)
        return self.root / "metrics" / experiment_id / f"seed_{seed}"

    def _validate_run(self, experiment_id: str, seed: int) -> None:
        self._validate_component(experiment_id, "experiment ID")
        if seed < 0:
            raise ValueError("seed must be non-negative")

    @staticmethod
    def _validate_component(value: str, label: str) -> None:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError(f"unsafe {label}: {value!r}")
