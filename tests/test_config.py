from pathlib import Path

import pytest
import yaml

from explainable_kd.common.config import (
    get_experiment,
    load_config,
    load_experiment_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def test_load_config_merges_smoke_overlay_and_artifact_override(tmp_path):
    config = load_config(
        ROOT / "configs/base.yaml",
        ROOT / "configs/smoke.yaml",
        artifact_root=tmp_path / "drive-artifacts",
    )

    assert config.runtime.seed == 42
    assert config.runtime.smoke_test is True
    assert config.data.dataset_name == "lukasgarbas/trec"
    assert config.data.tokenizer_name == "google-bert/bert-base-cased"
    assert config.data.max_length == 512
    assert config.data.max_train_samples == 32
    assert config.training.epochs == 1
    assert config.training.max_steps == 2
    assert config.paths.root == (tmp_path / "drive-artifacts").resolve()
    assert len(config.config_hash) == 64


def test_registry_contains_exactly_the_canonical_thirteen_experiments():
    registry = load_experiment_registry(ROOT / "configs/experiments.yaml")

    assert len(registry) == 13
    assert len({spec.experiment_id for spec in registry}) == 13
    assert get_experiment(
        "student_d8_ig_kd", ROOT / "configs/experiments.yaml"
    ).teacher_ref == "teacher_d12_supervised"


def test_registry_rejects_duplicate_ids(tmp_path):
    registry_path = tmp_path / "experiments.yaml"
    entry = {
        "experiment_id": "teacher_d12_supervised",
        "role": "teacher",
        "depth": 12,
        "method": "supervised",
        "teacher_ref": None,
    }
    registry_path.write_text(
        yaml.safe_dump({"teacher": [entry], "students": [entry]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate experiment_id"):
        load_experiment_registry(registry_path)
