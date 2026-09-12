"""Thin Colab-friendly CLI for common experiment infrastructure."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from explainable_kd.common.config import load_config, load_experiment_registry
from explainable_kd.common.runtime import describe_device, resolve_device
from explainable_kd.common.seed import seed_everything
from explainable_kd.data.pipeline import prepare_dataset, run_smoke_test
from explainable_kd.training.teacher import train_teacher
from explainable_kd.training.baseline import train_baselines
from explainable_kd.training.kd import train_kd
from explainable_kd.training.ig_kd import train_ig_kd
from explainable_kd.xai.ig import extract_ig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    device = subparsers.add_parser("device", help="print CUDA and device details")
    device.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    listing = subparsers.add_parser(
        "list-experiments", help="validate and list the 13 experiment IDs"
    )
    listing.add_argument("--registry", default="configs/experiments.yaml")

    prepare = subparsers.add_parser("prepare-data", help="prepare and save TREC-6")
    _add_config_arguments(prepare)

    smoke = subparsers.add_parser(
        "smoke-test", help="create one real tokenized BERT input batch"
    )
    _add_config_arguments(smoke)
    smoke.add_argument("--batch-size", type=int, default=4)
    teacher = subparsers.add_parser("train-teacher", help="fine-tune the fixed 12-layer Teacher")
    _add_config_arguments(teacher)
    baselines = subparsers.add_parser("train-baselines", help="train 10-, 8-, and 6-layer labels-only baselines")
    _add_config_arguments(baselines)
    kd = subparsers.add_parser("train-kd", help="train 10-, 8-, and 6-layer students with standard KD")
    _add_config_arguments(kd)
    ig = subparsers.add_parser("extract-ig", help="extract and visualize Integrated Gradients")
    _add_config_arguments(ig)
    ig.add_argument("--max-examples", type=int)
    ig_kd = subparsers.add_parser("train-ig-kd", help="train 10-, 8-, and 6-layer IG KD students")
    _add_config_arguments(ig_kd)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    prepare_runner: Callable[[Any], dict[str, Any]] | None = None,
    smoke_runner: Callable[[Any, int], dict[str, Any]] | None = None,
    train_runner: Callable[[Any], dict[str, Any]] | None = None,
    baseline_runner: Callable[[Any], dict[str, Any]] | None = None,
    kd_runner: Callable[[Any], dict[str, Any]] | None = None,
    ig_runner: Callable[[Any, int], dict[str, Any]] | None = None,
    ig_kd_runner: Callable[[Any], dict[str, Any]] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list-experiments":
        for spec in load_experiment_registry(args.registry):
            print(spec.experiment_id)
        return 0

    if args.command == "device":
        _print_device(args.device)
        return 0

    config = load_config(
        args.config,
        args.overlay,
        artifact_root=args.artifact_root,
    )
    _print_device(config.runtime.device)
    seed_everything(config.runtime.seed, config.runtime.deterministic)

    if args.command == "prepare-data":
        if prepare_runner is None:
            prepared = prepare_dataset(config)
            result = {
                "data_fingerprint": prepared.manifest.data_fingerprint,
                "split_counts": dict(prepared.manifest.split_counts),
                "artifact_root": str(config.paths.root),
            }
        else:
            result = prepare_runner(config)
    elif args.command == "smoke-test":
        runner = smoke_runner or run_smoke_test
        result = runner(config, args.batch_size)
    elif args.command == "train-teacher":
        result = (train_runner or train_teacher)(config)
    elif args.command == "train-baselines":
        result = (baseline_runner or train_baselines)(config)
    elif args.command == "train-kd":
        result = (kd_runner or train_kd)(config)
    elif args.command == "extract-ig":
        max_examples = args.max_examples or config.xai.max_examples
        result = (ig_runner or extract_ig)(config, max_examples)
    elif args.command == "train-ig-kd":
        result = (ig_kd_runner or train_ig_kd)(config)
    else:
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--artifact-root", type=Path)


def _print_device(requested: str) -> None:
    summary = describe_device(resolve_device(requested))
    print(f"CUDA available: {summary['cuda_available']}")
    print(f"Device: {summary['device']}")
    if "cuda_device_name" in summary:
        print(f"CUDA device name: {summary['cuda_device_name']}")


if __name__ == "__main__":
    raise SystemExit(main())
