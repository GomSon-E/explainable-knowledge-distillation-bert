import json

from explainable_kd.cli import main


def test_device_command_prints_cuda_availability(capsys):
    exit_code = main(["device"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "CUDA available:" in output
    assert "Device:" in output


def test_list_experiments_prints_all_canonical_ids(capsys):
    exit_code = main(["list-experiments", "--registry", "configs/experiments.yaml"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert len([line for line in output.splitlines() if line]) == 13
    assert "teacher_d12_supervised" in output
    assert "student_d6_lrp_kd" in output


def test_prepare_data_can_use_an_injected_runner(tmp_path, capsys):
    def runner(config):
        return {
            "data_fingerprint": "abc",
            "split_counts": {"train": 32, "validation": 16, "test": 16},
            "artifact_root": str(config.paths.root),
        }

    exit_code = main(
        [
            "prepare-data",
            "--config",
            "configs/base.yaml",
            "--overlay",
            "configs/smoke.yaml",
            "--artifact-root",
            str(tmp_path),
        ],
        prepare_runner=runner,
    )

    output = capsys.readouterr().out
    payload = json.loads(output[output.index("{") :])
    assert exit_code == 0
    assert payload["data_fingerprint"] == "abc"
    assert payload["artifact_root"] == str(tmp_path)


def test_train_teacher_routes_smoke_and_full_modes(tmp_path, capsys):
    calls = []

    def runner(config):
        calls.append(config.runtime.smoke_test)
        return {"status": "passed", "mode": "smoke" if config.runtime.smoke_test else "full"}

    exit_code = main(
        [
            "train-teacher",
            "--config",
            "configs/base.yaml",
            "--overlay",
            "configs/smoke.yaml",
            "--artifact-root",
            str(tmp_path),
        ],
        train_runner=runner,
    )
    smoke_output = capsys.readouterr().out

    assert exit_code == 0
    assert calls == [True]
    assert json.loads(smoke_output[smoke_output.index("{") :])["mode"] == "smoke"

    exit_code = main(
        [
            "train-teacher",
            "--config",
            "configs/base.yaml",
            "--artifact-root",
            str(tmp_path),
        ],
        train_runner=runner,
    )
    full_output = capsys.readouterr().out

    assert exit_code == 0
    assert calls == [True, False]
    assert json.loads(full_output[full_output.index("{") :])["mode"] == "full"
