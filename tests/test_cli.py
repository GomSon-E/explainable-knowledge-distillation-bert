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
