from explainable_kd.common.checkpoint import ArtifactPaths


def test_artifact_paths_are_isolated_by_experiment_and_seed(tmp_path):
    paths = ArtifactPaths(tmp_path)

    assert paths.split_manifest == tmp_path / "data/split_manifest.json"
    assert paths.processed("abc") == tmp_path / "data/processed/abc"
    assert paths.checkpoint("student_d8_kd", 42, "best") == (
        tmp_path / "checkpoints/student_d8_kd/seed_42/best"
    )
    assert paths.metrics_json("student_d8_kd", 42) == (
        tmp_path / "metrics/student_d8_kd/seed_42/metrics.json"
    )
    assert paths.metrics_csv("student_d8_kd", 42) == (
        tmp_path / "metrics/student_d8_kd/seed_42/metrics.csv"
    )
    assert paths.plots == tmp_path / "figures"


def test_artifact_paths_reject_path_traversal(tmp_path):
    paths = ArtifactPaths(tmp_path)

    for invalid_id in ("../other", "student/d8", ""):
        try:
            paths.checkpoint(invalid_id, 42, "best")
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe experiment ID: {invalid_id}")
