import os
from pathlib import Path

import pytest

from explainable_kd.common.config import load_config
from explainable_kd.data.pipeline import run_smoke_test


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_SMOKE") != "1",
    reason="set RUN_NETWORK_SMOKE=1 to download TREC-6 and the BERT tokenizer",
)
def test_real_trec_batch_reaches_bert_tensor_inputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs/base.yaml",
        root / "configs/smoke.yaml",
        artifact_root=tmp_path,
    )

    result = run_smoke_test(config, batch_size=4)

    assert result["status"] == "passed"
    assert result["tokenizer"] == "google-bert/bert-base-cased"
    assert result["batch_size"] == 4
    assert result["sequence_length"] == 10
    assert result["tensor_shapes"]["input_ids"] == [4, 10]
