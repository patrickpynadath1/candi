import json
import pytest
from examples.tune_distillation import build


@pytest.mark.parametrize("partial_params", [True, False])
def test_polling_retries_partially_written_ray_metadata(tmp_path, partial_params):
    trial = tmp_path / "run_job_test"
    trial.mkdir()
    (trial / "params.json").write_text(
        "{"
        if partial_params
        else json.dumps(
            {"name": "fewstep-3000-alpha-k4-text8-seed-1", "stage": "train"}
        )
    )
    (trial / "result.json").write_text("{")
    assert build(tmp_path, expected=None) == []
    with pytest.raises(json.JSONDecodeError):
        build(tmp_path, expected=1)
