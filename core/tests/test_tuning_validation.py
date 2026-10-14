import pytest
from core.sweeps.nested_tuning import validate_inference as validate
from core.evaluation.reporting import metric_value


def protocol():
    return dict(
        tuning_indices=[1, 2],
        test_indices=[3, 4],
        seeds=[1],
        test_seeds=[2],
        nfe=128,
        trials=2,
        direction="maximize",
        budget_parameter="steps",
        space={},
        score={"kind": "metric", "metric": "accuracy"},
    )


def test_partition_leakage_rejected():
    spec = protocol()
    spec["test_indices"] = [2, 3]
    with pytest.raises(ValueError, match="disjoint"):
        validate(spec)


@pytest.mark.parametrize(
    "key", ["sampler.steps", "sampler.max_steps", "optimizer.lr", "task.length"]
)
def test_search_cannot_change_training_or_nfe(key):
    spec = protocol()
    spec["space"] = {key: [1, 2]}
    with pytest.raises(ValueError, match="Not a tunable"):
        validate(spec)


def test_word_metric_uses_length_label_not_list_position():
    result = {
        "generated": {
            "by_min_length": [
                {
                    "min_length": 5,
                    "aggregate": {"mean_per_sample": {"unique_valid": 4}},
                },
                {
                    "min_length": 6,
                    "aggregate": {"mean_per_sample": {"unique_valid": 2}},
                },
            ]
        }
    }
    assert (
        metric_value(
            result, "generated/by_min_length/5/aggregate/mean_per_sample/unique_valid"
        )
        == 4
    )
