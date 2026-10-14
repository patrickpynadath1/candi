import json
from dataclasses import asdict

import pytest
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from pathlib import Path

from core.tasks.tinystories import TinyStoriesTask
from core.tasks.tinystories_metrics import TinyStoriesMetrics
from core.data.tinystories import StoryByteTokenizer
from core.training.runner import run


def fixture_data(tmp_path):
    for split, names in [
        ("train", ["Amy", "Bob", "Cal", "Dan"]),
        ("validation", ["Eve", "Flo", "Gil", "Hal"]),
    ]:
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(
                json.dumps({"text": f"{name} saw a cat. It ran home."}) + "\n"
                for name in names
            )
        )
    return OmegaConf.create(
        dict(
            cache_dir=str(tmp_path / "cache"),
            dataset_name="roneneldan/TinyStories",
            revision="main",
            download=False,
            train_path=str(tmp_path / "train.jsonl"),
            validation_path=str(tmp_path / "validation.jsonl"),
            max_train_examples=4,
            max_validation_examples=4,
        )
    )


def test_masks_and_no_answer_leakage(tmp_path):
    task = TinyStoriesTask(fixture_data(tmp_path), length=40, prompt_bytes=4)
    task.prepare_data()
    example = task.validation_dataset()[0]
    changed = dict(example, input_ids=example["input_ids"][:5] + [19] * 7 + [2])
    for a, b in zip(
        task.prepare_generation([example]), task.prepare_generation([changed])
    ):
        assert torch.equal(a, b)
    batch = task.collate_training([example])
    assert batch["loss_mask"][batch["padding_mask"]].all()
    task.variable_length = True
    batch = task.collate_training([example])
    assert not batch["loss_mask"][batch["padding_mask"]].any()
    assert batch["loss_mask"][0, len(example["input_ids"]) - 1]


def test_metrics_eos_and_variable_length(tmp_path):
    task = TinyStoriesTask(fixture_data(tmp_path), length=40, prompt_bytes=4)
    example = task.validation_dataset()[0]
    metric = TinyStoriesMetrics(task.tokenizer)
    metric.update(
        [example["input_ids"] + [0] * 9],
        [example],
        [0],
        lengths=[len(example["input_ids"])],
    )
    result = metric.compute()
    assert result.termination_rate == 1
    assert result.invalid_special_rate == 0
    assert result.records[0].continuation == example["reference"]
    assert result.records[0].id.startswith("validation:")
    json.dumps(asdict(result))


def test_filters_truncation_and_deduplicates(tmp_path):
    data = fixture_data(tmp_path)
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"text": t})
            for t in ["a" * 100, "hello world", "hello world", "ok"]
        )
    )
    task = TinyStoriesTask(data, length=20, prompt_bytes=4)
    assert len(task.training_dataset()) == 1
    tokenizer = StoryByteTokenizer()
    assert tokenizer.decode(tokenizer.encode("café 🐈")) == "café 🐈"


@pytest.mark.parametrize("insertion", [False, True])
def test_runner_train_evaluate(tmp_path, insertion):
    data = fixture_data(tmp_path)
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        settings = compose(
            config_name="config",
            overrides=[
                "+experiment="
                + ("tinystories_insertion" if insertion else "tinystories"),
                "task.length=40",
                "task.prompt_bytes=4",
                "output_dir=/tmp/stories_smoke",
                "data.cache_dir=/tmp/stories_smoke_cache",
                *([] if insertion else ["sampler=ancestral"]),
            ],
        )
    settings.data = data
    settings.output_dir = str(tmp_path / "run")
    model = settings.model.backbone if insertion else settings.model
    model.hidden_size = 16
    model.cond_dim = 16
    model.n_blocks = 1
    model.n_heads = 2
    settings.trainer.accelerator = "cpu"
    settings.trainer.precision = "32-true"
    settings.trainer.accumulate_grad_batches = 1
    settings.trainer.num_sanity_val_steps = 0
    settings.trainer.val_check_interval = 2
    settings.loader.num_workers = 0
    settings.loader.pin_memory = False
    settings.evaluation.device = "cpu"
    settings.training.warmup_steps = 1
    settings.trainer.max_steps = 2
    settings.trainer.enable_progress_bar = False
    settings.callback.every_n_steps = 2
    settings.callback.subset_size = 2
    settings.checkpoint.every_n_steps = 2
    settings.loader.global_batch_size = 2
    settings.loader.eval_batch_size = 2
    settings.sampler.steps = 2
    run(settings)
    checkpoint = tmp_path / "run/checkpoints/last.ckpt"
    assert checkpoint.exists()
    settings.mode = "evaluate"
    settings.evaluation.checkpoint = str(checkpoint)
    settings.evaluation.max_examples = 2
    settings.output_dir = str(tmp_path / "eval")
    # Eval must work without preparing/loading training data.
    settings.data.train_path = "/does/not/exist"
    run(settings)
    result = json.loads((tmp_path / "eval/results.json").read_text())
    assert result["num_total"] == 2
    assert len(result["records"]) == 2


def test_metrics_malformed_bytes_and_no_eos(tmp_path):
    task = TinyStoriesTask(fixture_data(tmp_path), length=40, prompt_bytes=4)
    example = task.validation_dataset()[0]
    prefix = example["input_ids"][: example["prompt_length"]]
    metrics = TinyStoriesMetrics(task.tokenizer)
    metrics.update([prefix + [0, 259]], [example], [0])
    result = metrics.compute()
    assert result.termination_rate == 0
    assert result.invalid_utf8_rate == 1
    assert result.invalid_special_rate == 0.5
    assert result.average_length == 2


def test_train_validation_overlap_refused(tmp_path):
    data = fixture_data(tmp_path)
    data.validation_path = data.train_path
    with pytest.raises(ValueError, match="duplicate stories"):
        TinyStoriesTask(data, length=40, prompt_bytes=4).prepare_data()


def test_optional_bleu_perfect_reference(tmp_path):
    pytest.importorskip("sacrebleu")
    task = TinyStoriesTask(fixture_data(tmp_path), length=40, prompt_bytes=4)
    example = task.validation_dataset()[0]
    metric = TinyStoriesMetrics(task.tokenizer, calculate_bleu=True)
    metric.update([example["input_ids"]], [example], [0])
    result = metric.compute()
    assert result.corpus_bleu == pytest.approx(100.0)
    assert result.bleu_signature
