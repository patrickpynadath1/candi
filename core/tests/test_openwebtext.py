"""Independent checks against legacy OpenWebText metric and packing equations."""

import math
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from core.tests.test_mdlm import config
from core.data.openwebtext import pack_documents
from core.tasks.metrics.generative import CausalReferenceScorer, token_entropy
from core.tasks.openwebtext_metrics import OpenWebTextMetrics


class Tokenizer:
    pad_token = "pad"
    pad_token_id = 0
    eos_token = "eos"
    eos_token_id = 4

    def __call__(self, texts, **kwargs):
        rows = [
            [int(x) for x in text.split()][: kwargs["max_length"]] for text in texts
        ]
        size = max(map(len, rows))
        return {
            "input_ids": torch.tensor(
                [r + [self.pad_token_id] * (size - len(r)) for r in rows]
            ),
            "attention_mask": torch.tensor(
                [[1] * len(r) + [0] * (size - len(r)) for r in rows]
            ),
        }

    def batch_decode(self, samples):
        return [" ".join(str(int(x)) for x in sample) for sample in samples]


class Model(torch.nn.Module):
    def forward(self, tokens, attention_mask):
        # Different target losses ensure weighted corpus aggregation is tested.
        logits = torch.arange(5).float().expand(*tokens.shape, 5).clone()
        return (logits,)


def test_reference_scoring_matches_original_math():
    tokenizer = Tokenizer()
    scorer = CausalReferenceScorer(
        model=Model(), tokenizer=tokenizer, batch_size=1, context_size=4
    )
    texts = ["1 2 4 3 1 2 3", "2 1 3", "4 1 2 3 1 2 3"]
    actual = scorer.score(texts, 7)
    encoded = tokenizer(texts, max_length=7)
    expected_nll, expected_count = 0, 0
    for row, attention in zip(encoded["input_ids"], encoded["attention_mask"]):
        row_nll, row_count = 0, 0
        for chunk, mask in zip(row.split(4), attention.split(4)):
            if len(chunk) < 2:
                continue
            logits = Model()(chunk[None], mask[None])[0].transpose(-1, -2)
            loss = F.cross_entropy(logits[..., :-1], chunk[None, 1:], reduction="none")[
                0
            ]
            eos_seen = False
            valid = []
            for i, token in enumerate(chunk):
                include = not eos_seen
                eos_seen |= token.item() == 4
                if i:
                    valid.append(include and bool(mask[i]))
            selected = loss[torch.tensor(valid)]
            row_nll += float(selected.sum())
            row_count += len(selected)
        expected_nll += row_nll
        expected_count += row_count
    assert math.isclose(
        sum(x.negative_log_likelihood for x in actual), expected_nll, rel_tol=1e-7
    )
    assert sum(x.scored_tokens for x in actual) == expected_count
    assert (
        actual[2].scored_tokens == 2
    )  # EOS at chunk start suppresses that chunk only.


def test_metrics_use_corpus_weighting_and_empirical_entropy():
    tokenizer = Tokenizer()
    scorer = CausalReferenceScorer(model=Model(), tokenizer=tokenizer, context_size=4)
    accumulator = OpenWebTextMetrics(tokenizer, scorer, 4)
    samples = torch.tensor([[1, 2, 4, 3], [1, 1, 2, 3]])
    accumulator.update(samples, [None] * 2, [5, 8])
    result = accumulator.compute()
    assert result.scored_tokens == 5
    assert result.generative_perplexity == math.exp(result.negative_log_likelihood / 5)
    assert result.entropy == sum(token_entropy(row) for row in samples) / 2
    assert math.isclose(
        token_entropy(torch.tensor([1, 1, 2, 2])), math.log(2), rel_tol=1e-7
    )


def test_document_packing_and_config_are_self_contained():
    assert pack_documents({"input_ids": [[1, 2, 9], [3, 4, 9]]}, 5, 9, 9) == {
        "input_ids": [[9, 1, 2, 9, 9], [9, 3, 4, 9, 9]]
    }
    cfg = config("task=openwebtext")
    task = instantiate(cfg.task)
    assert task.vocab_size == 50257
    assert cfg.token_space.mask_index == 50257
    assert instantiate(cfg.token_space.model_vocab_size) == 50258
    assert task.primary_metric.name == "generative_perplexity"
    assert len(task.evaluation_dataset("validation")) == 128
    prompts, fixed = task.prepare_generation([0, 1])
    assert prompts.shape == (2, 1024) and not fixed.any()


def test_eos_padding_shared_id_and_truncation():
    tokenizer = Tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    scorer = CausalReferenceScorer(model=Model(), tokenizer=tokenizer, context_size=4)
    scores = scorer.score(["1 2 4 3", "1 2"], 3)
    assert [s.scored_tokens for s in scores] == [2, 1]
    assert scorer.score(["4"], 3)[0].scored_tokens == 0


def test_local_documents_train_through_task(tmp_path):
    import json
    from omegaconf import OmegaConf
    from core.tasks.openwebtext import OpenWebTextTask

    class TrainingTokenizer(Tokenizer):
        bos_token_id = 4

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[int(x) for x in t.split()] for t in texts]}

    path = tmp_path / "text.jsonl"
    path.write_text("\n".join(json.dumps({"text": "1 2 3 1 2 3"}) for _ in range(9)))
    data = OmegaConf.create(
        {
            "cache_dir": str(tmp_path / "cache"),
            "num_proc": 1,
            "files_train": [str(path)],
            "files_validation": [str(path)],
        }
    )
    task = OpenWebTextTask(data, scorer=None, length=5)
    task._tokenizer = TrainingTokenizer()
    task.prepare_data()
    dataset = task.training_dataset()
    assert len(dataset) == 20  # pack8 remainder discarded independently per group
    batch = task.collate_training([dataset[0], dataset[1]])
    assert batch["input_ids"].shape == (2, 5)
    assert batch["loss_mask"].all() and not batch["prompt_mask"].any()


def test_openwebtext_trains_checkpoints_and_evaluates(tmp_path, monkeypatch):
    import json
    from core.training.runner import run
    from core.tasks.openwebtext import OpenWebTextTask

    class TrainingTokenizer(Tokenizer):
        bos_token_id = 4

        def __call__(self, texts, **kwargs):
            if kwargs.get("add_special_tokens") is False:
                return {"input_ids": [[int(x) for x in t.split()] for t in texts]}
            return super().__call__(texts, **kwargs)

    monkeypatch.setattr(
        OpenWebTextTask, "tokenizer", property(lambda self: TrainingTokenizer())
    )

    def load_reference(self):
        self.model, self.tokenizer = Model(), Tokenizer()

    monkeypatch.setattr(CausalReferenceScorer, "_load", load_reference)
    source = tmp_path / "documents.jsonl"
    source.write_text("\n".join(json.dumps({"text": "1 2 3 1 2 3"}) for _ in range(9)))
    settings = config(
        "task=openwebtext",
        "task.length=5",
        "task.vocab_size=5",
        "task.num_eval_samples=2",
        f"data.files_train=[{source}]",
        f"data.files_validation=[{source}]",
        f"data.cache_dir={tmp_path / 'cache'}",
    )
    settings.callback.every_n_steps = 2
    settings.output_dir = str(tmp_path / "train")
    settings.trainer.enable_progress_bar = False
    run(settings)
    assert (tmp_path / "train/task_callback/step_00000002.json").exists()
    settings.mode = "evaluate"
    settings.evaluation.checkpoint = str(tmp_path / "train/checkpoints/last.ckpt")
    settings.output_dir = str(tmp_path / "evaluate")
    run(settings)
    result = json.loads((tmp_path / "evaluate/results.json").read_text())
    assert result["num_total"] == 2
    assert result["primary_metric"]["name"] == "generative_perplexity"
    assert "entropy" in result and "scored_tokens" in result
    assert len(result["records"]) == 2


def test_shared_bos_eos_scores_document_but_not_text_after_eos():
    tokenizer = Tokenizer()
    tokenizer.bos_token_id = tokenizer.eos_token_id
    scorer = CausalReferenceScorer(model=Model(), tokenizer=tokenizer, context_size=4)
    scores = scorer.score(["4 1 2 4 4 1 2", "1 2 4 3"], 7)
    # Initial BOS is context; terminal EOS remains included. An EOS at the start
    # of the second chunk is not a document BOS and still ends that chunk.
    assert [s.scored_tokens for s in scores] == [3, 2]
    assert scores[0].negative_log_likelihood > 0
    scorer.ignore_initial_bos = False
    assert scorer.score(["4 1 2 4"], 4)[0].scored_tokens == 0
