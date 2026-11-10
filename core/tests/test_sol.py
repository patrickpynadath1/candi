"""SOL: properties of the double sliced Wasserstein estimator and its TinyStories wiring."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from core.evaluation import sol
from core.evaluation.sol import SOLConfig, SOLScorer, TextEncoder

SMALL = SOLConfig(n_directions=24, n_projections=16, n_quantiles=8)


def corpus(seed, n=6, d=5, shift=0.0, spread=1.0, lengths=(3, 17)):
    rng = np.random.default_rng(seed)
    return [
        torch.tensor(rng.standard_normal((rng.integers(*lengths), d)) * spread + shift)
        for _ in range(n)
    ]


def distance(x, y, config=SMALL):
    fx = sol.quantile_features(x, config, device="cpu")
    fy = sol.quantile_features(y, config, device="cpu")
    return sol.sol_from_features(fx, fy, config, x[0].shape[1], device="cpu")


def test_identical_corpora_are_at_distance_zero():
    x = corpus(0)
    assert distance(x, [s.clone() for s in x]) == 0.0


def test_symmetric_and_invariant_to_document_order():
    x, y = corpus(0), corpus(1, n=9, shift=0.3)
    assert distance(x, y) == pytest.approx(distance(y, x), rel=1e-6)
    order = np.random.default_rng(5).permutation(len(y))
    assert distance(x, [y[i] for i in order]) == pytest.approx(distance(x, y), rel=1e-6)


def test_translation_scales_linearly_and_noise_increases_distance():
    x = corpus(0, n=8)
    direction = torch.randn(5, dtype=torch.float64)
    shifted = [distance(x, [s + c * direction for s in x]) for c in (0.5, 1.0, 2.0)]
    # Every slice sees the same 1D shift, so W2 (and SOL) is proportional to it.
    assert shifted[1] == pytest.approx(2 * shifted[0], rel=1e-4)
    assert shifted[2] == pytest.approx(4 * shifted[0], rel=1e-4)
    noise = [torch.randn_like(s) for s in x]
    perturbed = [distance(x, [s + c * e for s, e in zip(x, noise)]) for c in (0.1, 0.5, 2.0)]
    assert perturbed[0] < perturbed[1] < perturbed[2]


def brute_force(x, y, config):
    """Independent loop implementation: fine-grid quantile integration, explicit W2."""
    u = sol.directions(config.n_directions, x[0].shape[1], config.seed)
    g = sol.gp_directions(config.n_projections, config.n_quantiles, config.lengthscale, config.gp_seed)
    q = config.n_quantiles

    def binned(doc):
        doc = doc.numpy()
        out = np.zeros((len(u), q))
        for k, direction in enumerate(u):
            values = np.sort(doc @ direction)
            # Quantile function on a grid fine enough to resolve every atom and bin.
            fine = len(values) * q
            grid = (np.arange(fine) + 0.5) / fine
            quantile = values[np.minimum((grid * len(values)).astype(int), len(values) - 1)]
            out[k] = quantile.reshape(q, -1).mean(1)
        return out

    bx, by = [binned(d) for d in x], [binned(d) for d in y]
    total = 0.0
    for k in range(len(u)):
        for j in range(len(g)):
            a = np.sort([g[j] @ b[k] / q for b in bx])
            b = np.sort([g[j] @ b[k] / q for b in by])
            if len(a) == len(b):
                total += np.mean((a - b) ** 2)
            else:  # W2 between empirical measures through their quantile functions
                grid = (np.arange(len(a) * len(b)) + 0.5) / (len(a) * len(b))
                qa = a[(grid * len(a)).astype(int)]
                qb = b[(grid * len(b)).astype(int)]
                total += np.mean((qa - qb) ** 2)
    return math.sqrt(x[0].shape[1] * total / (len(u) * len(g)))


@pytest.mark.parametrize("sizes", [(5, 5), (4, 7)])
@pytest.mark.parametrize("lengthscale", [0.1, None])
def test_matches_brute_force_reference(sizes, lengthscale):
    config = SOLConfig(n_directions=6, n_projections=5, n_quantiles=4, lengthscale=lengthscale)
    x, y = corpus(2, n=sizes[0], d=3), corpus(3, n=sizes[1], d=3, spread=1.5)
    assert distance(x, y, config) == pytest.approx(brute_force(x, y, config), rel=1e-5)


def test_matches_reference_package_when_installed():
    reference = pytest.importorskip("sol_metric")
    x, y = corpus(4, n=7, d=6), corpus(5, n=9, d=6, shift=0.2)
    (tx, ox), (ty, oy) = (
        (torch.cat(c).numpy(), np.r_[0, np.cumsum([len(s) for s in c])]) for c in (x, y)
    )
    theirs = reference.SOL(
        device="cpu", dtype="float64", n_directions=24, n_projections=16, n_quantiles=8
    ).from_tokens(tx, ox, ty, oy)
    assert distance(x, y) == pytest.approx(theirs, rel=1e-6)


def test_fixed_seeds_give_identical_projections_and_bootstrap():
    x, y = corpus(0), corpus(1, shift=0.4)
    assert distance(x, y) == distance(x, y)
    fx = sol.quantile_features(x, SMALL, device="cpu")
    fy = sol.quantile_features(y, SMALL, device="cpu")
    first = sol.sol_bootstrap(fx, fy, SMALL, 5, replicates=20, device="cpu")
    assert first == sol.sol_bootstrap(fx, fy, SMALL, 5, replicates=20, device="cpu")
    assert first["low"] <= first["high"]
    same = sol.sol_bootstrap(fx, fx.clone(), SMALL, 5, replicates=5, device="cpu")
    assert same["low"] == same["high"] == 0.0


def test_bootstrap_sub_slices_are_the_smaller_seeded_families():
    x, y = corpus(0), corpus(1, shift=0.4)
    small = SOLConfig(n_directions=6, n_projections=4, n_quantiles=8)
    # The first k seeded directions and GP draws are exactly the k-draw families.
    full = sol.quantile_features(x, SMALL, device="cpu")[:, :6]
    torch.testing.assert_close(full, sol.quantile_features(x, small, device="cpu"))
    fx = sol.quantile_features(x, SMALL, device="cpu")
    fy = sol.quantile_features(y, SMALL, device="cpu")
    whole = sol.sol_bootstrap(fx, fy, SMALL, 5, replicates=30, device="cpu", slices=None)
    sub = sol.sol_bootstrap(fx, fy, SMALL, 5, replicates=30, device="cpu", slices=(6, 4))
    assert sub["slices"] == [6, 4] and whole["slices"] == [24, 16]
    assert 0 <= sub["low"] <= sub["high"] and 0 <= whole["low"] <= whole["high"]


def test_features_do_not_depend_on_batching():
    x = corpus(7, n=9)
    whole = sol.quantile_features(x, SMALL, device="cpu", documents_per_batch=64)
    batched = sol.quantile_features(x, SMALL, device="cpu", documents_per_batch=2)
    torch.testing.assert_close(whole, batched)


class CharTokenizer:
    """Bytes as token ids, EOS = 255: enough of the Hugging Face interface for TextEncoder."""

    eos_token, eos_token_id = "<eos>", 255
    pad_token_id = None
    pad_token = None

    def __call__(self, text, truncation=False, max_length=None):
        ids = list(text.encode()[: max_length if truncation else None])
        return {"input_ids": [min(i, 254) for i in ids]}


def tiny_encoder(calls=None):
    from transformers import GPT2Config, GPT2Model

    torch.manual_seed(0)
    model = GPT2Model(GPT2Config(vocab_size=256, n_positions=64, n_embd=16, n_layer=1, n_head=2))
    encoder = TextEncoder("tiny", device="cpu", dtype="float32", model=model, tokenizer=CharTokenizer())
    if calls is not None:
        original = encoder.__call__

        class Counting:
            provenance = encoder.provenance

            def __call__(self, texts):
                calls.append(list(texts))
                return original(texts)

        return Counting()
    return encoder


STORIES = ["Amy saw a cat.", "Bob ran home fast.", "The sun was warm.", "", "Dan ate a big red apple."]


def test_text_encoder_excludes_padding_and_encodes_empty_text():
    encoder = tiny_encoder()
    batched = encoder(STORIES)
    encoder.batch_size = 1
    single = encoder(STORIES)
    assert [len(s) for s in batched] == [len(t) or 1 for t in STORIES]
    for a, b in zip(batched, single):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_scorer_caches_references_and_is_order_invariant(tmp_path):
    calls = []
    scorer = SOLScorer(
        tiny_encoder(calls), n_directions=16, n_projections=8, n_quantiles=4,
        cache_dir=tmp_path, device="cpu",
    )
    generated = ["Amy saw a dog.", "Bob ran away.", "The sun set.", "Hi", "Dan ate."]
    first = scorer.score(generated, STORIES)["sol"]
    assert scorer.score(STORIES, STORIES)["sol"] == 0.0
    assert len(list(tmp_path.glob("references-*.pt"))) == 1
    order = [3, 1, 4, 0, 2]
    again = scorer.score([generated[i] for i in order], [STORIES[i] for i in order])["sol"]
    assert again == pytest.approx(first, rel=1e-6)
    # References were encoded once; later calls encoded only the generated side.
    assert sum(sorted(c) == sorted(STORIES) for c in calls) == 2  # first featurize + STORIES as generated


def test_tinystories_metrics_record_sol_and_both_bleu_variants(tmp_path):
    from core.tasks.tinystories import TinyStoriesTask
    from core.tests.test_tinystories import fixture_data

    scorer = SOLScorer(tiny_encoder(), n_directions=8, n_projections=4, n_quantiles=4,
                       cache_dir=tmp_path / "sol", device="cpu")
    task = TinyStoriesTask(
        fixture_data(tmp_path), length=40, prompt_bytes=4, calculate_bleu=True,
        sol=scorer, primary_metric="sol",
    )
    assert task.primary_metric.name == "sol" and task.primary_metric.mode == "min"
    dataset = task.validation_dataset()
    examples = [dataset[i] for i in range(len(dataset))]
    metric = task.create_metric_accumulator("validation")
    metric.update([e["input_ids"] for e in examples], examples, range(len(examples)))
    result = metric.compute()
    assert result.sol == 0.0  # the references themselves
    assert result.corpus_bleu == pytest.approx(100.0)
    assert result.corpus_bleu_no_bp >= result.corpus_bleu
    assert result.sol_signature["config"]["n_directions"] == 8
    wrong = task.create_metric_accumulator("validation")
    truncated = [e["input_ids"][: e["prompt_length"] + 17] + [task.tokenizer.eos_token_id] for e in examples]
    wrong.update(truncated, examples, range(len(examples)))
    worse = wrong.compute()
    assert worse.sol > 0
    # Short output: the brevity penalty lowers BLEU, the no-penalty variant does not.
    assert worse.corpus_bleu_no_bp > worse.corpus_bleu


def test_primary_sol_requires_a_scorer_and_scorer_is_lazy(tmp_path, monkeypatch):
    from core.tasks.tinystories import TinyStoriesTask
    from core.tests.test_tinystories import fixture_data

    with pytest.raises(ValueError, match="task.sol"):
        TinyStoriesTask(fixture_data(tmp_path), length=40, prompt_bytes=4, primary_metric="sol")
    loads = []
    monkeypatch.setattr(TextEncoder, "_load", lambda self: loads.append(1))
    settings = OmegaConf.load(Path("configs/task/sol/gpt2_large.yaml"))
    settings.cache_dir = str(tmp_path)
    task = TinyStoriesTask(
        fixture_data(tmp_path), length=40, prompt_bytes=4, sol=settings, primary_metric="sol"
    )
    task.prepare_data()
    task.collate_training([task.training_dataset()[0]])
    assert task._sol is None and not loads  # training never builds the scorer
    scorer = task.sol_scorer()
    assert scorer.config == SOLConfig()  # paper defaults: 1024 x 1024 slices, 64 bins
    assert scorer.encoder.model_name == "gpt2-large" and not loads
