from types import SimpleNamespace

import pytest
import torch

import metrics as metrics_module
from metrics import BPD, Metrics, NLL, Perplexity


def test_nll_aggregates_total_nll_by_token_count():
    metric = NLL()
    metric.update(torch.tensor(6.0), torch.tensor(2.0))
    metric.update(torch.tensor(9.0), torch.tensor(3.0))
    assert metric.compute().item() == pytest.approx(3.0)


def test_bpd_and_bpc_convert_nats_to_bits():
    metric = BPD()
    metric.update(torch.tensor(4 * metrics_module.LOG2), torch.tensor(4.0))
    assert metric.compute().item() == pytest.approx(1.0)


def test_perplexity_exponentiates_mean_nll():
    metric = Perplexity()
    metric.update(2 * torch.log(torch.tensor(3.0)), torch.tensor(2.0))
    assert metric.compute().item() == pytest.approx(3.0)


def test_metrics_reset_clears_all_accumulators():
    metrics = Metrics()
    metrics.update_train(torch.tensor(4.0), torch.tensor(2.0), torch.tensor(2.0))
    metrics.update_valid(torch.tensor(3.0), torch.tensor(1.0), torch.tensor(1.0))
    metrics.record_entropy(torch.tensor([[0, 0, 1, 1]]))

    metrics.reset()

    assert metrics.train_nlls.nll.weight.item() == 0
    assert metrics.valid_nlls.nll.weight.item() == 0
    assert metrics.sample_entropy.weight.item() == 0


def test_record_entropy_averages_empirical_entropy():
    metrics = Metrics()
    metrics.record_entropy(torch.tensor([[0, 0, 1, 1], [2, 2, 2, 2]]))
    assert metrics.sample_entropy.compute().item() == pytest.approx(torch.log(torch.tensor(2.0)).item() / 2)


def test_valid_tokens_stop_after_first_eos():
    metrics = Metrics()
    metrics._tokenizer = SimpleNamespace(eos_token_id=9)
    samples = torch.tensor([[1, 2, 9, 4, 5], [1, 2, 3, 4, 5]])
    attention = torch.ones_like(samples)

    valid = metrics._build_valid_tokens(samples, attention)

    assert torch.equal(valid, torch.tensor([[True, True, False, False], [True, True, True, True]]))


def test_valid_tokens_without_eos_uses_shifted_attention_mask():
    metrics = Metrics()
    metrics._tokenizer = SimpleNamespace(eos_token_id=None)
    attention = torch.tensor([[1, 1, 0, 0]])
    valid = metrics._build_valid_tokens(torch.tensor([[1, 2, 3, 4]]), attention)
    assert torch.equal(valid, torch.tensor([[True, False, False]]))


class DecodingTokenizer:
    eos_token_id = None

    def decode(self, ids, **_kwargs):
        mapping = {1: "a", 2: "é", 3: "🙂"}
        return "".join(mapping[index] for index in ids)


def test_utf8_lengths_count_bytes_not_codepoints():
    metrics = Metrics()
    metrics._tokenizer = DecodingTokenizer()
    samples = torch.tensor([[1, 2, 3], [2, 0, 0]])
    attention = torch.tensor([[1, 1, 1], [1, 0, 0]])
    assert torch.equal(metrics._compute_utf8_lengths(samples, attention), torch.tensor([7.0, 2.0], dtype=torch.float64))


def test_ensure_tokenizer_padding_uses_eos_fallback():
    metrics = Metrics()
    tokenizer = SimpleNamespace(pad_token=None, pad_token_id=None, eos_token="<eos>", eos_token_id=7)
    metrics._ensure_tokenizer_padding(tokenizer)
    assert tokenizer.pad_token == "<eos>"
    assert tokenizer.pad_token_id == 7


def test_hf_tokenizer_initialization_is_lazy_and_mocked(monkeypatch):
    tokenizer = SimpleNamespace(pad_token=None, pad_token_id=None, eos_token="e", eos_token_id=1)
    calls = []
    monkeypatch.setattr(
        metrics_module.transformers.AutoTokenizer,
        "from_pretrained",
        lambda name: calls.append(name) or tokenizer,
    )
    metrics = Metrics(gen_ppl_eval_model_name_or_path="tiny-local-name", eval_model_source="hf")

    assert calls == []
    assert metrics.tokenizer is tokenizer
    assert metrics.tokenizer is tokenizer
    assert calls == ["tiny-local-name"]


def test_local_tokenizer_initialization_uses_loader_and_padding(monkeypatch):
    config = SimpleNamespace(data=SimpleNamespace())
    tokenizer = SimpleNamespace(pad_token=None, pad_token_id=None, eos_token="e", eos_token_id=2)
    metrics = Metrics(eval_model_source="local", local_eval_checkpoint_path="unused")
    monkeypatch.setattr(metrics, "_load_local_eval_config", lambda: config)
    monkeypatch.setattr(metrics_module.dataloader, "get_tokenizer", lambda cfg: tokenizer)

    assert metrics.tokenizer is tokenizer
    assert tokenizer.pad_token_id == 2


class FakeHFModel:
    def __init__(self, vocab_size=4):
        self.vocab_size = vocab_size

    def __call__(self, tokens, attention_mask=None):
        logits = torch.zeros(*tokens.shape, self.vocab_size)
        return (logits,)


def test_hf_evaluator_branch_computes_real_cross_entropy(monkeypatch):
    metrics = Metrics(eval_ppl_batch_size=2, eval_model_source="hf")
    metrics._tokenizer = SimpleNamespace(eos_token_id=None)
    monkeypatch.setattr(metrics, "_get_eval_model", lambda _device: FakeHFModel())
    samples = torch.tensor([[0, 1, 2], [1, 2, 3]])
    attention = torch.ones_like(samples)

    total = metrics._record_hf_generative_perplexity(samples, attention, 3, "cpu")

    assert total.item() == pytest.approx(4 * torch.log(torch.tensor(4.0)).item())
    assert metrics.gen_ppl.compute().item() == pytest.approx(4.0)


class FakeLocalModel:
    def _process_model_input(self, samples, attention):
        return samples[:, :-1], samples[:, 1:], attention[:, 1:]

    def nll(self, _inputs, outputs, *_args):
        return torch.full_like(outputs, 0.5, dtype=torch.float32)


def test_local_evaluator_branch_uses_per_token_nll(monkeypatch):
    metrics = Metrics(eval_ppl_batch_size=2, eval_model_source="local")
    metrics._tokenizer = SimpleNamespace(eos_token_id=None)
    monkeypatch.setattr(metrics, "_get_eval_model", lambda _device: FakeLocalModel())
    samples = torch.tensor([[0, 1, 2], [1, 2, 3]])
    attention = torch.ones_like(samples)

    total = metrics._record_local_generative_perplexity(samples, attention, 3, "cpu")

    assert total.item() == pytest.approx(2.0)
    assert metrics.gen_ppl.compute().item() == pytest.approx(torch.exp(torch.tensor(0.5)).item())


def test_invalid_eval_source_is_rejected():
    metrics = Metrics(eval_model_source="remote")
    with pytest.raises(ValueError, match="Unsupported"):
        metrics._initialize_eval_tokenizer()


def test_valid_tokens_include_first_eos_and_exclude_later_eos_and_tokens():
    metrics = Metrics()
    metrics._tokenizer = SimpleNamespace(eos_token_id=9)
    samples = torch.tensor([[1, 9, 2, 9, 3]])
    valid = metrics._build_valid_tokens(samples, torch.ones_like(samples))
    assert torch.equal(valid, torch.tensor([[True, False, False, False]]))


def test_valid_tokens_when_sequence_starts_with_eos():
    metrics = Metrics()
    metrics._tokenizer = SimpleNamespace(eos_token_id=9)
    samples = torch.tensor([[9, 2, 3]])
    valid = metrics._build_valid_tokens(samples, torch.ones_like(samples))
    assert torch.equal(valid, torch.tensor([[False, False]]))


def test_valid_tokens_respect_full_padding():
    metrics = Metrics()
    metrics._tokenizer = SimpleNamespace(eos_token_id=None)
    samples = torch.tensor([[1, 2, 3]])
    assert not metrics._build_valid_tokens(samples, torch.zeros_like(samples)).any()


def test_hf_evaluator_splits_context_windows(monkeypatch):
    metrics = Metrics(eval_ppl_batch_size=1, eval_model_source="hf")
    metrics._tokenizer = SimpleNamespace(eos_token_id=None)
    lengths = []

    class RecordingModel(FakeHFModel):
        def __call__(self, tokens, attention_mask=None):
            lengths.append(tokens.shape[-1])
            return super().__call__(tokens, attention_mask)

    monkeypatch.setattr(metrics, "_get_eval_model", lambda _device: RecordingModel())
    samples = torch.tensor([[0, 1, 2, 3, 0]])
    attention = torch.ones_like(samples)
    total = metrics._record_hf_generative_perplexity(samples, attention, 2, "cpu")
    assert lengths == [2, 2, 1]
    assert total.item() == pytest.approx(2 * torch.log(torch.tensor(4.0)).item())


def test_retokenize_uses_expected_tokenizer_options():
    observed = {}

    class Tokenizer:
        eos_token_id = None

        def __call__(self, texts, **kwargs):
            observed.update(texts=texts, kwargs=kwargs)
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

    metrics = Metrics(eval_model_source="local", local_eval_context_size=17)
    metrics._tokenizer = Tokenizer()
    samples, attention, context = metrics._eval_retokenize(["hello"], 9, "cpu")
    assert observed == {
        "texts": ["hello"],
        "kwargs": {
            "return_tensors": "pt",
            "return_token_type_ids": False,
            "return_attention_mask": True,
            "truncation": True,
            "padding": True,
            "max_length": 9,
        },
    }
    assert samples.device.type == "cpu"
    assert attention.device.type == "cpu"
    assert context == 17
