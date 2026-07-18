import math
from typing import List, Dict, Tuple
import os
import typing
import json
import types

import torch
import torch.nn.functional as F
import torchmetrics
import transformers

import dataloader
import utils
from algos.registry import algo_registry

LOG2 = math.log(2)


def _to_attr_namespace(value):
    if isinstance(value, dict):
        return types.SimpleNamespace(
            **{key: _to_attr_namespace(val) for key, val in value.items()}
        )
    if isinstance(value, list):
        return [_to_attr_namespace(item) for item in value]
    return value


class NLL(torchmetrics.aggregation.MeanMetric):
    def update(
        self,
        value: typing.Union[float, torch.Tensor],
        weight: typing.Union[float, torch.Tensor] = 1.0,
    ) -> None:
        """Update state with data.

        Args:
          value: Either a float or tensor containing data.
            Additional tensor dimensions will be flattened
          weight: Either a float or tensor containing weights
            for calculating the average. Shape of weight should
            be able to broadcast with the shape of `value`.
            Default to `1.0` corresponding to simple harmonic
            average.
        """
        # broadcast weight to value shape
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)
        weight = torch.broadcast_to(weight, value.shape)
        value, weight = self._cast_and_nan_check_input(value, weight)

        if value.numel() == 0:
            return
        self.mean_value += value.sum()
        self.weight += weight.sum()


class BPD(NLL):
    def compute(self) -> torch.Tensor:
        """Computes the bits per dimension.

        Returns:
          bpd
        """
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> torch.Tensor:
        """Computes the Perplexity.

        Returns:
         Perplexity
        """
        return torch.exp(self.mean_value / self.weight)


class Metrics:
    def __init__(
        self,
        gen_ppl_eval_model_name_or_path=None,
        eval_ppl_batch_size=None,
        eval_model_source="hf",
        local_eval_checkpoint_path=None,
        local_eval_device="cuda",
        local_eval_context_size=None,
    ) -> None:
        metrics = torchmetrics.MetricCollection(
            {"nll": NLL(), "bpd": BPD(), "ppl": Perplexity()}
        )
        metrics.set_dtype(torch.float64)
        self.train_nlls = metrics.clone(prefix="train/")
        self.train_aux = BPD()
        self.valid_nlls = metrics.clone(prefix="val/")
        self.valid_aux = BPD()
        self.gen_ppl = Perplexity()
        self.gen_bpc = BPD()
        self.sample_entropy = torchmetrics.aggregation.MeanMetric()
        self.eval_ppl_batch_size = eval_ppl_batch_size
        self.eval_model_source = eval_model_source
        self.gen_ppl_eval_model_name_or_path = gen_ppl_eval_model_name_or_path
        self.local_eval_checkpoint_path = local_eval_checkpoint_path
        self.local_eval_device = local_eval_device
        self.local_eval_context_size = local_eval_context_size
        self._eval_model = None
        self._local_eval_config = None
        self._tokenizer = None

    def to(self, *args, **kwargs):
        self.gen_ppl = self.gen_ppl.to(*args, **kwargs)
        self.gen_bpc = self.gen_bpc.to(*args, **kwargs)
        self.sample_entropy = self.sample_entropy.to(*args, **kwargs)
        self.train_nlls = self.train_nlls.to(*args, **kwargs)
        self.train_aux = self.train_aux.to(*args, **kwargs)
        self.valid_nlls = self.valid_nlls.to(*args, **kwargs)
        self.valid_aux = self.valid_aux.to(*args, **kwargs)
        if self._eval_model is not None:
            self._eval_model = self._eval_model.to(*args, **kwargs)

    def reset(self):
        self.gen_ppl.reset()
        self.gen_bpc.reset()
        self.sample_entropy.reset()
        self.train_nlls.reset()
        self.train_aux.reset()
        self.valid_nlls.reset()
        self.valid_aux.reset()

    def _detach_metric_input(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach()
        return value

    def update_train(self, nll, aux_loss, num_tokens):
        nll = self._detach_metric_input(nll)
        aux_loss = self._detach_metric_input(aux_loss)
        num_tokens = self._detach_metric_input(num_tokens)
        self.train_nlls.update(nll, num_tokens)
        self.train_aux.update(aux_loss, num_tokens)

    def update_valid(self, nll, aux_loss, num_tokens):
        nll = self._detach_metric_input(nll)
        aux_loss = self._detach_metric_input(aux_loss)
        num_tokens = self._detach_metric_input(num_tokens)
        self.valid_nlls.update(nll, num_tokens)
        self.valid_aux.update(aux_loss, num_tokens)

    def _initialize_eval_tokenizer(self):
        if self.eval_model_source == "hf":
            if self.gen_ppl_eval_model_name_or_path is None:
                raise ValueError(
                    "gen_ppl_eval_model_name_or_path must be provided for HF eval."
                )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.gen_ppl_eval_model_name_or_path
            )
        elif self.eval_model_source == "local":
            tokenizer = self._load_local_eval_tokenizer()
        else:
            raise ValueError(
                f"Unsupported eval_model_source: {self.eval_model_source}. "
                "Expected 'hf' or 'local'."
            )
        self._ensure_tokenizer_padding(tokenizer)
        return tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self._initialize_eval_tokenizer()
        return self._tokenizer

    def _ensure_tokenizer_padding(self, tokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def _load_local_eval_config(self):
        if self._local_eval_config is not None:
            return self._local_eval_config
        if self.local_eval_checkpoint_path is None:
            raise ValueError(
                "local_eval_checkpoint_path must be provided for local eval."
            )
        checkpoint = torch.load(
            self.local_eval_checkpoint_path, map_location="cpu", weights_only=False
        )
        try:
            config = checkpoint["hyper_parameters"]["config"]
        except KeyError as exc:
            raise ValueError(
                "Local eval checkpoint must contain hyper_parameters['config']."
            ) from exc
        if isinstance(config, dict):
            config = _to_attr_namespace(config)
        self._local_eval_config = config
        return self._local_eval_config

    def _load_local_eval_tokenizer(self):
        config = self._load_local_eval_config()
        tokenizer = dataloader.get_tokenizer(config)
        if hasattr(config.data, "vocab_file"):
            vocab_path = os.path.join(config.data.cache_dir, config.data.vocab_file)
            new_vocab = json.load(open(vocab_path, "r"))
            tokenizer.add_tokens(new_vocab)
        return tokenizer

    def _get_eval_model(self, device):
        target_device = device
        if self.eval_model_source == "local" and self.local_eval_device is not None:
            target_device = self.local_eval_device
        if self._eval_model is None:
            if self.eval_model_source == "hf":
                self._eval_model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.gen_ppl_eval_model_name_or_path
                ).eval()
            else:
                self._eval_model = utils.get_aux_model(
                    self.local_eval_checkpoint_path,
                    algo_registry=algo_registry,
                    tokenizer=self.tokenizer,
                )
                if not hasattr(self._eval_model, "_process_model_input") or not hasattr(
                    self._eval_model, "nll"
                ):
                    raise TypeError(
                        "Local eval model must expose _process_model_input(...) and "
                        "nll(...) to compute generative perplexity."
                    )
                if hasattr(self._eval_model, "_eval_mode"):
                    self._eval_model._eval_mode()
        # Keep Llama-2 on the default device path used before.
        if not self._is_llama2_hf_eval():
            self._eval_model = self._eval_model.to(target_device)
        return self._eval_model

    def _is_llama2_hf_eval(self):
        return (
            self.eval_model_source == "hf"
            and self.gen_ppl_eval_model_name_or_path is not None
            and "llama2" in self.gen_ppl_eval_model_name_or_path.lower()
        )

    def _get_eval_context_size(self):
        if self.local_eval_context_size is not None:
            return self.local_eval_context_size
        if self.eval_model_source == "local":
            config = self._load_local_eval_config()
            return int(config.model.length)
        if self._is_llama2_hf_eval():
            return 4096
        return 1024

    def _get_tokenizer_kwargs(self, max_length):
        return {
            "return_tensors": "pt",
            "return_token_type_ids": False,
            "return_attention_mask": True,
            "truncation": True,
            "padding": True,
            "max_length": max_length,
        }

    def _build_valid_tokens(self, samples, attn_mask):
        if self.tokenizer.eos_token_id is None:
            return attn_mask[..., 1:].to(dtype=torch.bool)
        is_eos = samples == self.tokenizer.eos_token_id
        eos_count = is_eos.cumsum(-1)
        through_first_eos = (eos_count == 0) | (is_eos & (eos_count == 1))
        valid_tokens = through_first_eos[..., 1:] & attn_mask[..., 1:].bool()
        return valid_tokens

    def _compute_utf8_lengths(self, samples, attn_mask):
        utf8_lengths = []
        samples = samples.detach().cpu()
        attn_mask = attn_mask.detach().cpu()
        for sample_ids, sample_mask in zip(samples, attn_mask):
            valid_ids = sample_ids[sample_mask.bool()].tolist()
            text = self.tokenizer.decode(
                valid_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            utf8_lengths.append(len(text.encode("utf-8")))
        return torch.tensor(utf8_lengths, dtype=torch.float64)

    @torch.no_grad()
    def _eval_retokenize(self, text_samples, max_length, device):
        """Retokenizes samples for the eval model.

        Args:
            text_samples: List of sentences generated by the model.
        Returns:
            samples: Samples re-tokenized for the eval model
            attn_mask: Attention mask for the eval model
            eval_context_size: Size of the context for the eval model
        """
        tokenizer_kwargs = self._get_tokenizer_kwargs(max_length=max_length)
        eval_context_size = self._get_eval_context_size()
        samples = self.tokenizer(text_samples, **tokenizer_kwargs)
        attn_mask = samples["attention_mask"]
        samples = samples["input_ids"]
        if not self._is_llama2_hf_eval():
            attn_mask = attn_mask.to(device)
            samples = samples.to(device)
        return samples, attn_mask, eval_context_size

    @torch.no_grad()
    def _record_hf_generative_perplexity(self, samples, attn_mask, eval_context_size, device):
        eval_model = self._get_eval_model(device)
        batch_size = min(self.eval_ppl_batch_size, samples.shape[0])
        num_batches = math.ceil(samples.shape[0] / batch_size)
        total_nll = torch.zeros((), dtype=torch.float64, device=device)
        for i in range(num_batches):
            batch_slice = slice(i * batch_size, min((i + 1) * batch_size, samples.shape[0]))
            _samples = torch.split(samples[batch_slice], eval_context_size, dim=-1)
            _attn_mask = torch.split(attn_mask[batch_slice], eval_context_size, dim=-1)
            for sample_chunk, attn_mask_chunk in zip(_samples, _attn_mask):
                logits = eval_model(sample_chunk, attention_mask=attn_mask_chunk)
                logits = logits[0].transpose(-1, -2)
                nlls = F.cross_entropy(
                    logits[..., :-1], sample_chunk[..., 1:], reduction="none"
                )
                valid_tokens = self._build_valid_tokens(sample_chunk, attn_mask_chunk)
                self.gen_ppl.update(nlls * valid_tokens, valid_tokens)
                total_nll += (nlls * valid_tokens).sum().to(torch.float64)
        return total_nll

    @torch.no_grad()
    def _record_local_generative_perplexity(
        self, samples, attn_mask, eval_context_size, device
    ):
        eval_model = self._get_eval_model(device)
        batch_size = min(self.eval_ppl_batch_size, samples.shape[0])
        num_batches = math.ceil(samples.shape[0] / batch_size)
        total_nll = torch.zeros((), dtype=torch.float64, device=device)
        for i in range(num_batches):
            batch_slice = slice(i * batch_size, min((i + 1) * batch_size, samples.shape[0]))
            _samples = torch.split(samples[batch_slice], eval_context_size, dim=-1)
            _attn_mask = torch.split(attn_mask[batch_slice], eval_context_size, dim=-1)
            for sample_chunk, attn_mask_chunk in zip(_samples, _attn_mask):
                input_tokens, output_tokens, _ = eval_model._process_model_input(
                    sample_chunk, attn_mask_chunk
                )
                nlls = eval_model.nll(
                    input_tokens,
                    output_tokens,
                    None,
                    False,
                )
                valid_tokens = self._build_valid_tokens(sample_chunk, attn_mask_chunk)
                self.gen_ppl.update(nlls * valid_tokens, valid_tokens)
                total_nll += (nlls * valid_tokens).sum().to(torch.float64)
        return total_nll

    @torch.no_grad()
    def record_entropy(self, tokens):
        for sample in tokens:
            _, counts = torch.unique(sample, return_counts=True, sorted=False)
            entropy = torch.special.entr(counts.float() / counts.sum()).sum().item()
            self.sample_entropy.update(entropy)

    @torch.no_grad()
    def record_generative_perplexity(
        self,
        text_samples: typing.List[str],
        max_length: int,
        retokenize: bool = True,
        device="cuda",
    ) -> None:

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Re-tokenize using eval model's tokenizer
        if retokenize:
            (samples, attn_mask, eval_context_size) = self._eval_retokenize(
                text_samples, max_length=max_length, device=device
            )
        else:
            samples = text_samples
            attn_mask = torch.ones(samples.shape).to(device)
            eval_context_size = samples.shape[-1]
        utf8_lengths = self._compute_utf8_lengths(samples, attn_mask).to(device)
        if self.eval_model_source == "hf":
            total_nll = self._record_hf_generative_perplexity(
                samples, attn_mask, eval_context_size, device
            )
        else:
            total_nll = self._record_local_generative_perplexity(
                samples, attn_mask, eval_context_size, device
            )
        self.gen_bpc.update(total_nll, utf8_lengths.sum())
