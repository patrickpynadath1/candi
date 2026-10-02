"""Generator-independent token entropy and causal-reference perplexity."""

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CausalScore:
    negative_log_likelihood: float
    scored_tokens: int


def token_entropy(tokens):
    """Legacy empirical unigram entropy in nats, including special tokens."""
    _, counts = torch.unique(torch.as_tensor(tokens), return_counts=True, sorted=False)
    return torch.special.entr(counts.float() / counts.sum()).sum().item()


def valid_next_tokens(tokens, attention_mask, eos_token_id, bos_token_id=None):
    if eos_token_id is None:
        return attention_mask[..., 1:].bool()
    is_eos = tokens == eos_token_id
    if bos_token_id == eos_token_id:
        # GPT-2's leading BOS shares the EOS id; it is context, not termination.
        is_eos[..., 0] = False
    eos_count = is_eos.cumsum(-1)
    through_first_eos = (eos_count == 0) | (is_eos & (eos_count == 1))
    return through_first_eos[..., 1:] & attention_mask[..., 1:].bool()


def corpus_perplexity(nll, tokens):
    if not tokens:
        return None
    if not math.isfinite(nll):
        raise ValueError("reference model produced nonfinite loss")
    try:
        return math.exp(nll / tokens)
    except OverflowError as exc:
        raise ValueError("reference model perplexity overflowed") from exc


class CausalReferenceScorer:
    """HF causal model scoring with the original nonoverlapping context math.

    Retokenization truncates at max_length. Each context independently excludes
    its first token and stops scoring after its first EOS (inclusive). A shared
    BOS/EOS at the beginning of the document is treated as BOS by default.
    Model/tokenizer injection makes the mathematics testable without downloads.
    """

    def __init__(
        self,
        model_name="gpt2-large",
        batch_size=8,
        context_size=1024,
        device="cpu",
        local_files_only=False,
        model=None,
        tokenizer=None,
        ignore_initial_bos=True,
    ):
        if batch_size <= 0 or context_size < 2:
            raise ValueError("scorer batch_size must be positive and context_size >= 2")
        self.model_name, self.batch_size = model_name, batch_size
        self.context_size, self.device = context_size, torch.device(device)
        self.local_files_only = local_files_only
        self.model, self.tokenizer = model, tokenizer
        self.ignore_initial_bos = ignore_initial_bos

    def _load(self):
        if self.tokenizer is None or self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, local_files_only=self.local_files_only
                )
            if self.model is None:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, local_files_only=self.local_files_only
                )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.to(self.device).eval()

    @torch.no_grad()
    def score(self, texts, max_length):
        if not texts:
            return []
        self._load()
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        tokens = encoded["input_ids"].to(self.device)
        attention = encoded["attention_mask"].to(self.device)
        scores = []
        # The generation evaluator may enter BF16 autocast; the reference model
        # retains its own precision rather than inheriting generator precision.
        with torch.autocast(self.device.type, enabled=False):
            for offset in range(0, len(texts), self.batch_size):
                batch = tokens[offset : offset + self.batch_size]
                masks = attention[offset : offset + self.batch_size]
                totals = torch.zeros(
                    len(batch), dtype=torch.float64, device=self.device
                )
                counts = torch.zeros(len(batch), dtype=torch.long, device=self.device)
                for chunk_index, (chunk, mask) in enumerate(
                    zip(
                        batch.split(self.context_size, -1),
                        masks.split(self.context_size, -1),
                    )
                ):
                    if chunk.shape[-1] < 2:
                        continue
                    logits = self.model(chunk, attention_mask=mask)[0].transpose(-1, -2)
                    nll = F.cross_entropy(
                        logits[..., :-1], chunk[..., 1:], reduction="none"
                    )
                    bos = (
                        getattr(self.tokenizer, "bos_token_id", None)
                        if chunk_index == 0 and self.ignore_initial_bos
                        else None
                    )
                    valid = valid_next_tokens(
                        chunk, mask, self.tokenizer.eos_token_id, bos
                    )
                    totals += (nll * valid).sum(-1).double()
                    counts += valid.sum(-1)
                scores.extend(
                    CausalScore(float(nll), int(count))
                    for nll, count in zip(totals.cpu(), counts.cpu())
                )
        return scores
