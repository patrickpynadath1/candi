"""Bits recovered at matched channel capacity: one scorer for every corruption family.

For a trained recipe and a grid of global times t, on validation data:

- channel bits  I_ch(t) = H(X) - E[-log2 p(x_i | s_i)], the context-free Bayes posterior
  from the process's own likelihood and the unigram prior (what the state says about a
  position read alone; the same for every family at matched capacity);
- model bits    I_V(t)  = H(X) - E[-log2 q(x_i | s)], the network's clean-token posterior
  from the whole state (a Barber-Agakov lower bound on I(X_i; S), i.e. V-information with
  V = one pass of the network). SC models also get a second pass fed their first
  prediction (``bits_sc2``).

Every family is corrupted and predicted through its own HybridDiffusion code
(points -> _corrupt -> training-pathway posterior), so only the channel likelihood is
written here, per process. Plot I_V against I_ch to compare families at matched capacity.

usage: python bits_recovered.py GPU OUT_DIR CONFIG.yaml [CONFIG.yaml ...]
       python bits_recovered.py --selfcheck GPU CONFIG.yaml
CONFIG is a case's final/seed-*/config.yaml (it names the checkpoint).
"""

import json
import math
import os
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", sys.argv[2] if sys.argv[1] == "--selfcheck" else sys.argv[1])

import torch
from torch.nn import functional as F

LN2 = math.log(2)
# Target channel capacity as a fraction of H(X). Times are found per model by bisection,
# since learned clocks (CDCD) can pack the whole information path into a sliver of t.
FRACTIONS = [round(0.02 + 0.04 * i, 2) for i in range(25)]
MAX_EXAMPLES = 1024
BISECT_EXAMPLES = 256


# ---------------------------------------------------------------- channel likelihoods
def clean_table(view, vocab):
    """Clean state of every token, (V, d), in the state space the process corrupts."""
    tokens = torch.arange(vocab, device="cuda")
    return view.representation.encode(tokens[None])[0].float()


def channel_loglik(view, state, coefficients, vocab, table=None):
    """log p(s_i | x_i = v) up to a v-independent constant, shape (B, L, vocab).

    Each process corrupts positions independently given the clean token, so this is the
    exact per-position channel. Rows are batch rows with their own coefficients."""
    process = view.process
    name = type(process).__name__
    if name == "AbsorbingProcess":
        observed = state.ne(process.mask_index)
        hot = F.one_hot(state.clamp_max(vocab - 1), vocab).bool() & observed[..., None]
        out = torch.where(hot, 0.0, -torch.inf)
        return torch.where(observed[..., None], out, torch.zeros_like(out))
    if name in ("UniformProcess", "PosteriorUniformProcess"):
        keep = coefficients[0].double()[:, None, None]
        size = process.vocab_size
        same = (keep + (1 - keep) / size).log()
        other = ((1 - keep) / size).log().expand(-1, *state.shape[1:], vocab)
        hot = F.one_hot(state.clamp_max(vocab - 1), vocab).bool() & state.lt(vocab)[..., None]
        return torch.where(hot, same, other).float()
    if name == "GaussianProcess":
        alpha = coefficients.alpha.double()[:, None, None]
        variance = coefficients.sigma.double().square()[:, None, None]
        dot = torch.einsum("bld,vd->blv", state.double(), table.double())
        return (alpha / variance * dot - alpha.square() / (2 * variance) * table.double().square().sum(-1)).float()
    if name == "VMFProcess":
        kappa = coefficients[0].double()[:, None, None]
        unit = F.normalize(table.double(), dim=-1)
        return (kappa * torch.einsum("bld,vd->blv", F.normalize(state.double(), dim=-1), unit)).float()
    if name == "SphericalProcess":
        # s = slerp(e, u, f) with u uniform on the sphere and f = 1 - keep: the angle to
        # the clean point is f * phi, phi ~ sin^(d-2), so
        # p(s|e) proportional to sin^(d-2)(theta/f) / sin^(d-2)(theta) for theta < f*pi.
        d = state.shape[-1]
        f = (1 - coefficients[0].double()).clamp_min(1e-12)[:, None, None]
        unit = F.normalize(table.double(), dim=-1)
        cosine = torch.einsum("bld,vd->blv", F.normalize(state.double(), dim=-1), unit)
        theta = cosine.clamp(-1, 1).acos()
        ratio = torch.where(
            theta < 1e-6, -f.log().expand_as(theta),
            (theta / f).sin().clamp_min(1e-300).log() - theta.sin().clamp_min(1e-300).log(),
        )
        out = torch.where(theta < f * math.pi, (d - 2) * ratio, -torch.inf)
        return out.float()
    if name == "DirichletProcess":
        # beta = c((1-t) onehot(v) + t pi); only coordinate v depends on v, sum(beta) = c.
        t = coefficients.time.double()[:, None, None]
        c = coefficients.concentration.double()[:, None, None]
        prior = process.reference.double()[:vocab]
        # shortcut: assumes a reference with full support; zero weights put states on faces.
        assert (prior > 0).all(), "Dirichlet reference with zero weights is not handled"
        log_s = state.double()[..., :vocab]
        hot = c * (1 - t)
        base = c * t * prior
        return (hot * log_s + torch.lgamma(base) - torch.lgamma(base + hot)).float()
    raise NotImplementedError(f"no channel likelihood for {name}")


def channel_posterior(loglik, log_prior):
    return (loglik + log_prior).log_softmax(-1)


# ---------------------------------------------------------------------- loading
def load_case(config_path):
    import lightning as L
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from core.training.checkpoints import load_weights
    from core.training.module import TrainingModule

    config = OmegaConf.load(config_path)
    L.seed_everything(config.seed, workers=True)
    task = instantiate(config.task)
    datamodule = instantiate(config.datamodule, task=task)
    datamodule.prepare_data()
    datamodule.setup("fit")
    module = TrainingModule(config)
    if not config.evaluation.use_ema:
        module.ema = None
    step = load_weights(module, config.evaluation.checkpoint, legacy=config.evaluation.legacy)
    module.to("cuda").eval()
    return config, task, datamodule, module, step


def sudoku_blanks(batch):
    """Score only solution cells whose clue is blank: separators and given clues are free.

    Layout (checked): [11] 89 prompt tokens [11] 89 solution tokens, 10 = row separator,
    0 = blank clue."""
    ids = batch["input_ids"]
    assert ids.shape[1] == 180 and (ids[:, 0] == 11).all() and (ids[:, 90] == 11).all()
    mask = batch["loss_mask"].bool().clone()
    mask[:, 91:] &= ids[:, 1:90].eq(0)
    assert mask[:, :91].sum() == 0
    batch["loss_mask"] = mask
    return batch


def batches(datamodule, limit):
    sudoku = "sudoku" in type(datamodule.task).__name__.lower()
    seen = 0
    for batch in datamodule.val_dataloader():
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        if sudoku:
            batch = sudoku_blanks(batch)
        take = min(len(batch["input_ids"]), limit - seen)
        yield {k: v[:take] if torch.is_tensor(v) else v for k, v in batch.items()}
        seen += take
        if seen >= limit:
            return


def output_name(config_path):
    """experiment__case: the Gaussian-512 cases share names across two experiments."""
    case = Path(config_path).parents[2]
    return f"{case.parent.name}__{case.name}"


# ---------------------------------------------------------------------- scoring
def channel_bits(algo, view, data, log_prior, entropy, vocab, table, t):
    """Context-free channel bits at global time t, averaged over the scored positions."""
    total = n = 0.0
    for b in data:
        tokens, loss_mask = b["input_ids"], b["loss_mask"].bool()
        corruption = b.get("corruption_mask", b["loss_mask"])
        global_time = torch.full((len(tokens),), t, device="cuda")
        points = algo.points(global_time, training=False, corruption_mask=corruption)
        state = algo._corrupt(tokens, points, corruption)[0]
        ll = channel_loglik(view, state, points[0].coefficients, vocab, table)
        ce = -channel_posterior(ll, log_prior).gather(-1, tokens.clamp_max(vocab - 1)[..., None])[..., 0]
        total += float(ce[loss_mask].sum())
        n += float(loss_mask.sum())
    return entropy - total / n / LN2


def find_times(algo, view, data, log_prior, entropy, vocab, table, fractions, iterations=24):
    """Bisect global time so channel bits hit each target (channel bits fall as t rises).

    The estimate is Monte Carlo, so a target can only be met to within its noise."""
    times = []
    for fraction in fractions:
        target, lo, hi = fraction * entropy, 0.0, 1.0
        for _ in range(iterations):
            mid = (lo + hi) / 2
            if channel_bits(algo, view, data, log_prior, entropy, vocab, table, mid) > target:
                lo = mid
            else:
                hi = mid
        times.append((lo + hi) / 2)
    return times


@torch.no_grad()
def score_case(config_path, out_dir=None, fractions=FRACTIONS, limit=MAX_EXAMPLES):
    from core.modifiers.self_conditioning.prediction import TrajectoryState

    started = time.time()
    config, task, datamodule, module, step = load_case(config_path)
    algo = module.algorithm
    assert len(algo.views) == 1, "first pass handles one-view recipes only"
    view = algo.views[0]
    data = list(batches(datamodule, limit))
    vocab = int(task.vocab_size)
    # unigram over scored positions: the no-input predictor of V-information
    counts = torch.zeros(vocab, device="cuda", dtype=torch.float64)
    for b in data:
        m = b["loss_mask"].bool()
        counts += torch.bincount(b["input_ids"][m], minlength=vocab)[:vocab].double()
    prior = counts / counts.sum()
    log_prior = prior.clamp_min(1e-30).log().float()
    entropy = float(-(prior * prior.clamp_min(1e-30).log()).sum() / LN2)
    table = None if not any(k in type(view.process).__name__ for k in ("Gaussian", "VMF", "Spherical")) else clean_table(view, vocab)
    sc = bool(algo.prediction.enabled)
    autocast = "bf16" in str(config.trainer.precision)
    rows = []
    with module.evaluation_weights():
        bisect_data = [b for b in batches(datamodule, BISECT_EXAMPLES)]
        times = find_times(algo, view, bisect_data, log_prior, entropy, vocab, table, fractions)
        for fraction, t in zip(fractions, times):
            total = dict(n=0.0, ch=0.0, m1=0.0, m2=0.0)
            for b in data:
                tokens, loss_mask = b["input_ids"], b["loss_mask"].bool()
                corruption = b.get("corruption_mask", b["loss_mask"])
                global_time = torch.full((len(tokens),), t, device="cuda")
                points = algo.points(global_time, training=False, corruption_mask=corruption)
                states = algo._corrupt(tokens, points, corruption)
                target = tokens.clamp_max(vocab - 1)[..., None]
                ll = channel_loglik(view, states[0], points[0].coefficients, vocab, table)
                ch = -channel_posterior(ll, log_prior).gather(-1, target)[..., 0]
                trajectory = TrajectoryState() if sc else None
                passes = []
                for _ in range(2 if sc else 1):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                        logits = algo._prediction(algo._pathway("training"), states, points, global_time, trajectory=trajectory)
                    passes.append(-logits.float().log_softmax(-1).gather(-1, target)[..., 0])
                n = loss_mask.sum()
                total["n"] += float(n)
                total["ch"] += float(ch[loss_mask].sum())
                total["m1"] += float(passes[0][loss_mask].sum())
                total["m2"] += float(passes[-1][loss_mask].sum())
            n = total["n"]
            row = dict(
                target_fraction=fraction,
                t=t,
                bits_channel=entropy - total["ch"] / n / LN2,
                bits_model=entropy - total["m1"] / n / LN2,
            )
            if sc:
                row["bits_sc2"] = entropy - total["m2"] / n / LN2
            rows.append(row)
            print(json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)
    result = dict(
        config=str(config_path), checkpoint=str(config.evaluation.checkpoint), step=step,
        process=type(view.process).__name__, self_conditioning=sc, vocab=vocab,
        entropy_bits=entropy, positions=int(total["n"]), examples=sum(len(b["input_ids"]) for b in data),
        seconds=round(time.time() - started, 1), rows=rows,
    )
    if out_dir is not None:
        (Path(out_dir) / f"{output_name(config_path)}.json").write_text(json.dumps(result, indent=1))
    del module
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------- self-check
@torch.no_grad()
def selfcheck(config_path):
    """The channel likelihood has the right shape only if no rescaling of it lowers the
    channel CE on states the process itself drew (masked is exact, so this is trivial)."""
    config, task, datamodule, module, _ = load_case(config_path)
    algo, view = module.algorithm, module.algorithm.views[0]
    vocab = int(task.vocab_size)
    b = next(batches(datamodule, 256))
    prior = torch.bincount(b["input_ids"][b["loss_mask"].bool()], minlength=vocab)[:vocab].double()
    log_prior = (prior / prior.sum()).clamp_min(1e-30).log().float()
    table = clean_table(view, vocab) if any(k in type(view.process).__name__ for k in ("Gaussian", "VMF", "Spherical")) else None
    m = b["loss_mask"].bool()
    corruption = b.get("corruption_mask", b["loss_mask"])
    target = b["input_ids"].clamp_max(vocab - 1)[..., None]
    for t in (0.05, 0.5, 0.95):
        gt = torch.full((len(target),), t, device="cuda")
        points = algo.points(gt, training=False, corruption_mask=corruption)
        states = algo._corrupt(b["input_ids"], points, corruption)
        ll = channel_loglik(view, states[0], points[0].coefficients, vocab, table)
        ce = {s: float(-channel_posterior(s * ll.clamp_min(-1e30), log_prior).gather(-1, target)[..., 0][m].mean()) for s in (0.8, 1.0, 1.25)}
        print(f"t={t} channel CE at likelihood scale 0.8/1/1.25: {ce}")
        assert ce[1.0] <= min(ce[0.8], ce[1.25]) + 1e-3, "rescaled likelihood beats the claimed one"
    print("selfcheck OK:", type(view.process).__name__)


if __name__ == "__main__":
    if sys.argv[1] == "--selfcheck":
        selfcheck(sys.argv[3])
    else:
        out = Path(sys.argv[2])
        out.mkdir(parents=True, exist_ok=True)
        for path in sys.argv[3:]:
            if (out / f"{output_name(path)}.json").exists():
                continue
            try:
                score_case(path, out)
            except Exception as error:  # keep the GPU worker going; report at the end
                import traceback
                traceback.print_exc()
                print(json.dumps(dict(config=path, error=repr(error))), flush=True)
