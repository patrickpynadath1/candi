"""Experimental seeded, bounded recursive grammars over hidden HubPaths cores.

Run ``python -m tasks.hubpaths.grammar --output PATH`` for paired grammar and
broken-agreement corpora and held-out prefix probes. Parse trees are sidecars.
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import preset
from .data import _Writer
from .graph import N_SPECIAL, SEP, generate_graph
from .latent import (
    CorePrior,
    LatentCoreConfig,
    bridge_rsp,
    draw_document,
    estimate_bridge_visitation,
)


@dataclass(frozen=True)
class GrammarConfig:
    categories: int = 6
    alternatives: int = 3
    features: int = 4
    max_depth: int = 3
    max_cores: int = 40
    recursion_probability: float = 0.55
    membership_probability: float = 0.55
    seed: int = 0

    def validate(self):
        for name in (
            "categories",
            "alternatives",
            "features",
            "max_depth",
            "max_cores",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError("grammar counts must be positive integers")
        if self.features < 2 or self.max_cores < 3 or self.max_depth > 12:
            raise ValueError("require features >= 2, max_cores >= 3, max_depth <= 12")
        if (
            not 0 <= self.recursion_probability <= 1
            or not 0 < self.membership_probability <= 1
        ):
            raise ValueError("invalid grammar probabilities")
        return self


class RandomGrammar:
    """Random productions within a fixed recursive/agreeing grammar family.

    S_a -> OPEN_a[k] BODY CLOSE_a[k]. Each frozen BODY alternative has 1–3
    child categories; each child either emits ATOM_b or recursively expands S_b.
    At max_depth all children are atoms. Lexical roles overlap; feature classes
    partition nodes so agreement is independently checkable on emitted tokens.
    """

    def __init__(self, n_nodes, config):
        self.config = c = config.validate()
        if n_nodes < c.features:
            raise ValueError("vocabulary must cover every agreement feature")
        rng = np.random.default_rng(c.seed + 2101)
        self.node_features = rng.permutation(np.arange(n_nodes) % c.features)
        self.membership = (
            rng.random((c.categories, 3, n_nodes)) < c.membership_probability
        )
        # Every role/category has support for every feature, including tiny tests.
        for category in range(c.categories):
            for role in range(3):
                for feature in range(c.features):
                    candidates = np.flatnonzero(self.node_features == feature)
                    self.membership[category, role, rng.choice(candidates)] = True
        self.emission_logits = rng.normal(size=(c.categories, 3, n_nodes))
        self.rules = [
            [
                rng.integers(c.categories, size=int(rng.integers(1, 4))).tolist()
                for _ in range(c.alternatives)
            ]
            for _ in range(c.categories)
        ]
        self.rule_probabilities = rng.dirichlet(
            np.ones(c.alternatives), size=c.categories
        )

    def save(self, path):
        Path(path).write_text(
            json.dumps(
                dict(
                    config=asdict(self.config),
                    node_features=self.node_features.tolist(),
                    membership=self.membership.tolist(),
                    emission_logits=self.emission_logits.tolist(),
                    rules=self.rules,
                    rule_probabilities=self.rule_probabilities.tolist(),
                ),
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def load(cls, path):
        fields = json.loads(Path(path).read_text())
        grammar = cls(len(fields["node_features"]), GrammarConfig(**fields["config"]))
        for name in (
            "node_features",
            "membership",
            "emission_logits",
            "rule_probabilities",
        ):
            setattr(grammar, name, np.asarray(fields[name]))
        grammar.rules = fields["rules"]
        return grammar

    def derive(self, rng, agreement=True):
        c = self.config
        terminals, spans = [], []

        def expand(category, depth):
            feature = int(rng.integers(c.features))
            start = len(terminals)
            terminals.append((category, 0, feature))
            alternative = int(
                rng.choice(c.alternatives, p=self.rule_probabilities[category])
            )
            span = dict(
                category=category, depth=depth, alternative=alternative, start=start
            )
            spans.append(span)
            for child in self.rules[category][alternative]:
                if depth < c.max_depth and rng.random() < c.recursion_probability:
                    expand(child, depth + 1)
                else:
                    terminals.append((child, 1, -1))
                if len(terminals) >= c.max_cores:
                    raise ValueError("grammar core budget exceeded")
            close_feature = feature if agreement else int(rng.integers(c.features))
            span.update(
                end=len(terminals), feature=feature, close_feature=close_feature
            )
            terminals.append((category, 2, close_feature))
            if len(terminals) > c.max_cores:
                raise ValueError("grammar core budget exceeded")

        expand(int(rng.integers(c.categories)), 1)
        return terminals, spans


class GrammarPrior:
    def __init__(self, prior, grammar, agreement=True):
        self.prior, self.grammar, self.agreement = prior, grammar, agreement
        self.graph, self.config = prior.graph, prior.config

    def draw_cores(self, rng):
        terminals, spans = self.grammar.derive(rng, self.agreement)
        c = self.config
        topic = rng.normal(size=c.topic_dim)
        cores, topics, distances, logprobs, retained = [], [], [], [], []
        for category, role, feature in terminals:
            if cores:
                topic = c.topic_persistence * topic + np.sqrt(
                    1 - c.topic_persistence**2
                ) * rng.normal(size=c.topic_dim)
            _, probability, distance = self.prior.probabilities(
                cores[-1] if cores else None, topic
            )
            allowed = self.grammar.membership[category, role].copy()
            if feature >= 0:
                allowed &= self.grammar.node_features == feature
            logits = self.grammar.emission_logits[category, role]
            retained.append(float(probability[allowed].sum()))
            weights = probability * allowed * np.exp(logits - logits.max())
            if weights.sum() <= 0:
                raise ValueError("grammar has no feasible lexical realization")
            weights /= weights.sum()
            node = int(rng.choice(self.graph.n_nodes, p=weights))
            if distance is not None:
                distances.append(int(distance[node]))
            cores.append(node)
            topics.append(topic.tolist())
            logprobs.append(float(np.log(weights[node])))
        return dict(
            cores=cores,
            topics=topics,
            distances=distances,
            core_logprob_given_topics_and_parse=logprobs,
            terminals=terminals,
            spans=spans,
            grammar_support_mass=retained,
        )


def build_grammar_dataset(prior, grammar, output, counts, *, agreement=True):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("output directory must be empty")
    prior.graph.save(root / "graph.npz")
    prior.save(root / "core_prior.npz")
    grammar.save(root / "grammar.json")
    width = 2 + (grammar.config.max_cores - 1) * prior.config.bridge_max_steps
    distribution = GrammarPrior(prior, grammar, agreement)
    rsp = bridge_rsp(prior.graph, prior.config)
    owners, statistics, records = {}, {}, {}
    for split_index, (split, count) in enumerate(counts.items()):
        rng = np.random.default_rng(grammar.config.seed + 2203 + split_index)
        writer = _Writer(root, split, width)
        accepted, rejected = [], Counter()
        with (root / f"{split}_parse.jsonl").open("w") as sidecar:
            for _ in range(count):
                for attempt in range(prior.config.max_attempts_per_example):
                    try:
                        sample = draw_document(distribution, rsp, rng)
                    except ValueError as error:
                        if str(error) not in {
                            "grammar core budget exceeded",
                            "grammar has no feasible lexical realization",
                            "core has no feasible successor in the distance band",
                        }:
                            raise
                        rejected[str(error)] += 1
                        continue
                    key = hashlib.sha256(
                        np.asarray(sample["path"], dtype="<i4").tobytes()
                    ).digest()
                    if key in owners and owners[key] != split:
                        rejected["cross-split duplicate"] += 1
                        continue
                    owners[key] = split
                    break
                else:
                    raise RuntimeError("could not generate feasible grammar document")
                row = np.zeros(width, dtype=np.int32)
                row[0] = SEP
                row[1 : len(sample["path"]) + 1] = (
                    np.asarray(sample["path"]) + N_SPECIAL
                )
                mask = np.ones(width, dtype=np.uint8)
                mask[0] = 0
                writer.append(
                    row,
                    mask,
                    [
                        sum(sample["bridge_logprob"]),
                        sum(sample["bridge_entropy_bits"]),
                        sum(sample["distances"]),
                        0,
                    ],
                )
                sidecar.write(json.dumps(sample) + "\n")
                accepted.append(sample)
        writer.finish()
        records[split] = accepted
        statistics[split] = dict(examples=count, rejections=dict(rejected))
    metadata = dict(
        format_version=3,
        task_mode="grammar_core",
        config=prior.graph.config.to_dict(),
        latent_config=asdict(prior.config),
        grammar_config=asdict(grammar.config),
        seq_len=width,
        counts=counts,
        agreement=agreement,
        split_statistics=statistics,
        model_fields=["input_ids", "attention_mask"],
        entropy_conditioning="bridge entropy given latent endpoints, not observed-sequence entropy",
        exact_visible_sequence_likelihood=False,
        split_policy="no identical observed documents across splits; shared frozen grammar",
    )
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return records, metadata


def probe_features(records, n_nodes, window, *, before_bridge=False):
    """Prefix-only token bags at known closing-core positions; no parse input.

    Evaluation uses hidden annotations to choose query positions/labels. This is
    a supervised diagnostic, not an end-to-end language-model measurement.
    """
    from scipy.sparse import csr_matrix

    rows, cols, values, labels, doc_ids = [], [], [], [], []
    for doc_id, sample in enumerate(records):
        for span in sample["spans"]:
            end = (
                sample["positions"][span["end"] - 1] + 1
                if before_bridge
                else sample["positions"][span["end"]]
            )
            prefix = sample["path"][:end]  # strictly excludes the target token
            prefix = prefix if window is None else prefix[-window:]
            for token, count in Counter(prefix).items():
                rows.append(len(labels))
                cols.append(token)
                values.append(count)
            labels.append(span["close_feature"])
            doc_ids.append(doc_id)
    matrix = csr_matrix(
        (values, (rows, cols)), shape=(len(labels), n_nodes), dtype=float
    )
    return matrix, np.asarray(labels), np.asarray(doc_ids)


def summarize(records, grammar):
    from .latent_diagnostics import mutual_information

    samples = [s for split in records.values() for s in split]
    spans = [(s, span) for s in samples for span in s["spans"]]
    train = records["train"]
    pairs = [(a, b) for s in train for a, b in zip(s["cores"], s["cores"][1:])]
    gap = [
        s["positions"][span["end"]] - s["positions"][span["start"]] for s, span in spans
    ]
    ngrams = {}
    for n in range(1, 6):
        counts = Counter(
            tuple(s["path"][i : i + n])
            for s in records["train"]
            for i in range(len(s["path"]) - n + 1)
        )
        ngrams[str(n)] = dict(unique=len(counts), total=sum(counts.values()))
    return dict(
        examples=len(samples),
        statistic_scope="all splits for structural metrics; training only for frequencies, MI and n-grams",
        core_frequencies=dict(Counter(node for s in train for node in s["cores"])),
        visible_token_frequencies=dict(
            Counter(node for s in train for node in s["path"])
        ),
        consecutive_core_mi_bits=mutual_information(pairs),
        mi_caveat="raw plug-in estimate with finite-sample bias; includes graph, grammar and topic dependencies",
        agreement_distance_visible_quantiles=dict(
            zip(("p50", "p90", "p99"), np.quantile(gap, [0.5, 0.9, 0.99]).tolist())
        ),
        fraction_agreement_gaps_over_16=float(np.mean(np.asarray(gap) > 16)),
        fraction_agreement_gaps_over_64=float(np.mean(np.asarray(gap) > 64)),
        core_count_histogram=dict(Counter(len(s["cores"]) for s in samples)),
        depth_histogram=dict(
            Counter(max(span["depth"] for span in s["spans"]) for s in samples)
        ),
        rule_counts=dict(
            Counter(f"{span['category']}:{span['alternative']}" for _, span in spans)
        ),
        agreement_rate=float(
            np.mean(
                [
                    grammar.node_features[s["cores"][span["start"]]]
                    == grammar.node_features[s["cores"][span["end"]]]
                    for s, span in spans
                ]
            )
        ),
        mean_agreement_distance_visible=float(
            np.mean(
                [
                    s["positions"][span["end"]] - s["positions"][span["start"]]
                    for s, span in spans
                ]
            )
        ),
        mean_core_distance=float(np.mean([d for s in samples for d in s["distances"]])),
        mean_bridge_entropy_bits=float(
            np.mean([h for s in samples for h in s["bridge_entropy_bits"]])
        ),
        mean_grammar_support_mass=float(
            np.mean([m for s in samples for m in s["grammar_support_mass"]])
        ),
        ngrams=ngrams,
    )


def run_probes(records, n_nodes, *, before_bridge=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    results = {}
    for window in (4, 16, 64, None):
        x, y, _ = probe_features(
            records["train"], n_nodes, window, before_bridge=before_bridge
        )
        xv, yv, _ = probe_features(
            records["validation"], n_nodes, window, before_bridge=before_bridge
        )
        xt, yt, _ = probe_features(
            records["test"], n_nodes, window, before_bridge=before_bridge
        )
        candidates = []
        for regularization in (0.01, 0.1, 1.0):
            model = LogisticRegression(C=regularization, max_iter=500)
            model.fit(x, y)
            candidates.append(
                (
                    log_loss(yv, model.predict_proba(xv), labels=model.classes_),
                    regularization,
                    model,
                )
            )
        _, chosen, model = min(candidates, key=lambda item: item[0])
        results[str(window or "full")] = dict(
            test_accuracy=float(model.score(xt, yt)),
            test_logloss_bits=float(
                log_loss(yt, model.predict_proba(xt), labels=model.classes_) / np.log(2)
            ),
            C=chosen,
            train_queries=len(y),
            test_queries=len(yt),
            majority_test_accuracy=float(np.mean(yt == np.bincount(y).argmax())),
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--eval", type=int, default=250)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="recompute reports from saved corpora",
    )
    args = parser.parse_args()
    if min(args.train, args.eval) < 1:
        parser.error("train and eval counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    if args.analyze_only:
        for dataset in sorted(args.output.glob("seed*_*")):
            grammar = RandomGrammar.load(dataset / "grammar.json")
            records = {
                split: [
                    json.loads(line)
                    for line in (dataset / f"{split}_parse.jsonl")
                    .read_text()
                    .splitlines()
                ]
                for split in ("train", "validation", "test")
            }
            metadata = json.loads((dataset / "metadata.json").read_text())
            report = summarize(records, grammar)
            report["split_statistics"] = metadata["split_statistics"]
            report["prefix_probes"] = run_probes(records, len(grammar.node_features))
            report["before_bridge_probes"] = run_probes(
                records, len(grammar.node_features), before_bridge=True
            )
            reports[dataset.name] = report
            print(
                json.dumps(
                    dict(
                        variant=dataset.name,
                        before_bridge_probes=report["before_bridge_probes"],
                    )
                ),
                flush=True,
            )
        (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
        return
    for seed in args.seeds:
        gc = replace(
            preset("tiny_debug"), n_train=args.train, n_val=args.eval, n_test=args.eval
        )
        lc = LatentCoreConfig(
            seed=seed, pilot_paths=512, bridge_max_steps=8, max_distance=8
        )
        graph = generate_graph(gc)
        prior = CorePrior(graph, lc, estimate_bridge_visitation(graph, lc))
        grammar = RandomGrammar(graph.n_nodes, GrammarConfig(seed=seed))
        for agreement in (True, False):
            name = f"seed{seed}_{'grammar' if agreement else 'broken_agreement'}"
            records, metadata = build_grammar_dataset(
                prior,
                grammar,
                args.output / name,
                dict(train=args.train, validation=args.eval, test=args.eval),
                agreement=agreement,
            )
            report = summarize(records, grammar)
            report["split_statistics"] = metadata["split_statistics"]
            report["prefix_probes"] = run_probes(records, graph.n_nodes)
            report["before_bridge_probes"] = run_probes(
                records, graph.n_nodes, before_bridge=True
            )
            reports[name] = report
            (args.output / "report.json").write_text(
                json.dumps(reports, indent=2) + "\n"
            )
            print(
                json.dumps(
                    dict(
                        variant=name,
                        agreement_rate=report["agreement_rate"],
                        probes=report["prefix_probes"],
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
