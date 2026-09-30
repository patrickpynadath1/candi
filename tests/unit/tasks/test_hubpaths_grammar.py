from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tasks.hubpaths.config import preset
from tasks.hubpaths.data import HubPathsDataset, get_tokenizer
from tasks.hubpaths.grammar import (
    GrammarConfig,
    GrammarPrior,
    RandomGrammar,
    build_grammar_dataset,
    probe_features,
    summarize,
)
from tasks.hubpaths.graph import generate_graph
from tasks.hubpaths.latent import CorePrior, LatentCoreConfig, bridge_rsp, draw_document


def make_prior():
    graph = generate_graph(preset("tiny_debug"))
    config = LatentCoreConfig(bridge_max_steps=8, max_distance=8)
    return CorePrior(graph, config, np.ones(graph.n_nodes, dtype=int))


def test_recursive_derivation_bounded_and_frozen():
    config = GrammarConfig(max_cores=100, recursion_probability=1)
    a, b = RandomGrammar(200, config), RandomGrammar(200, config)
    assert a.rules == b.rules
    np.testing.assert_array_equal(a.membership, b.membership)
    terminals, spans = a.derive(np.random.default_rng(1))
    assert max(s["depth"] for s in spans) == 3
    for span in spans:
        start, end = span["start"], span["end"]
        assert terminals[start] == (span["category"], 0, span["feature"])
        assert terminals[end] == (span["category"], 2, span["feature"])
        for other in spans:
            assert not start < other["start"] < end < other["end"]
    shallow = RandomGrammar(200, replace(config, max_depth=1))
    assert len(shallow.derive(np.random.default_rng(1))[1]) == 1


def test_agreement_control_and_valid_paths():
    prior = make_prior()
    grammar = RandomGrammar(200, GrammarConfig(recursion_probability=0))
    rsp = bridge_rsp(prior.graph, prior.config)
    rng = np.random.default_rng(5)
    for agreement in (True, False):
        matches = []
        for _ in range(100):
            sample = draw_document(GrammarPrior(prior, grammar, agreement), rsp, rng)
            nodes = np.asarray(sample["path"])
            assert np.isin(nodes[:-1] * 200 + nodes[1:], prior.graph.edge_keys).all()
            np.testing.assert_array_equal(nodes[sample["positions"]], sample["cores"])
            for node, (category, role, feature) in zip(
                sample["cores"], sample["terminals"]
            ):
                assert grammar.membership[category, role, node]
                if feature >= 0:
                    assert grammar.node_features[node] == feature
            for span in sample["spans"]:
                matches.append(span["feature"] == span["close_feature"])
        assert np.mean(matches) == 1 if agreement else 0.1 < np.mean(matches) < 0.4


def test_corpus_reproducible_hidden_parse_and_prefix_probe(tmp_path):
    prior = make_prior()
    grammar = RandomGrammar(200, GrammarConfig(max_depth=2, max_cores=24))
    counts = dict(train=12, validation=4, test=4)
    records, metadata = build_grammar_dataset(prior, grammar, tmp_path / "a", counts)
    build_grammar_dataset(prior, grammar, tmp_path / "b", counts)
    for file in (tmp_path / "a").iterdir():
        assert file.read_bytes() == (tmp_path / "b" / file.name).read_bytes()
    loaded = RandomGrammar.load(tmp_path / "a" / "grammar.json")
    np.testing.assert_array_equal(loaded.emission_logits, grammar.emission_logits)
    assert loaded.derive(np.random.default_rng(11)) == grammar.derive(
        np.random.default_rng(11)
    )
    sets = []
    for split in counts:
        dataset = HubPathsDataset(tmp_path / "a", split, view="ar")
        assert set(dataset[0]) == {"input_ids", "attention_mask"}
        assert (dataset.tokens[:, 0] == 1).all()
        assert not np.isin(dataset.tokens[:, 1:], [1, 2]).any()
        sets.append({row.tobytes() for row in dataset.tokens})
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    config = SimpleNamespace(
        data=SimpleNamespace(prepared_dir=str(tmp_path / "a")),
        model=SimpleNamespace(length=metadata["seq_len"]),
    )
    assert get_tokenizer(config).prompt_len == 1
    # A unique target/suffix token must not appear in prefix features.
    dummy = [
        dict(
            path=[7, 8, 99, 100], positions=[0, 2], spans=[dict(end=1, close_feature=3)]
        )
    ]
    x, labels, _ = probe_features(dummy, 200, None)
    assert x[0, 7] == x[0, 8] == 1
    assert x[0, 99] == x[0, 100] == 0
    assert labels.tolist() == [3]
    before, _, _ = probe_features(dummy, 200, None, before_bridge=True)
    assert before[0, 7] == 1 and before[0, 8] == 0
    report = summarize(records, grammar)
    assert report["agreement_rate"] == 1
    assert report["ngrams"]["1"]["total"] == sum(
        len(s["path"]) for s in records["train"]
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(features=1),
        dict(max_depth=13),
        dict(max_cores=2),
        dict(recursion_probability=1.1),
    ],
)
def test_invalid_grammar(change):
    with pytest.raises(ValueError):
        replace(GrammarConfig(), **change).validate()
