from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp

from tasks.hubpaths.config import HubPathsConfig, preset
from tasks.hubpaths.data import HubPathsDataset, get_tokenizer
from tasks.hubpaths.diagnostics import diagnose
from tasks.hubpaths.graph import Graph, N_SPECIAL, PAD, SEP
from tasks.hubpaths.latent import (
    CorePrior,
    LatentCoreConfig,
    bridge_rsp,
    build_latent_dataset,
    draw_document,
    estimate_bridge_visitation,
)
from tasks.hubpaths.latent_diagnostics import mutual_information
from tasks.hubpaths.verify import verify_path


def line_graph():
    config = replace(HubPathsConfig(), n_nodes=7, max_steps=5, min_dist=1, max_dist=5)
    return Graph.from_edges(7, [(i, i + 1) for i in range(6)], config)


def small_config(**kw):
    values = dict(
        n_cores=4,
        bridge_max_steps=4,
        min_distance=1,
        max_distance=3,
        pilot_paths=24,
        pilot_starts_per_goal=4,
        correlation_rank=2,
        topic_dim=2,
    )
    values.update(kw)
    return LatentCoreConfig(**values)


def test_soft_avoidance_and_frozen_low_rank(tmp_path):
    graph = line_graph()
    c = small_config(baseline_log_std=0.0, topic_strength=0.0, correlation_strength=0.0)
    counts = np.array([0, 1, 10, 100, 10, 1, 0])
    prior = CorePrior(graph, c, counts)
    assert np.all(prior.base_probabilities > 0)
    assert (
        prior.base_probabilities[0]
        > prior.base_probabilities[2]
        > prior.base_probabilities[3]
    )
    assert np.array_equal(
        prior.source_factors, CorePrior(graph, c, counts).source_factors
    )
    assert (
        np.linalg.matrix_rank(prior.source_factors @ prior.target_factors.T)
        <= c.correlation_rank
    )
    prior.save(tmp_path / "prior.npz")
    loaded = CorePrior.load(graph, tmp_path / "prior.npz")
    for before, after in zip(
        prior.probabilities(0, np.zeros(2))[:2],
        loaded.probabilities(0, np.zeros(2))[:2],
    ):
        np.testing.assert_array_equal(before, after)
    with np.load(tmp_path / "prior.npz") as a:
        assert all(a[k].shape != (7, 7) for k in a.files)


def test_distance_filter_changes_prior():
    graph = line_graph()
    c = small_config(
        min_distance=2,
        max_distance=2,
        baseline_log_std=0.0,
        bridge_avoidance=0.0,
        topic_strength=0.0,
        correlation_strength=0.0,
    )
    prior = CorePrior(graph, c, np.zeros(7, dtype=int))
    before, after, d = prior.probabilities(0, np.zeros(2))
    np.testing.assert_allclose(before, np.ones(7) / 7)
    assert after[2] == 1 and np.count_nonzero(after) == 1
    assert d[2] == 2
    assert np.abs(before - after).sum() / 2 > 0.8


def test_pair_factors_change_preferences_but_are_repeatable():
    graph = line_graph()
    c = small_config(min_distance=1, max_distance=4)
    prior = CorePrior(graph, c, np.zeros(7, dtype=int))
    first = prior.probabilities(0, np.zeros(2))[0]
    second = prior.probabilities(1, np.zeros(2))[0]
    assert not np.allclose(first, second)
    np.testing.assert_array_equal(first, prior.probabilities(0, np.zeros(2))[0])
    assert not np.allclose(first, prior.probabilities(0, np.ones(2))[0])


def test_neutral_pilot_only_counts_interiors():
    graph = line_graph()
    c = small_config(bridge_max_steps=1, min_distance=1, max_distance=1)
    counts = estimate_bridge_visitation(graph, c)
    assert counts.sum() == 0  # every bridge consists of two endpoints only


def test_bridge_entropy_and_topic_persistence():
    graph = line_graph()
    c = small_config(n_cores=500, topic_persistence=0.98)
    prior = CorePrior(graph, c, np.ones(7, dtype=int))
    rsp = bridge_rsp(graph, c)
    sample = draw_document(prior, rsp, np.random.default_rng(321))
    topics = np.asarray(sample["topics"])
    assert np.corrcoef(topics[:-1].ravel(), topics[1:].ravel())[0, 1] > 0.90
    assert len(sample["positions"]) == c.n_cores
    for i in range(c.n_cores - 1):
        path = sample["path"][sample["positions"][i] : sample["positions"][i + 1] + 1]
        start, goal = sample["cores"][i : i + 2]
        assert verify_path(
            graph,
            start,
            goal,
            np.asarray(path) + N_SPECIAL,
            max_steps=c.bridge_max_steps,
        ).valid
        assert sample["bridge_logprob"][i] == pytest.approx(
            rsp.true_logprob(start, goal, np.asarray(path) + N_SPECIAL)
        )
        assert sample["bridge_entropy_bits"][i] == pytest.approx(
            rsp.cond_entropy_bits(start, goal)
        )


def test_known_mutual_information():
    assert mutual_information([[0, 0], [0, 1], [1, 0], [1, 1]]) == pytest.approx(0)
    assert mutual_information([[0, 0], [1, 1]] * 100) == pytest.approx(1)
    assert mutual_information([]) == 0


def test_latent_dataset_roundtrip_and_diagnostics(tmp_path):
    gc = replace(preset("tiny_debug"), n_train=30, n_val=8, n_test=8)
    lc = small_config()
    a, b = tmp_path / "a", tmp_path / "b"
    metadata = build_latent_dataset(gc, lc, a)
    build_latent_dataset(gc, lc, b)
    assert metadata["seq_len"] == 14
    for file in a.rglob("*"):
        if file.is_file():
            assert file.read_bytes() == (b / file.relative_to(a)).read_bytes()
    graph = Graph.load(a / "graph.npz")
    sequences = []
    for split in ("train", "validation", "test"):
        ds = HubPathsDataset(a, split, view="ar")
        assert set(ds[0]) == {"input_ids", "attention_mask"}
        assert ds.tokens.shape[1] == lc.seq_len
        assert (ds.tokens[:, 0] == SEP).all()
        assert not np.isin(ds.tokens[:, 1:], [SEP, 2]).any()
        assert (ds.masks[:, 0] == 0).all() and (ds.masks[:, 1:] == 1).all()
        cores = np.load(a / f"{split}_latent_cores.npy")
        positions = np.load(a / f"{split}_latent_positions.npy")
        for row, core, pos in zip(ds.tokens, cores, positions):
            path = row[1:][row[1:] != PAD] - N_SPECIAL
            np.testing.assert_array_equal(path[pos], core)
            keys = path[:-1] * graph.n_nodes + path[1:]
            assert np.isin(keys, graph.edge_keys).all()
        sequences.append({row.tobytes() for row in ds.tokens})
    assert not (
        sequences[0] & sequences[1]
        or sequences[0] & sequences[2]
        or sequences[1] & sequences[2]
    )
    config = SimpleNamespace(
        data=SimpleNamespace(prepared_dir=str(a)),
        model=SimpleNamespace(length=lc.seq_len),
    )
    tokenizer = get_tokenizer(config)
    assert tokenizer.prompt_len == 1 and tokenizer.total_seq_len == lc.seq_len
    assert tokenizer.decode([SEP]) == "<BOS>"
    report = diagnose(a, ngrams=(1, 2), entropy_prompts=3)
    latent = report["latent_core"]
    assert report["token_count"] == int(
        (np.load(a / "train_tokens.npy")[:, 1:] != PAD).sum()
    )
    assert latent["core_occurrences"] == gc.n_train * lc.n_cores
    assert latent["consecutive_core_pairs"] == gc.n_train * (lc.n_cores - 1)
    assert latent["connecting_path_entropy_bits"]["mean"] >= 0
    assert not report["exact_walk_statistics"]
    assert (a / "diagnostics_train/core_frequencies.tsv").exists()
    assert np.isfinite(latent["consecutive_core_mi_bits"])
    with pytest.raises(ValueError, match="no model-visible"):
        diagnose(a, output=tmp_path / "bad", scope="prompt")
    # Partial reports recompute expectation sums over just those contexts.
    partial = diagnose(
        a, output=tmp_path / "partial", max_examples=4, entropy_prompts=0
    )
    assert partial["latent_core"]["examples"] == 4


def test_bridge_entropy_against_enumeration():
    graph = line_graph()
    c = small_config(bridge_max_steps=4)
    rsp = bridge_rsp(graph, c)
    paths = [[1, 2, 3], [1, 0, 1, 2, 3], [1, 2, 1, 2, 3]]
    logweights = []
    for path in paths:
        # Uniform neighbor reference, unit costs, temperature=1.
        logweights.append(
            sum(
                -np.log(graph.adjacency.indptr[a + 1] - graph.adjacency.indptr[a]) - 1
                for a in path[:-1]
            )
        )
    probs = np.exp(logweights - logsumexp(logweights))
    assert rsp.cond_entropy_bits(1, 3) == pytest.approx(
        -np.sum(probs * np.log2(probs)), abs=1e-10
    )


@pytest.mark.parametrize(
    "override",
    [
        dict(topic_persistence=1.0),
        dict(correlation_rank=0),
        dict(bridge_pseudocount=0.0),
        dict(max_distance=11),
        dict(bridge_avoidance=-1.0),
    ],
)
def test_bad_config(override):
    with pytest.raises(ValueError):
        replace(LatentCoreConfig(), **override).validate()
