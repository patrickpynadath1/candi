from collections import Counter
from dataclasses import replace
import itertools

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import chisquare
import torch

from tasks.hubpaths import (
    HubPathsConfig,
    Graph,
    RSP,
    preset,
    generate_graph,
    build_dataset,
)
from tasks.hubpaths.data import HubPathsDataset, PairSplits, hints_from_row
from tasks.hubpaths.diagnostics import diagnose
from tasks.hubpaths.graph import N_SPECIAL, PAD, SEP
from tasks.hubpaths.routes import RouteDistribution, route_shortest
from tasks.hubpaths.verify import (
    Reason,
    verify_path,
    verify_batch,
    verify_route,
    verify_route_batch,
)
from tasks.hubpaths.evaluation import evaluate_samples


def toy(**kwargs):
    c = replace(
        HubPathsConfig(),
        n_nodes=10,
        max_steps=5,
        min_dist=1,
        max_dist=5,
        n_waypoints=2,
        **kwargs,
    )
    return Graph.from_edges(10, [(4, 9), (9, 2), (4, 7), (7, 2), (7, 5), (5, 2)], c)


def encode(path):
    return [p + N_SPECIAL for p in path]


def enumerate_walks(graph, points, horizon, exact=False, hints=None):
    result = []

    def visit(path, progress):
        i = path[-1]
        if hints and len(path) - 1 in hints and hints[len(path) - 1] != i:
            return
        if progress == len(points):
            if (not exact or len(path) - 1 == horizon) and not any(
                p >= len(path) for p in (hints or {})
            ):
                result.append(path)
            return
        if len(path) - 1 == horizon:
            return
        for j in graph.adjacency.indices[
            graph.adjacency.indptr[i] : graph.adjacency.indptr[i + 1]
        ]:
            if j not in points[progress + 1 :]:
                visit(path + [int(j)], progress + int(j == points[progress]))

    visit([points[0]], 1)
    return result


def brute_probabilities(graph, paths):
    # Compute independently from the dense adjacency and the defining formula.
    a = graph.adjacency.toarray()
    pref = a * graph.weights[None, :] ** graph.config.alpha
    pref /= np.where(pref.sum(1, keepdims=True) > 0, pref.sum(1, keepdims=True), 1)
    weights = pref * np.exp(
        -(graph.config.edge_cost + graph.config.hub_cost * graph.is_hub[None, :])
        / graph.config.temperature
    )
    logs = np.array(
        [sum(np.log(weights[a, b]) for a, b in zip(p, p[1:])) for p in paths]
    )
    return np.exp(logs - logsumexp(logs))


@pytest.mark.parametrize("points", [(4, 2), (4, 7, 2), (4, 9, 7, 2)])
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("hints", [None, {1: 9}])
def test_enumeration(points, exact, hints):
    graph = toy(length_mode="exact" if exact else "at_most")
    paths = enumerate_walks(graph, points, 5, exact, hints)
    route = RouteDistribution(RSP(graph), points, hints)
    if not paths:
        assert route.log_partition == -np.inf
        return
    probs = brute_probabilities(graph, paths)
    np.testing.assert_allclose(
        [route.true_logprob(encode(p)) for p in paths], np.log(probs), atol=1e-9, rtol=0
    )
    assert abs(route.cond_entropy_bits() + np.sum(probs * np.log2(probs))) < 1e-9
    first, lengths = np.zeros(graph.n_nodes), np.zeros(6)
    for path, probability in zip(paths, probs):
        first[path[1]] += probability
        lengths[len(path) - 1] += probability
    np.testing.assert_allclose(route.first_step_marginal(), first, atol=1e-9, rtol=0)
    np.testing.assert_allclose(route.length_distribution(), lengths, atol=1e-9, rtol=0)
    if len(points) == 2:
        rsp = RSP(graph)
        np.testing.assert_allclose(
            [rsp.true_logprob(*points, encode(p), hints) for p in paths],
            np.log(probs),
            atol=1e-9,
            rtol=0,
        )
        assert (
            abs(rsp.cond_entropy_bits(*points, hints) - route.cond_entropy_bits())
            < 1e-9
        )
        np.testing.assert_allclose(
            rsp.length_distribution(*points, hints), lengths, atol=1e-9
        )


def test_tiny_enumeration():
    graph = generate_graph(
        replace(
            preset("tiny_debug"), max_steps=3, min_dist=1, max_dist=3, n_waypoints=2
        )
    )
    a = 0
    b = int(graph.adjacency.indices[graph.adjacency.indptr[a]])
    paths = enumerate_walks(graph, (a, b), 3)
    probs = brute_probabilities(graph, paths)
    route = RouteDistribution(RSP(graph), (a, b))
    np.testing.assert_allclose(
        [route.true_logprob(encode(p)) for p in paths], np.log(probs), atol=1e-9
    )
    assert abs(route.cond_entropy_bits() + np.sum(probs * np.log2(probs))) < 1e-9


@pytest.mark.parametrize("temperature", [1.0, 1e8])
def test_sampler_chisquare(temperature):
    graph = toy(temperature=temperature)
    points = (4, 7, 2)
    paths = enumerate_walks(graph, points, 5)
    probabilities = brute_probabilities(graph, paths)
    route = RouteDistribution(RSP(graph), points)
    rng = np.random.default_rng(123)
    counts = Counter(tuple(route.sample(rng)) for _ in range(20000))
    observed = np.array([counts[tuple(p)] for p in paths])
    assert chisquare(observed, probabilities * observed.sum()).pvalue > 1e-4


def test_low_temperature_and_global_budget():
    graph = toy(temperature=0.05)
    graph.config = replace(graph.config, max_steps=128)
    route = RouteDistribution(RSP(graph), (4, 9, 7, 2))
    assert np.isfinite(route.log_partition)
    rng = np.random.default_rng(1)
    shortest, _ = route_shortest(graph, route.waypoints)
    assert all(len(route.sample(rng)) - 1 == shortest for _ in range(200))
    assert route.length_distribution().sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "path,reason,kwargs",
    [
        ([], Reason.MALFORMED, {}),
        ([4, 2], Reason.NOT_AN_EDGE, {}),
        ([9, 2], Reason.WRONG_START, {}),
        ([4, 7, 5], Reason.WRONG_END, {}),
        ([4, 7, 2, 5, 2], Reason.GOAL_BEFORE_END, {}),
        ([4, 9, 4, 7, 2], Reason.REPEATED_NODE, {"require_simple": True}),
        ([4, 9, 4, 7, 2], Reason.TOO_LONG, {"max_steps": 3}),
        ([4, 7, 2], Reason.HINT_MISMATCH, {"hints": {1: 9}}),
    ],
)
def test_verifier_reasons(path, reason, kwargs):
    result = verify_path(toy(), 4, 2, encode(path), **{"max_steps": 5, **kwargs})
    assert result.reason == reason


def test_waypoint_order_missing_and_padding():
    graph = toy()
    assert verify_route(graph, [4, 9, 7, 2], encode([4, 9, 4, 7, 2]), max_steps=5).valid
    assert (
        verify_route(graph, [4, 9, 7, 2], encode([4, 7, 4, 9, 2]), max_steps=5).reason
        == Reason.WAYPOINT_ORDER
    )
    assert (
        verify_route(graph, [4, 9, 2], encode([4, 7, 2]), max_steps=5).reason
        == Reason.WAYPOINT_MISSING
    )
    assert (
        verify_path(
            graph, 4, 2, encode([4, 7]) + [PAD] + encode([2]), max_steps=5
        ).reason
        == Reason.MALFORMED
    )
    assert verify_path(graph, 4, 2, encode([4, 7, 2]) + [PAD] * 2, max_steps=5).valid


def test_batch_10000():
    graph = toy()
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 14, size=(10000, 9))
    rows[:, :3] = [7, 5, SEP]
    rows[:1000, 3:] = [7, 10, 5, PAD, PAD, PAD]
    for i in range(1000, 2000):
        rows[i] = rows[0]
        rows[i, rng.integers(3, 9)] = rng.integers(0, 14)
    valid, reasons = verify_batch(graph, torch.tensor(rows), max_steps=5)
    expected = [verify_path(graph, 4, 2, row[3:], max_steps=5) for row in rows]
    assert valid.tolist() == [r.valid for r in expected]
    assert reasons.tolist() == [r.reason for r in expected]
    prompt = torch.tensor([[7, 12, 5]])
    routes = np.concatenate([rows[:, :1], np.full((10000, 1), 12), rows[:, 1:]], axis=1)
    valid, reasons = verify_route_batch(graph, routes, n_waypoints=3, max_steps=5)
    expected = [verify_route(graph, [4, 9, 2], row[4:], max_steps=5) for row in routes]
    assert reasons.tolist() == [r.reason for r in expected]
    _, altered = verify_route_batch(
        graph, routes[:1], prompts=prompt + 1, n_waypoints=3
    )
    assert altered.item() == Reason.PROMPT_ALTERED


@pytest.mark.parametrize(
    "backbone", ["ring", "chain", "tree", "grid", "random", "complete"]
)
def test_graph(backbone):
    import networkx as nx

    c = replace(
        preset("tiny_debug"),
        backbone=backbone,
        hub_link_prob=0.1,
        backbone_link_prob=0.1,
        n_levels=0 if backbone == "grid" else 1,
    )
    graph = generate_graph(c)
    assert graph.n_nodes == c.n_nodes
    assert nx.is_connected(nx.from_scipy_sparse_array(graph.adjacency))
    assert not graph.adjacency.diagonal().any()
    cross = graph.cluster[graph.rows] != graph.cluster[graph.adjacency.indices]
    assert graph.is_hub[graph.rows[cross]].all()
    assert graph.is_hub[graph.adjacency.indices[cross]].all()


def test_dataset_reproducibility_splits_and_diagnostics(tmp_path):
    c = replace(
        preset("tiny_debug"), n_train=80, n_val=8, n_test=8, n_hints=1, min_edge_count=0
    )
    first, second = tmp_path / "a", tmp_path / "b"
    build_dataset(c, first)
    build_dataset(c, second)
    for file in first.rglob("*"):
        if file.is_file():
            assert file.read_bytes() == (second / file.relative_to(first)).read_bytes()
    graph = Graph.load(first / "graph.npz")
    sets = []
    for split in ["train", "validation", "test", "test_ood"]:
        ds = HubPathsDataset(first, split)
        keys = set()
        for i, row in enumerate(ds.tokens):
            points = row[: c.n_waypoints] - N_SPECIAL
            keys.add(tuple(sorted((int(points[0]), int(points[-1])))))
            assert verify_route(
                graph,
                points,
                row[c.n_waypoints + 1 :],
                max_steps=c.max_steps,
                hints=hints_from_row(row, ds.masks[i], c.n_waypoints),
            ).valid
            if split == "train":
                assert PairSplits(graph).split(points[0], points[-1]) == "train"
        sets.append(keys)
        assert not ds[0]["attention_mask"][: c.n_waypoints + 1].any()
        assert torch.all(ds.infilling_input(0)[ds[0]["attention_mask"].bool()] == 2)
    assert all(not (a & b) for a, b in itertools.combinations(sets, 2))
    report = diagnose(first, ngrams=(1, 2, 3), entropy_prompts=4)
    tokens = np.load(first / "train_tokens.npy")
    brute = Counter()
    for row in tokens:
        path = [int(t) for t in row[c.n_waypoints + 1 :] if t != PAD]
        brute.update(zip(path, path[1:]))
    assert report["ngrams"]["2"]["unique"] == len(brute)
    assert report["ngrams"]["2"]["total"] == sum(brute.values())
    with pytest.raises(ValueError, match="n_hints=0"):
        HubPathsDataset(first, view="ar")


def test_sample_metrics():
    graph = toy()
    graph.config = replace(graph.config, n_waypoints=3)
    route = RouteDistribution(RSP(graph), [4, 7, 2])
    rng = np.random.default_rng(1)
    samples = []
    for _ in range(16):
        path = encode(route.sample(rng))
        samples.append(encode([4, 7, 2]) + [SEP] + path + [PAD] * (6 - len(path)))
    result = evaluate_samples(
        graph,
        np.array([encode([4, 7, 2])]),
        np.array([samples]),
        nll_bits=[route.cond_entropy_bits()],
    )
    assert result["validity"] == 1.0
    assert result["kl_bits"] == 0.0
    assert result["unique_valid@K"] > 0
    assert 0 <= result["first_step_tv"] <= 1


def test_batched_suffix_tables_and_disk_cache(tmp_path):
    graph = toy()
    rsp = RSP(graph)
    points = [(4, 7, 2), (9, 5, 2), (7, 9, 2)]
    batched = RouteDistribution.batch(rsp, points)
    for route, prompt in zip(batched, points):
        direct = RouteDistribution(rsp, prompt)
        np.testing.assert_allclose(route.z, direct.z, atol=1e-12)
        np.testing.assert_allclose(route.entropies, direct.entropies, atol=1e-12)
    route = batched[0]
    np.save(tmp_path / "suffix_7_2.npy", route.z)
    np.save(tmp_path / "suffix_7_2_entropy.npy", route.entropies)
    loaded = RouteDistribution(RSP(graph, cache_dir=tmp_path), points[0])
    assert isinstance(loaded.z, np.memmap)
    assert loaded.log_partition == route.log_partition


def test_no_valid_samples_report_null_metrics():
    graph = toy()
    samples = np.zeros((1, 16, 9), dtype=int)
    samples[:, :, :3] = [7, 5, SEP]
    result = evaluate_samples(graph, np.array([[7, 5]]), samples)
    assert result["validity"] == 0
    assert result["mean_stretch"] is None
    assert result["first_step_tv"] is None
