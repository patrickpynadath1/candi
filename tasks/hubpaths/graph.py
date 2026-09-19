"""Seeded hierarchical Chung–Lu graph with portable deterministic storage."""

import io
import json
import zipfile
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from .config import HubPathsConfig

PAD, SEP, MASK, N_SPECIAL = 0, 1, 2, 3


def save_npz(path, **arrays):
    # Fixed ZIP timestamps make reproducibility independent of wall-clock time.
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, value in sorted(arrays.items()):
            buf = io.BytesIO()
            np.save(buf, np.asarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, buf.getvalue())


@dataclass
class Graph:
    adjacency: csr_matrix
    weights: np.ndarray
    cluster: np.ndarray
    level: np.ndarray
    is_hub: np.ndarray
    config: HubPathsConfig

    def __post_init__(self):
        self.adjacency.sort_indices()
        self.n_nodes = self.adjacency.shape[0]
        self.rows = np.repeat(np.arange(self.n_nodes), np.diff(self.adjacency.indptr))
        self.edge_keys = self.rows * self.n_nodes + self.adjacency.indices

    def distances(self, goal):
        return shortest_path(
            self.adjacency, directed=False, unweighted=True, indices=goal
        )

    def save(self, path):
        save_npz(
            path,
            indptr=self.adjacency.indptr,
            indices=self.adjacency.indices,
            edge_keys=self.edge_keys.astype(np.int64),
            weights=self.weights,
            cluster=self.cluster,
            level=self.level,
            is_hub=self.is_hub,
            config=json.dumps(self.config.to_dict(), sort_keys=True),
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as a:
            c = HubPathsConfig(**json.loads(str(a["config"])))
            adj = csr_matrix(
                (np.ones(len(a["indices"])), a["indices"], a["indptr"]),
                shape=(c.n_nodes, c.n_nodes),
            )
            return cls(adj, a["weights"], a["cluster"], a["level"], a["is_hub"], c)

    @classmethod
    def from_edges(cls, n_nodes, edges, config=None):
        c = config or HubPathsConfig(n_nodes=n_nodes)
        pairs = sorted({(u, v) for a, b in edges for u, v in [(a, b), (b, a)]})
        rows, cols = np.array(pairs).T
        adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes))
        return cls(
            adj,
            np.ones(n_nodes),
            np.zeros(n_nodes, dtype=int),
            np.zeros(n_nodes, dtype=int),
            np.zeros(n_nodes, dtype=bool),
            c,
        )


def generate_graph(config):
    c = config.validate()
    rng = np.random.default_rng(c.seed)
    units = c.n_clusters
    routers = 0
    for _ in range(c.n_levels):
        units = (units + c.branching - 1) // c.branching
        routers += units * c.hubs_per_group
    leaves = c.n_nodes - routers
    minimum = max(2, c.hubs_per_cluster)
    if leaves < minimum * c.n_clusters:
        raise ValueError("n_nodes is too small for clusters and router hierarchy")
    proportions = (
        np.ones(c.n_clusters)
        if c.cluster_size_dist == "equal"
        else rng.pareto(c.gamma - 1, c.n_clusters) + 1
    )
    sizes = minimum + np.floor(
        (leaves - minimum * c.n_clusters) * proportions / proportions.sum()
    ).astype(int)
    sizes[-1] += leaves - sizes.sum()
    g = nx.Graph()
    g.add_nodes_from(range(c.n_nodes))
    weights = np.ones(c.n_nodes)
    cluster = np.full(c.n_nodes, -1, dtype=int)
    level = np.zeros(c.n_nodes, dtype=int)
    is_hub = np.zeros(c.n_nodes, dtype=bool)
    units = []
    offset = 0
    for ci, size in enumerate(sizes):
        ids = np.arange(offset, offset + size)
        upper = min(c.max_degree or size - 1, size - 1)
        lower = min(c.min_degree, upper)
        q = 1 - c.gamma
        w = (lower**q + rng.random(size) * (upper**q - lower**q)) ** (1 / q)
        w *= c.mean_degree / w.mean()
        local = nx.expected_degree_graph(
            w, selfloops=False, seed=int(rng.integers(2**32))
        )
        components = sorted(nx.connected_components(local), key=lambda x: min(x))
        connected = sorted(components[0])
        for component in components[1:]:
            component = sorted(component)
            a = rng.choice(component, p=w[component] / w[component].sum())
            b = rng.choice(connected, p=w[connected] / w[connected].sum())
            local.add_edge(int(a), int(b))
            connected.extend(component)
        g.add_edges_from((int(a + offset), int(b + offset)) for a, b in local.edges)
        hubs = ids[np.argsort(-w, kind="stable")[: c.hubs_per_cluster]].tolist()
        is_hub[hubs] = True
        weights[ids], cluster[ids] = w, ci
        units.append(hubs)
        offset += size

    def link(left, right, probability):
        made = False
        for a in left:
            for b in right:
                if rng.random() < probability:
                    g.add_edge(a, b)
                    made = True
        if not made:
            g.add_edge(int(rng.choice(left)), int(rng.choice(right)))

    for lev in range(1, c.n_levels + 1):
        parents = []
        for at in range(0, len(units), c.branching):
            hubs = list(range(offset, offset + c.hubs_per_group))
            offset += len(hubs)
            is_hub[hubs], level[hubs] = True, lev
            children = units[at : at + c.branching]
            for child in children:
                link(hubs, child, c.hub_link_prob)
            # Join router siblings so even sparse attachment cannot disconnect children.
            g.add_edges_from(zip(hubs[:-1], hubs[1:]))
            weights[hubs] = max(weights[np.concatenate(children)])
            parents.append(hubs)
        units = parents
    n = len(units)
    pairs = {(i, i + 1) for i in range(n - 1)}
    if c.backbone == "ring" and n > 2:
        pairs.add((0, n - 1))
    elif c.backbone == "tree":
        pairs = {((i - 1) // c.branching, i) for i in range(1, n)}
    elif c.backbone == "grid":
        width = max(1, int(np.ceil(np.sqrt(n))))
        pairs = {(i, i + 1) for i in range(n - 1) if (i + 1) % width}
        pairs |= {(i, i + width) for i in range(n - width)}
    elif c.backbone == "random":
        order = rng.permutation(n)
        pairs = {
            tuple(sorted((int(order[i]), int(order[rng.integers(i)]))))
            for i in range(1, n)
        }
    elif c.backbone == "complete":
        pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}
    available = sorted({(i, j) for i in range(n) for j in range(i + 1, n)} - pairs)
    for i in rng.choice(
        len(available), size=min(len(available), c.backbone_shortcuts), replace=False
    ):
        pairs.add(available[i])
    for a, b in sorted(pairs):
        link(units[a], units[b], c.backbone_link_prob)
    nonhubs = np.flatnonzero(~is_hub)
    for _ in range(int(c.bridge_rate * c.n_nodes)):
        a = int(rng.choice(nonhubs))
        candidates = nonhubs[cluster[nonhubs] != cluster[a]]
        if not len(candidates):
            raise ValueError("bridge_rate needs non-hubs in multiple clusters")
        g.add_edge(a, int(rng.choice(candidates)))
    adj = csr_matrix(
        nx.to_scipy_sparse_array(g, nodelist=range(c.n_nodes), dtype=np.float64)
    )
    assert nx.is_connected(g) and offset == c.n_nodes
    return Graph(adj, weights, cluster, level, is_hub, c)
