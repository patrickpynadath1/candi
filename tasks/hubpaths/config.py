"""All persisted HubPaths generation parameters."""

from dataclasses import dataclass, asdict, replace


@dataclass(frozen=True)
class HubPathsConfig:
    n_waypoints: int = 3
    n_nodes: int = 10000
    max_steps: int = 77
    gamma: float = 2.5
    mean_degree: float = 8
    min_degree: float = 2
    max_degree: float | None = None
    alpha: float = 0.0
    endpoint_alpha: float = 0.0
    n_clusters: int = 50
    cluster_size_dist: str = "equal"
    hubs_per_cluster: int = 2
    n_levels: int = 2
    branching: int = 5
    hubs_per_group: int = 2
    hub_link_prob: float = 1.0
    backbone_link_prob: float = 1.0
    backbone: str = "ring"
    backbone_shortcuts: int = 0
    bridge_rate: float = 0.0
    temperature: float = 1.0
    edge_cost: float = 1.0
    hub_cost: float = 0.0
    length_mode: str = "at_most"
    require_simple: bool = False
    min_dist: int = 4
    max_dist: int = 24
    n_hints: int = 0
    n_train: int = 2000000
    n_val: int = 5000
    n_test: int = 5000
    starts_per_goal: int = 256
    min_edge_count: int = 5
    holdout_cluster_pairs: int = 20
    seed: int = 0
    goal_batch_size: int = 4
    cache_goals: int = 4
    max_sampling_attempts: int = 100000

    @property
    def seq_len(self):
        return self.max_steps + self.n_waypoints + 2

    def validate(self):
        if not 2 <= self.n_waypoints <= self.n_nodes:
            raise ValueError("n_waypoints must be between 2 and n_nodes")
        if not 1 <= self.min_dist <= self.max_dist <= self.max_steps:
            raise ValueError("require 1 <= min_dist <= max_dist <= max_steps")
        if (
            self.temperature <= 0
            or min(self.edge_cost, self.edge_cost + self.hub_cost) <= 0
        ):
            raise ValueError("temperature and all step costs must be positive")
        if self.gamma <= 1 or self.min_degree <= 0 or self.mean_degree <= 0:
            raise ValueError("gamma must exceed 1 and degrees must be positive")
        if self.cluster_size_dist not in {
            "equal",
            "powerlaw",
        } or self.length_mode not in {"exact", "at_most"}:
            raise ValueError("unknown size distribution or length mode")
        if self.backbone not in {"ring", "chain", "tree", "grid", "random", "complete"}:
            raise ValueError("unknown backbone")
        if (
            min(
                self.n_clusters,
                self.hubs_per_cluster,
                self.hubs_per_group,
                self.starts_per_goal,
                self.goal_batch_size,
                self.cache_goals,
            )
            < 1
            or self.branching < 2
        ):
            raise ValueError("counts must be positive; branching >= 2")
        if (
            min(
                self.n_levels,
                self.n_train,
                self.n_val,
                self.n_test,
                self.n_hints,
                self.min_edge_count,
                self.bridge_rate,
                self.backbone_shortcuts,
            )
            < 0
        ):
            raise ValueError("counts must be nonnegative")
        if (
            not 0
            <= self.holdout_cluster_pairs
            <= self.n_clusters * (self.n_clusters - 1) // 2
        ):
            raise ValueError("too many held-out cluster pairs")
        if not 0 <= self.hub_link_prob <= 1 or not 0 <= self.backbone_link_prob <= 1:
            raise ValueError("link probabilities must lie in [0, 1]")
        return self

    def to_dict(self):
        return asdict(self)


def preset(name):
    c = HubPathsConfig()
    if name == "tiny_debug":
        return replace(
            c,
            n_nodes=200,
            n_clusters=8,
            n_levels=1,
            max_steps=16,
            max_dist=10,
            min_dist=2,
            n_train=2000,
            n_val=100,
            n_test=100,
            holdout_cluster_pairs=2,
            min_edge_count=1,
        )
    if name == "language_like":
        return replace(c, gamma=2.1, n_levels=3)
    if name == "sudoku_like":
        return replace(preset("language_like"), temperature=0.05, n_hints=8)
    if name == "flat_control":
        return replace(c, gamma=6.0, n_levels=0, backbone="complete")
    if name == "default":
        return c
    raise ValueError(f"unknown preset {name}")
