"""HubPaths: hidden graph navigation with exactly scored first-hit walks."""

from .config import HubPathsConfig, preset
from .graph import Graph, generate_graph
from .rsp import RSP
from .verify import Reason, VerifyResult, verify_path, verify_batch, is_correct
from .data import HubPathsDataset, HubPathsTokenizer, build_dataset

__all__ = [
    "HubPathsConfig",
    "preset",
    "Graph",
    "generate_graph",
    "RSP",
    "Reason",
    "VerifyResult",
    "verify_path",
    "verify_batch",
    "is_correct",
    "HubPathsDataset",
    "HubPathsTokenizer",
    "build_dataset",
    "RouteDistribution",
    "verify_route",
    "verify_route_batch",
]

from .routes import RouteDistribution
from .verify import verify_route, verify_route_batch
