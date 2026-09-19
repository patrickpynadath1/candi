"""Independent verifier: no sampler or partition-function code is used here."""

from dataclasses import dataclass
from enum import IntEnum
import numpy as np
import torch
from .graph import PAD, SEP, N_SPECIAL


class Reason(IntEnum):
    OK = 0
    MALFORMED = 1
    WRONG_START = 2
    WRONG_END = 3
    TOO_LONG = 4
    NOT_AN_EDGE = 5
    GOAL_BEFORE_END = 6
    REPEATED_NODE = 7
    HINT_MISMATCH = 8
    PROMPT_ALTERED = 9
    WAYPOINT_MISSING = 10
    WAYPOINT_ORDER = 11


@dataclass(frozen=True)
class VerifyResult:
    valid: bool
    reason: Reason
    n_steps: int = 0
    cost: float = 0.0
    is_shortest: bool = False
    stretch: float = float("nan")
    n_hubs_visited: int = 0


def verify_path(
    graph, start, goal, path, *, max_steps, require_simple=False, hints=None
):
    tokens = list(path)

    def fail(reason):
        return VerifyResult(False, reason)

    if any(
        not isinstance(t, (int, np.integer))
        and not (
            isinstance(t, torch.Tensor)
            and t.ndim == 0
            and not t.dtype.is_floating_point
        )
        for t in tokens
    ):
        return fail(Reason.MALFORMED)
    tokens = [int(t) for t in tokens]
    while tokens and tokens[-1] == PAD:
        tokens.pop()
    if not tokens or any(
        t < N_SPECIAL or t >= graph.n_nodes + N_SPECIAL for t in tokens
    ):
        return fail(Reason.MALFORMED)
    nodes = [t - N_SPECIAL for t in tokens]
    if nodes[0] != start:
        return fail(Reason.WRONG_START)
    if nodes[-1] != goal:
        return fail(Reason.WRONG_END)
    steps = len(nodes) - 1
    if steps > max_steps:
        return fail(Reason.TOO_LONG)
    for a, b in zip(nodes, nodes[1:]):
        neighbors = graph.adjacency.indices[
            graph.adjacency.indptr[a] : graph.adjacency.indptr[a + 1]
        ]
        if b not in neighbors:
            return fail(Reason.NOT_AN_EDGE)
    if goal in nodes[:-1]:
        return fail(Reason.GOAL_BEFORE_END)
    if require_simple and len(set(nodes)) != len(nodes):
        return fail(Reason.REPEATED_NODE)
    if any(
        pos < 0 or pos >= len(nodes) or nodes[pos] != node
        for pos, node in (hints or {}).items()
    ):
        return fail(Reason.HINT_MISMATCH)
    distance = graph.distances(goal)[start]
    cost = (
        steps * graph.config.edge_cost
        + graph.config.hub_cost * graph.is_hub[nodes[1:]].sum()
    )
    return VerifyResult(
        True,
        Reason.OK,
        steps,
        float(cost),
        steps == distance,
        float(steps / distance) if distance else 1.0,
        int(graph.is_hub[nodes].sum()),
    )


def is_correct(graph, start, goal, path, **kwargs):
    return verify_path(graph, start, goal, path, **kwargs).valid


def verify_batch(
    graph, sequences, prompts=None, *, max_steps=None, require_simple=False, hints=None
):
    """Vectorized token/edge checks. Prompts contain encoded node TOKEN ids."""
    x = torch.as_tensor(sequences)
    if x.ndim != 2 or x.shape[1] < 4 or x.dtype.is_floating_point:
        raise ValueError("sequences must be an integer [B, >=4] tensor")
    path = x[:, 3:]
    b, width = path.shape
    device = x.device
    reasons = torch.zeros(b, dtype=torch.long, device=device)

    def mark(mask, reason):
        reasons[(reasons == 0) & mask] = int(reason)

    lengths = (path != PAD).sum(1)
    positions = torch.arange(width, device=device)[None, :]
    active = positions < lengths[:, None]
    malformed = (
        (active & ((path < N_SPECIAL) | (path >= graph.n_nodes + N_SPECIAL)))
        | (~active & (path != PAD))
    ).any(1)
    malformed |= (
        (lengths == 0)
        | (x[:, 2] != SEP)
        | ((x[:, :2] < N_SPECIAL) | (x[:, :2] >= graph.n_nodes + N_SPECIAL)).any(1)
    )
    if prompts is not None:
        mark(
            (x[:, :2] != torch.as_tensor(prompts, device=device)).any(1),
            Reason.PROMPT_ALTERED,
        )
    mark(malformed, Reason.MALFORMED)
    mark(path[:, 0] != x[:, 0], Reason.WRONG_START)
    last = path.gather(1, (lengths - 1).clamp_min(0)[:, None])[:, 0]
    mark(last != x[:, 1], Reason.WRONG_END)
    mark(
        lengths - 1 > (max_steps if max_steps is not None else graph.config.max_steps),
        Reason.TOO_LONG,
    )
    edges = torch.as_tensor(graph.edge_keys, device=device)
    keys = (path[:, :-1] - N_SPECIAL) * graph.n_nodes + path[:, 1:] - N_SPECIAL
    idx = torch.searchsorted(edges, keys.contiguous())
    matches = (idx < len(edges)) & (edges[idx.clamp_max(len(edges) - 1)] == keys)
    mark(
        (~matches & (positions[:, :-1] < lengths[:, None] - 1)).any(1),
        Reason.NOT_AN_EDGE,
    )
    mark(
        ((path == x[:, 1:2]) & (positions < lengths[:, None] - 1)).any(1),
        Reason.GOAL_BEFORE_END,
    )
    if require_simple:
        ordered = (
            path.masked_fill(~active, graph.n_nodes + N_SPECIAL).sort(dim=1).values
        )
        duplicates = (ordered[:, :-1] == ordered[:, 1:]) & (
            ordered[:, :-1] < graph.n_nodes + N_SPECIAL
        )
        mark(duplicates.any(1), Reason.REPEATED_NODE)
    if hints:
        per_row = [hints] * b if isinstance(hints, dict) else hints
        mismatches = [
            any(
                p < 0 or p >= int(lengths[i]) or int(path[i, p]) != node + N_SPECIAL
                for p, node in h.items()
            )
            for i, h in enumerate(per_row)
        ]
        mark(torch.tensor(mismatches, device=device), Reason.HINT_MISMATCH)
    return reasons == 0, reasons


def verify_route(
    graph, waypoints, path, *, max_steps, require_simple=False, hints=None
):
    """Waypoints and hint values are node ids; path entries are token ids."""
    waypoints = tuple(int(v) for v in waypoints)
    if (
        len(waypoints) < 2
        or len(set(waypoints)) != len(waypoints)
        or any(v < 0 or v >= graph.n_nodes for v in waypoints)
    ):
        return VerifyResult(False, Reason.MALFORMED)
    result = verify_path(
        graph,
        waypoints[0],
        waypoints[-1],
        path,
        max_steps=max_steps,
        require_simple=require_simple,
        hints=hints,
    )
    if not result.valid:
        return result
    nodes = [int(t) - N_SPECIAL for t in path if int(t) != PAD]
    if any(v not in nodes for v in waypoints):
        return VerifyResult(False, Reason.WAYPOINT_MISSING)
    first_visits = [nodes.index(v) for v in waypoints]
    if first_visits != sorted(first_visits):
        return VerifyResult(False, Reason.WAYPOINT_ORDER)
    # Independent BFS over progress states; do not use sampler's transition code.
    from collections import deque

    queue = deque([(waypoints[0], 1, 0)])
    seen = {(waypoints[0], 1)}
    shortest = None
    while queue:
        node, stage, distance = queue.popleft()
        if stage == len(waypoints):
            shortest = distance
            break
        for dest in graph.adjacency.indices[
            graph.adjacency.indptr[node] : graph.adjacency.indptr[node + 1]
        ]:
            if dest in waypoints[stage + 1 :]:
                continue
            next_stage = stage + int(dest == waypoints[stage])
            state = int(dest), next_stage
            if state not in seen:
                seen.add(state)
                queue.append((*state, distance + 1))
    return VerifyResult(
        True,
        Reason.OK,
        result.n_steps,
        result.cost,
        result.n_steps == shortest,
        result.n_steps / shortest,
        result.n_hubs_visited,
    )


def verify_route_batch(
    graph,
    sequences,
    prompts=None,
    *,
    n_waypoints=None,
    max_steps=None,
    require_simple=False,
    hints=None,
):
    x = torch.as_tensor(sequences)
    count = n_waypoints or graph.config.n_waypoints
    if x.ndim != 2 or x.shape[1] < count + 2:
        raise ValueError("invalid waypoint sequence shape")
    # Reuse vectorized edge validation on a two-endpoint projection.
    projected = torch.cat((x[:, :1], x[:, count - 1 : count], x[:, count:]), dim=1)
    valid, reasons = verify_batch(
        graph,
        projected,
        max_steps=max_steps,
        require_simple=require_simple,
        hints=hints,
    )
    malformed = (
        (x[:, :count] < N_SPECIAL) | (x[:, :count] >= graph.n_nodes + N_SPECIAL)
    ).any(1)
    sorted_prompt = x[:, :count].sort(1).values
    malformed |= (sorted_prompt[:, 1:] == sorted_prompt[:, :-1]).any(1)
    reasons[malformed] = int(Reason.MALFORMED)
    path = x[:, count + 1 :]
    positions = torch.arange(path.shape[1], device=x.device)
    visits = torch.where(
        path[:, None, :] == x[:, :count, None], positions, path.shape[1]
    ).amin(2)
    missing = (visits == path.shape[1]).any(1)
    reasons[(reasons == 0) & missing] = int(Reason.WAYPOINT_MISSING)
    wrong_order = (visits[:, 1:] <= visits[:, :-1]).any(1)
    reasons[(reasons == 0) & wrong_order] = int(Reason.WAYPOINT_ORDER)
    if prompts is not None:
        altered = (x[:, :count] != torch.as_tensor(prompts, device=x.device)).any(1)
        reasons[altered] = int(Reason.PROMPT_ALTERED)
    return reasons == 0, reasons
