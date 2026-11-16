"""Joint candidate selection over continuation-aware road connections."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import routing
from roadnet import Candidate, Graph, Observation, validate_observations
from routing import Span, origin, stitch, transitions


@dataclass(frozen=True)
class Analysis:
    selected: tuple[str, ...]
    distance: Fraction
    penalty: Fraction
    spans: tuple[Span, ...]

    @property
    def total(self):
        return self.distance + self.penalty

    @property
    def roads(self):
        return tuple(s.road for s in self.spans)


@dataclass(frozen=True)
class _Label:
    candidate: Candidate
    history: tuple[str, ...]
    result: Analysis


def analyze(graph: Graph, observations: tuple[Observation, ...]) -> Analysis | None:
    observations = validate_observations(graph, observations)
    labels = {}
    for candidate in observations[0].candidates:
        history, spans = origin(candidate.position)
        result = Analysis((candidate.id,), Fraction(0), candidate.penalty, spans)
        labels[(candidate.id, history)] = _Label(candidate, history, result)
    for observation in observations[1:]:
        following = {}
        for label in labels.values():
            for candidate in observation.candidates:
                for route in transitions(graph, label.candidate.position,
                                         candidate.position, label.history):
                    old = label.result
                    result = Analysis(
                        old.selected + (candidate.id,),
                        old.distance + route.distance,
                        old.penalty + candidate.penalty,
                        stitch(old.spans, route.spans, label.candidate.position))
                    key = (candidate.id, route.history)
                    if key not in following or result.total < following[key].result.total:
                        following[key] = _Label(candidate, route.history, result)
        labels = following
    return min((label.result for label in labels.values()),
               key=lambda result: result.total, default=None)


class _Stat:
    """Compact aggregate for one DP state: cost plus exact occurrence counts.

    ``ways`` is the number of optimal prefix explanations reaching the state;
    ``support`` gives, per observation layer, the number of those explanations
    that accept each candidate id; ``initial`` is the road occupied at the
    first interior candidate (if any), counted once per explanation;
    ``connection_roads`` accumulates road-entry events produced by actual
    movement connections. Explanations themselves are never enumerated.
    """

    __slots__ = ("cost", "ways", "support", "initial", "connection_roads")

    def __init__(self, cost, ways, support, initial, connection_roads):
        self.cost = cost
        self.ways = ways
        self.support = support
        self.initial = initial
        self.connection_roads = connection_roads


def ambiguity(graph: Graph, observations: tuple[Observation, ...]) -> dict | None:
    """Exact ambiguity statistics for the global optimum.

    Returns None when no legal explanation exists, otherwise a dict with
    ``total`` (Fraction), ``count`` (int), ``candidates`` (one candidate-id
    support dict per observation layer) and ``roads`` (cumulative road-entry
    counts for every graph road). Explanations are distinguished by their
    road-entry events and layer-tagged candidate acceptance events, as defined
    in AMBIGUITY.md; statistics come from finite-state dynamic programming over
    ``(candidate, exit history)`` labels consuming compact routing summaries,
    never from multiplying independent per-segment optima or enumerating
    explanations.
    """
    observations = validate_observations(graph, observations)
    layer_count = len(observations)
    empty_roads = {rid: 0 for rid in graph.roads}

    def merge(bucket, key, cost, ways, support, initial, connection_roads):
        """Insert or combine one DP aggregate at ``key`` (dominance prune)."""
        current = bucket.get(key)
        if current is None or cost < current.cost:
            bucket[key] = _Stat(cost, ways, support, initial, connection_roads)
            return
        if cost > current.cost:
            return
        # Equal-cost merge. Equal aggregates reaching the same key describe the
        # same (candidate, exit history) state, hence the same initial road.
        if current.initial != initial:
            raise AssertionError("merged states disagree on initial road")
        current.ways += ways
        # Copy before mutating so aggregates shared with other states or with
        # previously returned reports are never modified through an alias.
        current.support = tuple(
            dict(layer_entries) for layer_entries in current.support)
        for layer, entries in enumerate(support):
            target = current.support[layer]
            for cid, value in entries.items():
                target[cid] = target.get(cid, 0) + value
        current.connection_roads = dict(current.connection_roads)
        for rid, value in connection_roads.items():
            current.connection_roads[rid] = (
                current.connection_roads.get(rid, 0) + value)

    # Layer 0: choosing and immediately accepting a first candidate. An
    # interior candidate contributes one initial occupation entry event, once.
    # The initial occupation is tracked separately from connection entries:
    # it belongs to the explanation itself and is therefore counted once per
    # final explanation, never multiplied by the number of later connections.
    labels = {}
    first = observations[0]
    for candidate in first.candidates:
        history, prefix = origin(candidate.position)
        support = tuple({} for _ in range(layer_count))
        support[0][candidate.id] = 1
        initial = candidate.position.road if prefix else None
        merge(labels, (candidate.id, history), candidate.penalty,
              1, support, initial, dict(empty_roads))

    for layer in range(1, layer_count):
        following = {}
        observation = observations[layer]
        for (previous_id, history), stat in labels.items():
            source = next(c for c in observations[layer - 1].candidates
                          if c.id == previous_id).position
            for candidate in observation.candidates:
                # Routing hands over equal-cost connection detail split by
                # entry/exit history; the trajectory layer alone combines it
                # with candidate penalties and later continuation eligibility.
                for connection in routing.connection_summaries(
                        graph, source, candidate.position, history):
                    produced = stat.ways * connection.ways
                    cost = stat.cost + connection.distance + candidate.penalty
                    # Every produced explanation inherits each prior acceptance
                    # once, and accepts this layer's candidate exactly once.
                    support = tuple(
                        {cid: value * connection.ways
                         for cid, value in layer_entries.items()}
                        for layer_entries in stat.support)
                    support[layer][candidate.id] = produced
                    # Movement entries are summed over the connection.ways
                    # distinct entry sequences, each exercised by every prefix
                    # way. The initial occupation stays outside this total.
                    connection_roads = dict(stat.connection_roads)
                    for rid, value in connection.road_occurrences.items():
                        connection_roads[rid] = (
                            connection_roads.get(rid, 0) + value * stat.ways)
                    merge(following, (candidate.id, connection.history), cost,
                          produced, support, stat.initial, connection_roads)
        labels = following
        if not labels:
            return None

    if not labels:
        return None
    total = min(stat.cost for stat in labels.values())
    finals = [stat for stat in labels.values() if stat.cost == total]

    count = sum(stat.ways for stat in finals)
    support = tuple({candidate.id: 0 for candidate in observation.candidates}
                    for observation in observations)
    roads = dict(empty_roads)
    for stat in finals:
        for layer, entries in enumerate(stat.support):
            target = support[layer]
            for cid, value in entries.items():
                target[cid] = target.get(cid, 0) + value
        for rid, value in stat.connection_roads.items():
            roads[rid] = roads.get(rid, 0) + value
        # One initial occupation per explanation.
        if stat.initial is not None:
            roads[stat.initial] += stat.ways
    return {"total": total, "count": count,
            "candidates": support, "roads": roads}
