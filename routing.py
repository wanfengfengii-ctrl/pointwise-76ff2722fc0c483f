"""Road routing with exact distance and finite continuation histories."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from heapq import heappop, heappush
from itertools import count

from roadnet import Graph, Position


@dataclass(frozen=True)
class Span:
    road: str
    start: Fraction
    end: Fraction


@dataclass(frozen=True)
class Route:
    distance: Fraction
    spans: tuple[Span, ...]
    history: tuple[str, ...]

    @property
    def roads(self):
        return tuple(s.road for s in self.spans)


def enter(graph, history, rid):
    trial = history + (rid,)
    if any(len(trial) >= len(p) and trial[-len(p):] == p for p in graph.forbidden):
        return None
    return trial[-2:]


def origin(position):
    """An interior starting observation already occupies its road."""
    if position.node is not None:
        return (), ()
    return (position.road,), (Span(position.road, position.offset, position.offset),)


def stitch(prefix, suffix, boundary):
    """Only an interior continuation joins two pieces of one road occurrence."""
    if not suffix:
        return prefix
    if boundary.road is not None and prefix:
        a, b = prefix[-1], suffix[0]
        if a.road != boundary.road or b.road != a.road or a.end != b.start:
            raise ValueError("broken interior continuation")
        return prefix[:-1] + (Span(a.road, a.start, b.end),) + suffix[1:]
    return prefix + suffix


def transitions(graph, source, target, history):
    """Cheapest connection for EACH reachable exit history.

    source/target are canonical positions; history belongs to source.
    Returned spans exclude the source's pre-existing road occupancy.
    The search deliberately explores the whole finite state graph.
    """
    best, paths, exits = {}, {}, {}
    heap, serial = [], count()

    def offer(node, tail, distance, spans):
        state = (node, tail)
        if state not in best or distance < best[state]:
            best[state], paths[state] = distance, spans
            heappush(heap, (distance, next(serial), state))

    def finish(tail, distance, spans):
        if tail not in exits or distance < exits[tail].distance:
            exits[tail] = Route(distance, spans, tail)

    if source.node is not None:
        offer(source.node, history, Fraction(0), ())
    else:
        if not history or history[-1] != source.road:
            raise ValueError("interior position requires occupied-road history")
        road = graph.roads[source.road]
        if target.road == source.road and target.offset >= source.offset:
            spans = () if target.offset == source.offset else (
                Span(road.id, source.offset, target.offset),)
            finish(history, target.offset - source.offset, spans)
        offer(road.head, history, road.length - source.offset,
              (Span(road.id, source.offset, road.length),))

    while heap:
        distance, _, state = heappop(heap)
        if best[state] != distance:
            continue
        node, tail = state
        spans = paths[state]
        if target.node == node:
            finish(tail, distance, spans)
        for rid in graph.outgoing[node]:
            road = graph.roads[rid]
            next_tail = enter(graph, tail, rid)
            if next_tail is None:
                continue
            if target.road == rid:
                finish(next_tail, distance + target.offset,
                       spans + (Span(rid, Fraction(0), target.offset),))
            offer(road.head, next_tail, distance + road.length,
                  spans + (Span(rid, Fraction(0), road.length),))
    return tuple(exits.values())


def shortest_route(graph: Graph, source: Position, target: Position) -> Route | None:
    source, target = graph.check_position(source), graph.check_position(target)
    history, prefix = origin(source)
    choices = transitions(graph, source, target, history)
    if not choices:
        return None
    route = min(choices, key=lambda r: r.distance)
    return Route(route.distance, stitch(prefix, route.spans, source), route.history)
