"""Road routing with exact distance and finite continuation histories."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from heapq import heappop, heappush
from itertools import count
from types import MappingProxyType

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


@dataclass(frozen=True)
class Connection:
    """Compact summary of every cheapest connection with one exit history.

    A connection is distinguished by its ordered road-entry events; only the
    last two entered roads (``history``) remain relevant to later forbidden
    checks. ``ways`` counts distinct minimum-distance entry sequences reaching
    the target with this exact exit history, and ``road_occurrences`` sums each
    road's entry events across those ways (one entry event per road occurrence;
    interior continuation and stationary occupancy never re-enter, so they add
    nothing). No individual path is stored.
    """

    history: tuple[str, ...]
    distance: Fraction
    ways: int
    road_occurrences: MappingProxyType


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


def _add_occurrences(target, source, multiplier=1):
    """Accumulate road-entry counts in place."""
    for rid, value in source.items():
        target[rid] = target.get(rid, 0) + value * multiplier


def connection_summaries(graph: Graph, source: Position, target: Position,
                         history: tuple[str, ...]) -> tuple[Connection, ...]:
    """Cheapest-connection summaries split by source entry and exit history.

    Mirrors the finite state graph of ``transitions`` but keeps, per
    ``(node, tail)`` search state, only the minimum distance together with the
    number of equal-cost entry sequences reaching it and their summed road
    occurrences. Positive road lengths make every graph edge strictly costly,
    so all shortest paths to a state are finalized before it is expanded and
    equal-cost arrivals can be merged without revisiting the heap. The zero
    distance case is limited to the single stationary interior continuation.

    The returned tuple holds one summary per reachable exit history at the
    minimum distance for that history. Summaries never include the source's
    pre-existing road occupancy, so the trajectory layer charges an initial
    interior occupation exactly once for the whole explanation.
    """
    best, ways, occurrences = {}, {}, {}
    heap, serial = [], count()
    exits = {}
    settled = set()

    def offer(node, tail, distance, added_road=None, parent=None):
        """Relax one search state, merging equal-cost predecessors.

        A parent aggregate already sums occurrences over all of that parent's
        shortest ways, so it is adopted once. Equal-cost predecessors describe
        disjoint path sets, so merging adds their aggregates directly.
        """
        state = (node, tail)
        if state in best and distance > best[state]:
            return
        if state not in best or distance < best[state]:
            # First (or strictly better) arrival: a brand-new aggregate.
            best[state] = distance
            heappush(heap, (distance, next(serial), state))
            ways[state] = 0
            accumulated = {}
        else:
            # Equal-cost arrival. Copy the existing aggregate first; mutating
            # it in place would also alter states/records sharing this dict.
            accumulated = dict(occurrences[state])
        count_in = 1 if parent is None else ways[parent]
        ways[state] += count_in
        if parent is not None:
            # Adopt the parent aggregate once; it already totals parent ways.
            _add_occurrences(accumulated, occurrences[parent])
        if added_road is not None:
            accumulated[added_road] = accumulated.get(added_road, 0) + count_in
        occurrences[state] = accumulated

    def finish(tail, distance, parent=None, added_road=None):
        """Record one cheapest way of reaching the target under ``tail``.

        With a parent, the parent aggregate already holds the complete
        occurrence counts across its ways and is adopted once, never scaled.
        Only ``added_road`` (entering the target road and stopping inside it)
        contributes a new entry event, once per arriving way.
        """
        if tail in exits and distance > exits[tail][0]:
            return
        count_in = 1 if parent is None else ways[parent]
        base = {} if parent is None else dict(occurrences[parent])
        if added_road is not None:
            base[added_road] = base.get(added_road, 0) + count_in
        if tail not in exits or distance < exits[tail][0]:
            exits[tail] = [distance, count_in, base]
        else:
            record = exits[tail]
            record[1] += count_in
            # Independent dict so later equal-cost merges cannot mutate an
            # aggregate shared with a search state.
            accumulated = dict(record[2])
            for rid, value in base.items():
                accumulated[rid] = accumulated.get(rid, 0) + value
            record[2] = accumulated

    if source.node is not None:
        offer(source.node, history, Fraction(0))
    else:
        if not history or history[-1] != source.road:
            raise ValueError("interior position requires occupied-road history")
        road = graph.roads[source.road]
        # Staying inside the already occupied road never creates an entry,
        # whether stationary or a forward interior continuation.
        if target.road == source.road and target.offset >= source.offset:
            exits[history] = [target.offset - source.offset, 1, {}]
        # Reaching the head of the occupied road completes that occurrence:
        # the history is unchanged and no entry event is produced.
        offer(road.head, history, road.length - source.offset)

    while heap:
        distance, _, state = heappop(heap)
        if best.get(state) != distance or state in settled:
            continue
        # Strictly positive edge costs guarantee every shortest predecessor has
        # been merged by the time the state is popped; expand it exactly once.
        settled.add(state)
        node, tail = state
        if target.node == node:
            finish(tail, distance, parent=state)
        for rid in graph.outgoing[node]:
            road = graph.roads[rid]
            next_tail = enter(graph, tail, rid)
            if next_tail is None:
                continue
            if target.road == rid:
                # Entering rid and stopping at an interior offset of it.
                finish(next_tail, distance + target.offset,
                       parent=state, added_road=rid)
            offer(road.head, next_tail, distance + road.length,
                  added_road=rid, parent=state)
    return tuple(
        Connection(history=tail, distance=record[0], ways=record[1],
                   road_occurrences=MappingProxyType(record[2]))
        for tail, record in exits.items())
