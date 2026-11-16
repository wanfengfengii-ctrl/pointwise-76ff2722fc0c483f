"""Independent exhaustive enumeration of optimal trajectory explanations.

This reference is deliberately self-contained: it imports none of the
production routing summary or ambiguity counting machinery. It builds the
split-road / candidate-layer state graph directly, computes the optimum with
its own Bellman-Ford relaxation, and then enumerates every event sequence at
the optimum under an explicit cost bound. Explanations are keyed by exactly
the events of AMBIGUITY.md (road entries, the optional initial interior
occupation, and layer-tagged candidate acceptances), so duplicate geometric
splits collapse while distinct acceptance timings stay separate.
"""
from fractions import Fraction


def _location(pos):
    if pos.node is not None:
        return ("node", pos.node)
    return ("road", pos.road, pos.offset)


def _split_graph(graph, observations):
    """Arcs over nodes and candidate-interior cut points, with entry flags."""
    cuts = {rid: {Fraction(0), road.length}
            for rid, road in graph.roads.items()}
    for observation in observations:
        for candidate in observation.candidates:
            if candidate.position.road is not None:
                cuts[candidate.position.road].add(candidate.position.offset)
    arcs = {("node", node): [] for node in graph.nodes}

    def point(road, offset):
        if offset == 0:
            return ("node", road.tail)
        if offset == road.length:
            return ("node", road.head)
        return ("road", road.id, offset)

    for rid, road in graph.roads.items():
        points = sorted(cuts[rid])
        for lo, hi in zip(points, points[1:]):
            arcs.setdefault(point(road, lo), []).append(
                (point(road, hi), rid, hi - lo, lo == 0))
            arcs.setdefault(point(road, hi), [])
    return arcs


def _legal(graph, history, rid):
    trial = history + (rid,)
    if any(len(trial) >= len(pattern) and trial[-len(pattern):] == pattern
           for pattern in graph.forbidden):
        return None
    return trial[-2:]


def optimal_cost(graph, observations):
    """Own Bellman-Ford optimum over the finite layer/position/history graph."""
    if any(not observation.candidates for observation in observations):
        return None
    arcs = _split_graph(graph, observations)
    initial = {}
    for candidate in observations[0].candidates:
        history = (() if candidate.position.node is not None
                   else (candidate.position.road,))
        state = (1, _location(candidate.position), history)
        initial[state] = min(initial.get(state, candidate.penalty),
                             candidate.penalty)

    def successors(state):
        layer, where, history = state
        if layer == len(observations):
            return ()
        moves = []
        for candidate in observations[layer].candidates:
            if _location(candidate.position) == where:
                moves.append(((layer + 1, where, history), candidate.penalty))
        for there, rid, length, entering in arcs[where]:
            if not entering:
                moves.append(((layer, there, history), length))
                continue
            next_history = _legal(graph, history, rid)
            if next_history is not None:
                moves.append(((layer, there, next_history), length))
        return tuple(moves)

    states, seen, edges = list(initial), set(initial), {}
    for state in states:
        edges[state] = successors(state)
        for following, _ in edges[state]:
            if following not in seen:
                seen.add(following)
                states.append(following)
    distances = dict(initial)
    for _ in range(len(states) + 1):
        changed = False
        for state in states:
            if state not in distances:
                continue
            for following, cost in edges[state]:
                value = distances[state] + cost
                if following not in distances or value < distances[following]:
                    distances[following] = value
                    changed = True
        if not changed:
            break
    else:
        raise AssertionError("enumerator Bellman-Ford did not settle")
    values = [distance for (layer, _, _), distance in distances.items()
              if layer == len(observations)]
    return min(values, default=None)


def enumerate_report(graph, observations):
    """Exhaustively produce the same report shape as ``trajectory.ambiguity``."""
    optimum = optimal_cost(graph, observations)
    if optimum is None:
        return None
    arcs = _split_graph(graph, observations)
    depth = len(observations)
    findings = {}

    def record(events, road_counts, support, spent):
        if spent != optimum:
            return
        key = tuple(events)
        if key in findings:
            # Same complete event sequence: identical explanation by definition.
            return
        findings[key] = (dict(road_counts), tuple(dict(layer) for layer in support))

    def walk(layer, where, history, spent, events, road_counts, support):
        if spent > optimum:
            return
        if layer == depth:
            record(events, road_counts, support, spent)
            return
        # Acceptance moves first: a candidate at the current location is taken
        # now, tagging the acceptance with its observation layer.
        for candidate in observations[layer].candidates:
            if _location(candidate.position) != where:
                continue
            next_support = tuple(dict(entries) for entries in support)
            next_support[layer][candidate.id] = (
                next_support[layer].get(candidate.id, 0) + 1)
            walk(layer + 1, where, history, spent + candidate.penalty,
                 events + (("accept", layer, candidate.id),),
                 road_counts, next_support)
        # Physical moves along the split graph.
        for there, rid, length, entering in arcs[where]:
            if not entering:
                walk(layer, there, history, spent + length, events,
                     road_counts, support)
                continue
            next_history = _legal(graph, history, rid)
            if next_history is None:
                continue
            next_counts = dict(road_counts)
            next_counts[rid] = next_counts.get(rid, 0) + 1
            walk(layer, there, next_history, spent + length,
                 events + (("enter", rid),), next_counts, support)

    for candidate in observations[0].candidates:
        interior = candidate.position.road is not None
        history = (candidate.position.road,) if interior else ()
        road_counts = {rid: 0 for rid in graph.roads}
        events = []
        if interior:
            # The initial occupation precedes the first acceptance.
            road_counts[candidate.position.road] = 1
            events.append(("initial", candidate.position.road))
        support = tuple({} for _ in range(depth))
        support[0][candidate.id] = 1
        walk(1, _location(candidate.position), history, candidate.penalty,
             tuple(events) + (("accept", 0, candidate.id),), road_counts, support)

    count = len(findings)
    candidates = tuple({candidate.id: 0 for candidate in observation.candidates}
                       for observation in observations)
    roads = {rid: 0 for rid in graph.roads}
    for _, (road_counts, support) in findings.items():
        for rid, value in road_counts.items():
            roads[rid] += value
        for layer, entries in enumerate(support):
            for cid, value in entries.items():
                candidates[layer][cid] += value
    return {"total": optimum, "count": count,
            "candidates": candidates, "roads": roads}
