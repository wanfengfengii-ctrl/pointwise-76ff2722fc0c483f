"""Small-input independent reference and output replay; no production solver calls."""
from fractions import Fraction


def location(pos):
    return ("node", pos.node) if pos.node is not None else ("road", pos.road, pos.offset)


def optimum(graph, observations):
    """Explicit split-road/layer graph, exhaustive reachability, Bellman-Ford."""
    if any(not o.candidates for o in observations):
        return None
    cuts = {rid: {Fraction(0), road.length} for rid, road in graph.roads.items()}
    for observation in observations:
        for c in observation.candidates:
            if c.position.road is not None:
                cuts[c.position.road].add(c.position.offset)
    arcs = {("node", node): [] for node in graph.nodes}
    for rid, road in graph.roads.items():
        points = sorted(cuts[rid])
        def point(offset):
            if offset == 0:
                return ("node", road.tail)
            if offset == road.length:
                return ("node", road.head)
            return ("road", rid, offset)
        for lo, hi in zip(points, points[1:]):
            arcs.setdefault(point(lo), []).append((point(hi), rid, hi - lo, lo == 0))
            arcs.setdefault(point(hi), [])

    initial = {}
    for c in observations[0].candidates:
        history = () if c.position.node is not None else (c.position.road,)
        state = (1, location(c.position), history)
        initial[state] = min(initial.get(state, c.penalty), c.penalty)

    def successors(state):
        layer, where, history = state
        if layer == len(observations):
            return []
        result = []
        for c in observations[layer].candidates:
            if location(c.position) == where:
                result.append(((layer + 1, where, history), c.penalty))
        for there, rid, length, entering in arcs[where]:
            full = history + (rid,) if entering else history
            illegal = any(len(full) >= len(p) and full[-len(p):] == p
                          for p in graph.forbidden)
            if not illegal:
                result.append(((layer, there, full[-2:]), length))
        return result

    states, seen, edges = list(initial), set(initial), {}
    for state in states:
        edges[state] = successors(state)
        for following, _ in edges[state]:
            if following not in seen:
                seen.add(following)
                states.append(following)
    distances = dict(initial)
    for _ in range(len(states)):
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
        raise AssertionError("reference failed to settle finite nonnegative graph")
    values = [d for (layer, _, _), d in distances.items() if layer == len(observations)]
    return min(values, default=None)


def replay(graph, source, target, spans, distance, roads):
    """Check physical continuity, entry events, distance, and all visited positions."""
    assert tuple(s.road for s in spans) == tuple(roads)
    for pattern in graph.forbidden:
        assert all(tuple(roads[i:i + len(pattern)]) != pattern
                   for i in range(len(roads) - len(pattern) + 1))
    visits = {location(source): [Fraction(0)]}
    traveled = Fraction(0)
    previous = None
    for i, span in enumerate(spans):
        edge = graph.roads[span.road]
        assert 0 <= span.start <= span.end <= edge.length
        assert span.start < span.end or (
            i == 0 and source.road == span.road and source.offset == span.start)
        if previous is None:
            if source.node is not None:
                assert edge.tail == source.node and span.start == 0
            else:
                assert edge.id == source.road and span.start == source.offset
        else:
            old = graph.roads[previous.road]
            assert previous.end == old.length and span.start == 0
            assert old.head == edge.tail
        if span.start == 0:
            visits.setdefault(("node", edge.tail), []).append(traveled)
        if span.end == edge.length:
            visits.setdefault(("node", edge.head), []).append(traveled + span.end - span.start)
        previous = span
        traveled += span.end - span.start
    assert traveled == distance
    if not spans:
        assert source == target
    else:
        last = spans[-1]
        if target.node is not None:
            edge = graph.roads[last.road]
            assert last.end == edge.length and edge.head == target.node
        else:
            assert last.road == target.road and last.end == target.offset
    return visits


def check_analysis(graph, observations, result):
    assert len(result.selected) == len(observations)
    chosen = [next(c for c in o.candidates if c.id == cid)
              for o, cid in zip(observations, result.selected)]
    assert result.penalty == sum((c.penalty for c in chosen), Fraction(0))
    assert result.total == result.distance + result.penalty
    visits = replay(graph, chosen[0].position, chosen[-1].position,
                    result.spans, result.distance, result.roads)
    cursor = Fraction(0)
    for candidate in chosen:
        pos = candidate.position
        possibilities = list(visits.get(location(pos), []))
        traveled = Fraction(0)
        if pos.road is not None:
            for span in result.spans:
                if span.road == pos.road and span.start <= pos.offset <= span.end:
                    possibilities.append(traveled + pos.offset - span.start)
                traveled += span.end - span.start
        future = [value for value in possibilities if value >= cursor]
        assert future, ("candidate not visited in observation order", candidate)
        cursor = min(future)
