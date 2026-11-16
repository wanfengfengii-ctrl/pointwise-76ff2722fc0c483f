"""Exact finite directed road data and observation input."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (int, str, Fraction)):
        raise ValueError("expected integer or rational string")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid rational") from exc


def name(value):
    if not isinstance(value, str) or not value:
        raise ValueError("expected nonempty identifier")
    return value


def fields(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError("expected object")
    if not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError("invalid object fields")


def sequence(value):
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected finite array")
    return value


@dataclass(frozen=True)
class Road:
    id: str
    tail: str
    head: str
    length: Fraction


@dataclass(frozen=True)
class Position:
    node: str | None = None
    road: str | None = None
    offset: Fraction = Fraction(0)


@dataclass(frozen=True)
class Candidate:
    id: str
    position: Position
    penalty: Fraction = Fraction(0)


@dataclass(frozen=True)
class Observation:
    time: Fraction
    candidates: tuple[Candidate, ...]


class Graph:
    """Validated graph with read-only road and adjacency mappings."""
    def __init__(self, data):
        fields(data, ("nodes", "roads"), ("forbidden",))
        nodes = tuple(name(n) for n in sequence(data["nodes"]))
        if len(set(nodes)) != len(nodes):
            raise ValueError("duplicate node")
        self.nodes = frozenset(nodes)
        roads = {}
        outgoing = {n: [] for n in nodes}
        for row in sequence(data["roads"]):
            fields(row, ("id", "tail", "head", "length"))
            rid, tail, head = (name(row[k]) for k in ("id", "tail", "head"))
            length = rational(row["length"])
            if rid in roads or tail not in self.nodes or head not in self.nodes or length <= 0:
                raise ValueError("invalid road")
            roads[rid] = Road(rid, tail, head, length)
            outgoing[tail].append(rid)
        forbidden = set()
        for row in sequence(data.get("forbidden", [])):
            pattern = tuple(name(r) for r in sequence(row))
            if len(pattern) not in (2, 3) or any(r not in roads for r in pattern):
                raise ValueError("invalid forbidden sequence")
            if any(roads[a].head != roads[b].tail for a, b in zip(pattern, pattern[1:])):
                raise ValueError("disconnected forbidden sequence")
            forbidden.add(pattern)
        self.roads = MappingProxyType(roads)
        self.outgoing = MappingProxyType({n: tuple(sorted(rs)) for n, rs in outgoing.items()})
        self.forbidden = frozenset(forbidden)

    @classmethod
    def from_dict(cls, data):
        return cls(data)

    def position(self, data):
        if not isinstance(data, dict):
            raise ValueError("expected position object")
        if set(data) == {"node"}:
            node = name(data["node"])
            if node not in self.nodes:
                raise ValueError("unknown node")
            return Position(node=node)
        fields(data, ("road", "offset"))
        rid = name(data["road"])
        if rid not in self.roads:
            raise ValueError("unknown road")
        offset = rational(data["offset"])
        road = self.roads[rid]
        if not 0 <= offset <= road.length:
            raise ValueError("offset outside road")
        if offset == 0:
            return Position(node=road.tail)
        if offset == road.length:
            return Position(node=road.head)
        return Position(road=rid, offset=offset)

    def check_position(self, pos):
        if not isinstance(pos, Position):
            raise ValueError("expected Position")
        if pos.node is not None:
            if pos.road is not None or rational(pos.offset) != 0:
                raise ValueError("invalid node position")
            return self.position({"node": pos.node})
        return self.position({"road": pos.road, "offset": pos.offset})


def validate_observations(graph, observations):
    result = []
    previous = None
    for obs in sequence(observations):
        if not isinstance(obs, Observation):
            raise ValueError("expected Observation")
        time = rational(obs.time)
        if previous is not None and time < previous:
            raise ValueError("observation times must be sorted")
        previous = time
        candidates, seen = [], set()
        for candidate in sequence(obs.candidates):
            if not isinstance(candidate, Candidate):
                raise ValueError("expected Candidate")
            cid = name(candidate.id)
            penalty = rational(candidate.penalty)
            if cid in seen or penalty < 0:
                raise ValueError("duplicate candidate or negative penalty")
            seen.add(cid)
            candidates.append(Candidate(cid, graph.check_position(candidate.position), penalty))
        result.append(Observation(time, tuple(candidates)))
    if not result:
        raise ValueError("at least one observation required")
    return tuple(result)


def read_observations(graph, data):
    observations = []
    for row in sequence(data):
        fields(row, ("time", "candidates"))
        candidates = []
        for item in sequence(row["candidates"]):
            fields(item, ("id", "position", "penalty"))
            candidates.append(Candidate(name(item["id"]), graph.position(item["position"]),
                                        rational(item["penalty"])))
        observations.append(Observation(rational(row["time"]), tuple(candidates)))
    return validate_observations(graph, observations)
