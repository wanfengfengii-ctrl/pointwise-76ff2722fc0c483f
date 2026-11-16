"""Joint candidate selection over continuation-aware road connections."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

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
