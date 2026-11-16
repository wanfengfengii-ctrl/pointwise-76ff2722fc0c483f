"""Ambiguity statistics: fixed regression, independent enumeration, boundaries."""
import copy
import json
import unittest
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch

from roadnet import Candidate, Graph, Observation, Position, read_observations
import routing
import trajectory
from trajectory import ambiguity
from tests.enumeration import enumerate_report


def make_graph(rows, forbidden=(), extra=()):
    nodes = sorted(set(extra) | {n for _, a, b, _ in rows for n in (a, b)})
    return Graph.from_dict({
        "nodes": nodes,
        "roads": [{"id": r, "tail": a, "head": b, "length": str(w)}
                  for r, a, b, w in rows],
        "forbidden": [list(pattern) for pattern in forbidden],
    })


def cand(graph, cid, node=None, road=None, offset=0, penalty=0):
    raw = {"node": node} if node is not None else {"road": road, "offset": str(offset)}
    return Candidate(cid, graph.position(raw), F(penalty))


def obs(*groups):
    return tuple(Observation(F(i), tuple(group)) for i, group in enumerate(groups))


def assert_report_equal(test, report, expected_total, expected_count,
                        expected_candidates, expected_roads):
    test.assertEqual(report["total"], expected_total)
    test.assertEqual(report["count"], expected_count)
    test.assertEqual([dict(layer) for layer in report["candidates"]],
                     expected_candidates)
    test.assertEqual(dict(report["roads"]), expected_roads)


class FixedExampleRegression(unittest.TestCase):
    """Boundary regression against the shipped fixed expectations."""

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "ambiguity_examples.json"
        cls.examples = json.loads(path.read_text(encoding="utf-8"))

    def test_every_fixed_example_matches_expected_report(self):
        for example in self.examples:
            with self.subTest(example=example["name"]):
                graph = Graph.from_dict(example["graph"])
                observations = read_observations(graph, example["observations"])
                report = ambiguity(graph, observations)
                expected = example["expected"]
                if expected is None:
                    self.assertIsNone(report)
                    continue
                self.assertIsNotNone(report)
                self.assertEqual(str(report["total"]), expected["total"])
                self.assertIsInstance(report["total"], F)
                self.assertEqual(report["count"], expected["count"])
                self.assertEqual([dict(layer) for layer in report["candidates"]],
                                 expected["candidates"])
                self.assertEqual(dict(report["roads"]), expected["roads"])

    def test_all_candidates_and_roads_listed_even_when_unsupported(self):
        example = next(e for e in self.examples
                       if e["name"] == "candidate_penalty_filters_roads")
        graph = Graph.from_dict(example["graph"])
        observations = read_observations(graph, example["observations"])
        report = ambiguity(graph, observations)
        # Every input candidate id appears in its layer, zero if unsupported.
        for layer, observation in enumerate(observations):
            for candidate in observation.candidates:
                self.assertIn(candidate.id, report["candidates"][layer])
        # Every graph road appears, including the never-travelled ones.
        for rid in graph.roads:
            self.assertIn(rid, report["roads"])
        self.assertEqual(report["candidates"][1]["m"], 0)
        self.assertEqual(report["roads"]["sm"], 0)
        self.assertEqual(report["roads"]["mt"], 0)

    def test_returned_report_is_defensive_copy(self):
        example = next(e for e in self.examples
                       if e["name"] == "merged_equal_connections")
        graph = Graph.from_dict(example["graph"])
        observations = read_observations(graph, example["observations"])
        first = ambiguity(graph, observations)
        first["roads"]["a"] = 999
        first["candidates"][0]["s"] = 999
        second = ambiguity(graph, observations)
        self.assertEqual(second["roads"]["a"], 1)
        self.assertEqual(second["candidates"][0]["s"], 2)
        # Repeated calls do not mutate earlier reports through shared dicts.
        self.assertEqual(ambiguity(graph, observations)["count"], 2)

    def test_inputs_are_not_modified(self):
        example = next(e for e in self.examples
                       if e["name"] == "same_route_different_acceptance")
        graph = Graph.from_dict(example["graph"])
        observations = read_observations(graph, example["observations"])
        snapshot = copy.deepcopy(observations)
        ambiguity(graph, observations)
        self.assertEqual(observations, snapshot)


class EnumerationCrossCheck(unittest.TestCase):
    """The compact DP must equal an independent exhaustive event enumeration."""

    def cross_check(self, graph, observations):
        expected = enumerate_report(graph, observations)
        report = ambiguity(graph, observations)
        if expected is None:
            self.assertIsNone(report)
            return
        self.assertIsNotNone(report)
        self.assertEqual(report["total"], expected["total"])
        self.assertEqual(report["count"], expected["count"])
        self.assertEqual([dict(layer) for layer in report["candidates"]],
                         [dict(layer) for layer in expected["candidates"]])
        self.assertEqual(dict(report["roads"]), dict(expected["roads"]))
        # Counting invariants on every report.
        self.assertEqual(sum(report["candidates"][0].values()), report["count"])
        for layer_entries in report["candidates"]:
            self.assertEqual(sum(layer_entries.values()), report["count"])
        self.assertGreaterEqual(report["count"], 1)
        self.assertGreaterEqual(sum(report["roads"].values()), 0)
        return report

    def test_fixed_examples_against_independent_enumeration(self):
        path = Path(__file__).resolve().parents[1] / "ambiguity_examples.json"
        for example in json.loads(path.read_text(encoding="utf-8")):
            with self.subTest(example=example["name"]):
                graph = Graph.from_dict(example["graph"])
                observations = read_observations(graph, example["observations"])
                self.cross_check(graph, observations)

    def test_interior_forward_and_repeated_position_no_new_entry(self):
        graph = make_graph([("b", "U", "V", 3)])
        p = cand(graph, "p", road="b", offset=1)
        q = cand(graph, "q", road="b", offset=2)
        # Repeated same position, then forward inside the same occurrence.
        self.cross_check(graph, obs([p], [p], [q]))
        report = ambiguity(graph, obs([p], [p], [q]))
        assert_report_equal(self, report, F(1), 1,
                            [{"p": 1}, {"p": 1}, {"q": 1}], {"b": 1})

    def test_stationary_interior_trip_keeps_single_occurrence(self):
        graph = make_graph([("b", "U", "V", 3)])
        p = cand(graph, "p", road="b", offset=1)
        report = ambiguity(graph, obs([p], [p], [p]))
        assert_report_equal(self, report, F(0), 1,
                            [{"p": 1}, {"p": 1}, {"p": 1}], {"b": 1})
        self.cross_check(graph, obs([p], [p], [p]))

    def test_node_reentry_adds_occurrence_including_self_loop(self):
        graph = make_graph([("a", "S", "M", 1), ("l", "M", "M", 1),
                            ("b", "M", "T", 1)],
                           [("a", "b"), ("a", "l", "b")])
        report = self.cross_check(
            graph,
            obs([cand(graph, "s", node="S")],
                [cand(graph, "m", node="M")],
                [cand(graph, "t", node="T")]))
        # Three acceptance timings over the two loop laps.
        self.assertEqual(report["count"], 3)
        self.assertEqual(report["roads"], {"a": 3, "l": 6, "b": 3})

    def test_three_road_forbidden_rule_across_observations(self):
        graph = make_graph([("a", "S", "U", 1), ("b", "U", "V", 1),
                            ("c", "V", "T", 1)], [("a", "b", "c")])
        observations = obs(*[[cand(graph, name, node=name)] for name in "SUVT"])
        self.assertIsNone(ambiguity(graph, observations))
        self.cross_check(graph, observations)

    def test_distinct_candidates_same_location_are_separate_events(self):
        graph = make_graph([("a", "S", "T", 1)])
        report = self.cross_check(
            graph,
            obs([cand(graph, "s", node="S")],
                [cand(graph, "m1", node="T"), cand(graph, "m2", node="T")],
                [cand(graph, "t", node="T")]))
        assert_report_equal(
            self, report, F(1), 2,
            [{"s": 2}, {"m1": 1, "m2": 1}, {"t": 2}], {"a": 2})

    def test_initial_interior_occupation_counted_once_per_explanation(self):
        graph = make_graph([("a", "S", "M", 2), ("b", "M", "T", 1),
                            ("c", "M", "T", 1)])
        p = cand(graph, "p", road="a", offset=1)
        q = cand(graph, "q", road="a", offset=1)
        t = cand(graph, "t", node="T")
        report = self.cross_check(graph, obs([p], [q], [t]))
        # Two parallel exits; the occupied road 'a' is entered only initially.
        assert_report_equal(
            self, report, F(2), 2,
            [{"p": 2}, {"q": 2}, {"t": 2}], {"a": 2, "b": 1, "c": 1})

    def test_acceptance_timing_on_shared_route_is_separate_explanation(self):
        # Same road sequence, but the middle observation can be accepted at
        # different moments; enumeration keys these as distinct events.
        graph = make_graph([("a", "S", "M", 1), ("l", "M", "M", 1),
                            ("b", "M", "T", 1)],
                           [("a", "b"), ("a", "l", "b")])
        report = ambiguity(
            graph,
            obs([cand(graph, "s", node="S")],
                [cand(graph, "m", node="M")],
                [cand(graph, "t", node="T")]))
        self.assertEqual(report["count"], 3)
        self.cross_check(
            graph,
            obs([cand(graph, "s", node="S")],
                [cand(graph, "m", node="M")],
                [cand(graph, "t", node="T")]))

    def test_no_solution_and_empty_candidate_set(self):
        graph = make_graph([("a", "S", "T", 1)], extra=("I",))
        unreachable = obs([cand(graph, "i", node="I")],
                          [cand(graph, "t", node="T")])
        self.assertIsNone(ambiguity(graph, unreachable))
        empty = obs([cand(graph, "s", node="S")], [])
        self.assertIsNone(ambiguity(graph, empty))
        self.cross_check(graph, unreachable)
        self.cross_check(graph, empty)

    def test_penalty_breaks_tie_and_zero_support_is_reported(self):
        graph = make_graph([("sm", "S", "M", 1), ("mt", "M", "T", 2),
                            ("sx", "S", "X", 2), ("xt", "X", "T", 1)])
        report = self.cross_check(
            graph,
            obs([cand(graph, "s", node="S")],
                [cand(graph, "m", node="M", penalty=1),
                 cand(graph, "x", node="X")],
                [cand(graph, "t", node="T")]))
        self.assertEqual(report["candidates"][1]["m"], 0)
        self.assertEqual(report["roads"]["sm"], 0)


class FuzzAgainstEnumeration(unittest.TestCase):
    """Random small finite graphs: DP statistics equal exhaustive events."""

    def test_randomized_differential(self):
        import random
        rng = random.Random(20241116)
        for case in range(30):
            rows = [(f"e{i}", rng.choice("ABC"), rng.choice("ABC"),
                     F(rng.randint(1, 4), rng.randint(1, 2)))
                    for i in range(4)]
            patterns = []
            for _ in range(3):
                edge = rng.choice(rows)
                following = [r for r in rows if r[1] == edge[2]]
                if not following:
                    continue
                second = rng.choice(following)
                pattern = [edge[0], second[0]]
                third = [r for r in rows if r[1] == second[2]]
                if third and rng.randrange(2):
                    pattern.append(rng.choice(third)[0])
                patterns.append(pattern)
            graph = make_graph(rows, patterns, extra=("A", "B", "C", "I"))
            groups = []
            for layer in range(3):
                choices = []
                for j in range(2):
                    if rng.randrange(2):
                        choices.append(cand(
                            graph, str(j), node=rng.choice("ABCI"),
                            penalty=F(rng.randrange(2), 2)))
                    else:
                        road = rng.choice(list(graph.roads.values()))
                        choices.append(cand(
                            graph, str(j), road=road.id,
                            offset=road.length / 2,
                            penalty=F(rng.randrange(2), 2)))
                groups.append(choices)
            observations = obs(*groups)
            with self.subTest(case=case):
                expected = enumerate_report(graph, observations)
                report = ambiguity(graph, observations)
                if expected is None:
                    self.assertIsNone(report)
                    continue
                self.assertEqual(report["total"], expected["total"])
                self.assertEqual(report["count"], expected["count"])
                self.assertEqual([dict(x) for x in report["candidates"]],
                                 [dict(x) for x in expected["candidates"]])
                self.assertEqual(dict(report["roads"]), dict(expected["roads"]))


class BranchChainAccounting(unittest.TestCase):
    """Tandem equal-length double-branch chains: 2**b explanations."""

    @staticmethod
    def chain_graph(b):
        # b tandem stages, each with two equal-length parallel roads; the only
        # choices are the 2**b branch combinations between the two endpoints.
        rows = []
        for stage in range(1, b + 1):
            rows.append((f"x{stage}", f"N{stage - 1}", f"N{stage}", 1))
            rows.append((f"y{stage}", f"N{stage - 1}", f"N{stage}", 1))
        return make_graph(rows)

    @staticmethod
    def chain_observations(graph, b):
        return obs([cand(graph, "first", node="N0")],
                   [cand(graph, "last", node=f"N{b}")])

    def test_exponential_explanations_from_linear_state_work(self):
        calls = []
        original = routing.connection_summaries

        def counting(graph, source, target, history):
            calls.append((source, target, history))
            return original(graph, source, target, history)

        # trajectory resolves the summary through its routing module binding.
        with patch.object(routing, "connection_summaries", side_effect=counting):
            table = []
            for b in range(1, 9):
                graph = self.chain_graph(b)
                observations = self.chain_observations(graph, b)
                calls.clear()
                report = trajectory.ambiguity(graph, observations)
                self.assertEqual(report["count"], 2 ** b)
                self.assertEqual(report["total"], F(b))
                self.assertEqual(report["candidates"][0]["first"], 2 ** b)
                self.assertEqual(report["candidates"][1]["last"], 2 ** b)
                # Every branch road appears in exactly half of the explanations.
                for stage in range(1, b + 1):
                    self.assertEqual(report["roads"][f"x{stage}"], 2 ** (b - 1))
                    self.assertEqual(report["roads"][f"y{stage}"], 2 ** (b - 1))
                table.append((b, report["count"], len(calls)))
                # Two observation layers evaluate a single connection summary;
                # the 2**b explanations live in exact integers, never objects.
                self.assertEqual(len(calls), 1)
        # Explanations double per stage while summary/state work stays linear.
        for (b1, count1, work1), (b2, count2, work2) in zip(table, table[1:]):
            self.assertEqual(count2, 2 * count1)
            self.assertEqual(work2, work1)

    def test_chain_matches_independent_enumeration(self):
        for b in range(1, 5):
            graph = self.chain_graph(b)
            observations = self.chain_observations(graph, b)
            expected = enumerate_report(graph, observations)
            report = ambiguity(graph, observations)
            self.assertEqual(report["count"], expected["count"])
            self.assertEqual(dict(report["roads"]), dict(expected["roads"]))
            self.assertEqual([dict(x) for x in report["candidates"]],
                             [dict(x) for x in expected["candidates"]])


class SubdivisionInvariance(unittest.TestCase):
    """Extra interior cuts and repeated observations never change statistics."""

    def test_additional_interior_cuts_leave_report_unchanged(self):
        graph = make_graph([("a", "S", "M", 3), ("b", "M", "T", 1)])
        coarse = obs([cand(graph, "s", node="S")],
                     [cand(graph, "t", node="T")])
        fine = obs([cand(graph, "s", node="S")],
                   [cand(graph, "m", road="a", offset=1)],
                   [cand(graph, "m2", road="a", offset=2)],
                   [cand(graph, "again", road="a", offset=2)],
                   [cand(graph, "t", node="T")])
        coarse_report = ambiguity(graph, coarse)
        fine_report = ambiguity(graph, fine)
        self.assertEqual(fine_report["total"], coarse_report["total"])
        self.assertEqual(fine_report["count"], coarse_report["count"])
        self.assertEqual(fine_report["roads"], coarse_report["roads"])
        # Fine layers are individually still fully supported.
        for layer in fine_report["candidates"]:
            self.assertEqual(sum(layer.values()), fine_report["count"])
        # Enumeration agrees with both.
        from tests.enumeration import enumerate_report as enum
        self.assertEqual(enum(graph, fine)["roads"], fine_report["roads"])
        self.assertEqual(enum(graph, coarse)["count"], coarse_report["count"])

    def test_node_subdivision_does_not_create_entries(self):
        graph = make_graph([("a", "S", "M", 1), ("b", "M", "T", 1)])
        without_middle = obs([cand(graph, "s", node="S")],
                             [cand(graph, "t", node="T")])
        with_middle = obs([cand(graph, "s", node="S")],
                          [cand(graph, "m", node="M")],
                          [cand(graph, "t", node="T")])
        self.assertEqual(ambiguity(graph, with_middle)["roads"],
                         ambiguity(graph, without_middle)["roads"])


class InvalidInputs(unittest.TestCase):
    def test_empty_observations_and_bad_candidates_raise(self):
        graph = make_graph([("a", "S", "T", 1)])
        with self.assertRaises(ValueError):
            ambiguity(graph, ())
        bad = (Observation(F(0), (Candidate("x", Position(node="S"), F(-1)),)),)
        with self.assertRaises(ValueError):
            ambiguity(graph, bad)
        with self.assertRaises(ValueError):
            ambiguity(graph, (Observation(F(1),
                                          (Candidate("x", Position(node="S"), F(0)),)),
                             Observation(F(0),
                                         (Candidate("y", Position(node="T"), F(0)),))))


if __name__ == "__main__":
    unittest.main()
