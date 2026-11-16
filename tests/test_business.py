import copy
import random
import unittest
from fractions import Fraction as F

from roadnet import Candidate, Graph, Observation, Position, read_observations
from routing import shortest_route
from trajectory import analyze
from run import execute, render
from tests.reference import check_analysis, optimum, replay


def graph(rows, forbidden=(), extra=()):
    nodes = sorted(set(extra) | {n for _, a, b, _ in rows for n in (a, b)})
    return Graph.from_dict({
        "nodes": nodes,
        "roads": [{"id": r, "tail": a, "head": b, "length": str(w)}
                  for r, a, b, w in rows],
        "forbidden": [list(p) for p in forbidden],
    })


def candidate(g, cid, node=None, road=None, offset=0, penalty=0):
    raw = {"node": node} if node is not None else {"road": road, "offset": str(offset)}
    return Candidate(cid, g.position(raw), F(penalty))


def observations(*groups):
    return tuple(Observation(F(i), tuple(group)) for i, group in enumerate(groups))


class BusinessTests(unittest.TestCase):
    def checked(self, g, obs, expected):
        result = analyze(g, obs)
        self.assertEqual(optimum(g, obs), expected)
        self.assertEqual(None if result is None else result.total, expected)
        if result is not None:
            check_analysis(g, obs, result)
        return result

    def test_more_expensive_arrival_can_continue(self):
        g = graph([("a", "S", "M", 2), ("x", "S", "X", 1),
                   ("y", "X", "M", 2), ("c", "M", "T", 1)], [("a", "c")])
        obs = observations([candidate(g, "s", node="S")],
                           [candidate(g, "m", node="M")],
                           [candidate(g, "t", node="T")])
        self.assertEqual(shortest_route(g, obs[0].candidates[0].position,
                                       obs[1].candidates[0].position).distance, 2)
        result = self.checked(g, obs, F(4))
        self.assertEqual(result.roads, ("x", "y", "c"))

    def test_three_road_rule_across_internal_repeated_observations(self):
        g = graph([("a", "S", "U", 1), ("d", "S", "U", 2),
                   ("b", "U", "V", 1), ("c", "V", "T", 1)], [("a", "b", "c")])
        obs = observations(
            [candidate(g, "s", node="S")],
            [candidate(g, "p", road="b", offset=F(1, 3))],
            [candidate(g, "q", road="b", offset=F(2, 3))],
            [candidate(g, "r", road="b", offset=F(2, 3))],
            [candidate(g, "t", node="T")])
        result = self.checked(g, obs, F(4))
        self.assertEqual(result.roads, ("d", "b", "c"))
        self.assertEqual(result.spans[1].start, 0)
        self.assertEqual(result.spans[1].end, 1)
        simple = observations(obs[0].candidates, obs[-1].candidates)
        self.assertEqual(self.checked(g, simple, F(4)).roads, result.roads)

    def test_rule_across_two_node_boundaries(self):
        g = graph([("a", "S", "U", 1), ("b", "U", "V", 1),
                   ("c", "V", "T", 1)], [("a", "b", "c")])
        self.checked(g, observations(*[[candidate(g, n, node=n)] for n in "SUVT"]), None)

    def test_candidate_choice_and_exact_accounting(self):
        g = graph([("a", "S", "M", F(1, 3)), ("b", "M", "T", F(1, 3)),
                   ("x", "S", "X", F(1, 2)), ("y", "X", "T", F(1, 2))])
        obs = observations([candidate(g, "s", node="S", penalty=F(1, 7))],
                           [candidate(g, "m", node="M", penalty=2),
                            candidate(g, "x", node="X", penalty=F(1, 5))],
                           [candidate(g, "t", node="T")])
        result = self.checked(g, obs, F(47, 35))
        self.assertEqual(result.selected, ("s", "x", "t"))
        self.assertEqual((result.distance, result.penalty), (F(1), F(12, 35)))

    def test_internal_forward_and_repeated_location(self):
        g = graph([("b", "U", "V", 3)])
        c = candidate(g, "p", road="b", offset=1)
        obs = observations([c], [c], [candidate(g, "q", road="b", offset=2)])
        result = self.checked(g, obs, F(1))
        self.assertEqual(result.roads, ("b",))
        self.assertEqual((result.spans[0].start, result.spans[0].end), (1, 2))
        still = self.checked(g, observations([c], [c]), F(0))
        self.assertEqual(still.roads, ("b",))
        self.assertEqual(still.spans[0].start, still.spans[0].end)

    def test_initial_interior_occupancy_and_zero_node_query(self):
        g = graph([("a", "S", "M", 2), ("b", "M", "T", 1)], [("a", "b")])
        c = candidate(g, "inside", road="a", offset=1)
        obs = observations([c], [c], [candidate(g, "t", node="T")])
        self.checked(g, obs, None)
        self.assertIsNone(shortest_route(g, c.position, Position(node="T")))
        route = shortest_route(g, Position(node="M"), Position(node="M"))
        self.assertEqual((route.distance, route.roads), (0, ()))
        stationary = shortest_route(g, c.position, c.position)
        self.assertEqual((stationary.distance, stationary.roads), (0, ("a",)))
        replay(g, c.position, c.position, stationary.spans,
               stationary.distance, stationary.roads)

    def test_candidate_is_chosen_for_whole_trip(self):
        g = graph([("sm", "S", "M", 1), ("mt", "M", "T", 10),
                   ("sx", "S", "X", 2), ("xt", "X", "T", 1)])
        obs = observations([candidate(g, "s", node="S")],
                           [candidate(g, "m", node="M"), candidate(g, "x", node="X")],
                           [candidate(g, "t", node="T")])
        result = self.checked(g, obs, F(3))
        self.assertEqual(result.selected, ("s", "x", "t"))

    def test_internal_backward_requires_real_reentry(self):
        rows = [("b", "U", "V", 3), ("r", "V", "U", 2)]
        for forbidden, expected in [((), F(4)), ([("b", "r", "b")], None)]:
            g = graph(rows, forbidden)
            obs = observations([candidate(g, "p", road="b", offset=2)],
                               [candidate(g, "q", road="b", offset=1)])
            result = self.checked(g, obs, expected)
            route = shortest_route(g, obs[0].candidates[0].position,
                                   obs[1].candidates[0].position)
            self.assertEqual(None if route is None else route.distance, expected)
            if result:
                self.assertEqual(result.roads, ("b", "r", "b"))

    def test_self_loop_entries_can_change_continuation(self):
        g = graph([("a", "S", "M", 1), ("l", "M", "M", 1),
                   ("b", "M", "T", 1)], [("a", "b"), ("a", "l", "b")])
        obs = observations([candidate(g, "s", node="S")],
                           [candidate(g, "m", node="M")],
                           [candidate(g, "t", node="T")])
        result = self.checked(g, obs, F(4))
        self.assertEqual(result.roads, ("a", "l", "l", "b"))

    def test_endpoint_normalization_does_not_enter_road(self):
        g = graph([("a", "S", "M", 1), ("b", "M", "T", 1)], [("a", "b")])
        self.assertEqual(g.position({"road": "b", "offset": 0}), Position(node="M"))
        self.assertEqual(g.position({"road": "a", "offset": 1}), Position(node="M"))
        obs = observations([candidate(g, "s", node="S")],
                           [candidate(g, "m", road="b", offset=0)],
                           [candidate(g, "t", node="T")])
        self.checked(g, obs, None)
        start = observations(obs[1].candidates, obs[2].candidates)
        self.assertEqual(self.checked(g, start, F(1)).roads, ("b",))

    def test_single_empty_unreachable_and_equal_cost(self):
        g = graph([("a", "S", "T", 1), ("b", "S", "T", 1)], extra=("I",))
        obs = observations([candidate(g, "s", node="S")],
                           [candidate(g, "t", node="T")])
        self.checked(g, obs, F(1))
        self.checked(g, observations([candidate(g, "i", node="I", penalty=F(2, 3))]), F(2, 3))
        self.checked(g, observations(obs[0].candidates, []), None)
        self.checked(g, observations([candidate(g, "i", node="I")], obs[1].candidates), None)
        route = shortest_route(g, Position(node="S"), Position(node="T"))
        replay(g, Position(node="S"), Position(node="T"), route.spans, route.distance, route.roads)

    def test_invalid_graph_and_positions(self):
        base = {"nodes": ["S", "T"], "roads": [
            {"id": "a", "tail": "S", "head": "T", "length": "1"}]}
        for length in (0, -1, True, 0.25, "1/0", "NaN"):
            bad = copy.deepcopy(base)
            bad["roads"][0]["length"] = length
            with self.assertRaises(ValueError):
                Graph.from_dict(bad)
        for bad in (
            {"nodes": ["S", "S"], "roads": []},
            {"nodes": [], "roads": base["roads"]},
            dict(base, roads=base["roads"] * 2),
            dict(base, forbidden=[["a", "a"]]),
            dict(base, forbidden=[["missing", "a"]]),
            dict(base, forbidden=[["a"]]),
        ):
            with self.assertRaises(ValueError):
                Graph.from_dict(bad)
        g = Graph.from_dict(base)
        for pos in ({"node": "?"}, {"road": "a", "offset": 2},
                    {"road": "a", "offset": True}, {"node": "S", "road": "a"},
                    {"road": "a"}, {"node": "S", "extra": 1}):
            with self.assertRaises(ValueError):
                g.position(pos)
        with self.assertRaises(ValueError):
            shortest_route(g, Position(node="S", road="a"), Position(node="T"))

    def test_invalid_observations_and_public_json_api(self):
        data = {"graph": {"nodes": ["S", "T", "I"], "roads": [
            {"id": "a", "tail": "S", "head": "T", "length": "1/3"}]},
            "source": {"node": "S"}, "target": {"node": "T"},
            "observations": [
                {"time": 0, "candidates": [
                    {"id": "s", "position": {"node": "S"}, "penalty": "1/5"}]},
                {"time": 1, "candidates": [
                    {"id": "t", "position": {"node": "T"}, "penalty": 0}]}]}
        self.assertEqual(execute("route", data)["total"], "1/3")
        self.assertEqual(execute("analyze", data)["total"], "8/15")
        unreachable = dict(data, target={"node": "I"})
        self.assertEqual(execute("route", unreachable), {"status": "unreachable"})
        g = Graph.from_dict(data["graph"])
        for raw in ([], list(reversed(data["observations"])),
                    [{"time": True, "candidates": []}],
                    [{"time": 0, "candidates": data["observations"][0]["candidates"] * 2}],
                    [{"time": 0, "candidates": [
                        {"id": "bad", "position": {"node": "S"}, "penalty": "-1"}]}]):
            with self.assertRaises(ValueError):
                read_observations(g, raw)
        with self.assertRaises(ValueError):
            analyze(g, (Observation(F(0), (Candidate("bad", Position(node="S"), F(-1)),)),))
        self.assertEqual(render(None), {"status": "unreachable"})

    def test_fixed_seed_finite_state_differential(self):
        rng = random.Random(31017)
        for case in range(45):
            rows = [(f"e{i}", rng.choice("ABC"), rng.choice("ABC"),
                     F(rng.randint(1, 5), rng.randint(1, 3))) for i in range(5)]
            patterns = []
            for _ in range(5):
                a = rng.choice(rows)
                next_roads = [r for r in rows if r[1] == a[2]]
                if not next_roads:
                    continue
                b = rng.choice(next_roads)
                pattern = [a[0], b[0]]
                third = [r for r in rows if r[1] == b[2]]
                if third and rng.randrange(2):
                    pattern.append(rng.choice(third)[0])
                patterns.append(pattern)
            g = graph(rows, patterns, extra=("A", "B", "C", "I"))
            groups = []
            for layer in range(4):
                choices = []
                for j in range(2):
                    if rng.randrange(2):
                        c = candidate(g, str(j), node=rng.choice("ABCI"),
                                      penalty=F(rng.randrange(3), 3))
                    else:
                        road = rng.choice(list(g.roads.values()))
                        c = candidate(g, str(j), road=road.id, offset=road.length / 2,
                                      penalty=F(rng.randrange(3), 3))
                    choices.append(c)
                groups.append(choices)
            obs = observations(*groups)
            with self.subTest(case=case):
                self.checked(g, obs, optimum(g, obs))
                source, target = obs[0].candidates[0].position, obs[-1].candidates[0].position
                query_obs = observations([Candidate("s", source)], [Candidate("t", target)])
                expected = optimum(g, query_obs)
                route = shortest_route(g, source, target)
                self.assertEqual(None if route is None else route.distance, expected)
                if route is not None:
                    replay(g, source, target, route.spans, route.distance, route.roads)


if __name__ == "__main__":
    unittest.main()
