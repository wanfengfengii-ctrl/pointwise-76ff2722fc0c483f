"""New public inputs exercised only through the existing single-answer API."""
import json
import unittest
from fractions import Fraction
from pathlib import Path

from roadnet import Graph, read_observations
from trajectory import analyze
from tests.reference import check_analysis, optimum


class AmbiguityInputTests(unittest.TestCase):
    def test_public_inputs_keep_existing_cost_and_route_semantics(self):
        path = Path(__file__).resolve().parents[1] / "ambiguity_examples.json"
        examples = json.loads(path.read_text(encoding="utf-8"))
        for example in examples:
            with self.subTest(example=example["name"]):
                graph = Graph.from_dict(example["graph"])
                observations = read_observations(graph, example["observations"])
                expected = example["expected"]
                total = None if expected is None else Fraction(expected["total"])
                result = analyze(graph, observations)
                self.assertEqual(optimum(graph, observations), total)
                self.assertEqual(None if result is None else result.total, total)
                if result is not None:
                    check_analysis(graph, observations, result)


if __name__ == "__main__":
    unittest.main()
