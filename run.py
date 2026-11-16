"""CLI: python3.11 -B run.py {route,analyze,ambiguity,demo} <input.json>"""
from __future__ import annotations

import argparse
import json
import sys

from roadnet import Graph, fields, read_observations
from routing import shortest_route
from trajectory import ambiguity, analyze


def render(result):
    if result is None:
        return {"status": "unreachable"}
    output = {
        "status": "ok",
        "roads": list(result.roads),
        "distance": str(result.distance),
        "spans": [{"road": s.road, "start": str(s.start), "end": str(s.end)}
                  for s in result.spans],
    }
    if hasattr(result, "selected"):
        output.update(selected=list(result.selected), penalty=str(result.penalty),
                      total=str(result.total))
    else:
        output["total"] = str(result.distance)
    return output


def render_ambiguity(report):
    if report is None:
        return {"status": "unreachable"}
    return {
        "status": "ok",
        "total": str(report["total"]),
        "count": report["count"],
        "candidates": [dict(layer) for layer in report["candidates"]],
        "roads": dict(report["roads"]),
    }


def ambiguity_item(item):
    """Render one self-contained example, tolerating wrapper fields."""
    graph = Graph.from_dict(item["graph"])
    report = ambiguity(graph, read_observations(graph, item["observations"]))
    rendered = render_ambiguity(report)
    if "name" in item:
        rendered = {"name": item["name"], **rendered}
    return rendered


def execute(mode, data):
    fields(data, ("graph",), ("source", "target", "observations"))
    graph = Graph.from_dict(data["graph"])
    if mode == "route":
        if "source" not in data or "target" not in data:
            raise ValueError("route requires source and target")
        return render(shortest_route(graph, graph.position(data["source"]),
                                     graph.position(data["target"])))
    if mode == "analyze":
        if "observations" not in data:
            raise ValueError("analyze requires observations")
        return render(analyze(graph, read_observations(graph, data["observations"])))
    if mode == "ambiguity":
        if "observations" not in data:
            raise ValueError("ambiguity requires observations")
        return render_ambiguity(
            ambiguity(graph, read_observations(graph, data["observations"])))
    raise ValueError("unknown operation")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Exact road observation analysis")
    parser.add_argument("operation",
                        choices=("route", "analyze", "ambiguity", "demo"))
    parser.add_argument("input")
    args = parser.parse_args(argv)
    try:
        with open(args.input, encoding="utf-8") as stream:
            data = json.load(stream)
        if args.operation == "demo":
            output = {"route": execute("route", data), "analysis": execute("analyze", data)}
            if output["route"]["status"] != "ok" or output["analysis"]["status"] != "ok":
                raise ValueError("demo requires reachable route and observations")
        elif args.operation == "ambiguity":
            # ambiguity_examples.json is a list of self-contained inputs; a
            # single request object (graph plus observations, optionally with
            # wrapper fields such as name/expected) is also accepted. Only the
            # strict documented request shape routes through execute().
            if isinstance(data, list):
                output = [ambiguity_item(item) for item in data]
            elif set(data) <= {"graph", "observations"}:
                output = execute("ambiguity", data)
            else:
                output = ambiguity_item(data)
        else:
            output = execute(args.operation, data)
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
