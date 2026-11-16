"""CLI: python3.11 -B run.py {route,analyze,demo} samples.json"""
from __future__ import annotations

import argparse
import json
import sys

from roadnet import Graph, fields, read_observations
from routing import shortest_route
from trajectory import analyze


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
    raise ValueError("unknown operation")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Exact road observation analysis")
    parser.add_argument("operation", choices=("route", "analyze", "demo"))
    parser.add_argument("input")
    args = parser.parse_args(argv)
    try:
        with open(args.input, encoding="utf-8") as stream:
            data = json.load(stream)
        if args.operation == "demo":
            output = {"route": execute("route", data), "analysis": execute("analyze", data)}
            if output["route"]["status"] != "ok" or output["analysis"]["status"] != "ok":
                raise ValueError("demo requires reachable route and observations")
        else:
            output = execute(args.operation, data)
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
