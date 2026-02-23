#!/usr/bin/env python3
"""Render an MCTS tree from a saved results JSON as a Graphviz image (PNG/SVG)."""

import argparse
import json
import sys
from pathlib import Path

import graphviz


def _node_color(node: dict) -> str:
    if node["depth"] == 0:
        return "#D0D0D0"       # gray — root
    if node["is_terminal"]:
        return "#90EE90"       # green — terminal (completed answer)
    if not node["children"]:
        return "#FFD580"       # orange — non-terminal leaf (unexpanded)
    return "#ADD8E6"           # light blue — internal


def _node_label(node: dict) -> str:
    preview = node["text_preview"]
    if len(preview) > 50:
        preview = preview[:47] + "..."
    lines = [
        f"d={node['depth']}  visits={node['visit_count']}",
        f"Q={node['q_value']:.3f}  PRM={node['prm_score']:.3f}",
        preview,
    ]
    return "\n".join(lines)


def build_graph(tree: dict, fmt: str = "png") -> graphviz.Digraph:
    dot = graphviz.Digraph(format=fmt)
    dot.attr(rankdir="TB", fontname="Helvetica", bgcolor="white")
    dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="10")
    dot.attr("edge", fontname="Helvetica", fontsize="9")

    def _add(node: dict, parent_id: str | None = None):
        nid = str(node["id"])
        dot.node(
            nid,
            label=_node_label(node),
            fillcolor=_node_color(node),
        )
        if parent_id is not None:
            edge_attrs: dict = {}
            if node["on_best_path"]:
                edge_attrs.update(color="red", penwidth="2.5", style="bold")
            else:
                edge_attrs.update(color="gray")
            dot.edge(parent_id, nid, **edge_attrs)

        for child in node.get("children", []):
            _add(child, nid)

    _add(tree)
    return dot


def render_tree(tree: dict, output: str, fmt: str = "png") -> str:
    """Render *tree* dict to *output* file. Returns the output path."""
    dot = build_graph(tree, fmt=fmt)
    out_path = Path(output)
    # graphviz .render() appends the format extension itself, so strip it
    stem = str(out_path.with_suffix(""))
    rendered = dot.render(stem, cleanup=True)
    return rendered


def main():
    parser = argparse.ArgumentParser(description="Visualize an MCTS tree from results JSON")
    parser.add_argument("--results", required=True, help="Path to mcts_results JSON file")
    parser.add_argument("--problem_id", type=int, default=0, help="Index of the problem to visualize")
    parser.add_argument("--output", default="tree.png", help="Output file path (e.g. tree.png, tree.svg)")
    parser.add_argument("--format", default=None, help="Output format (png, svg). Inferred from --output if omitted.")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    # The results file is a list of problem results
    if isinstance(data, list):
        if args.problem_id >= len(data):
            print(f"Error: problem_id {args.problem_id} out of range (file has {len(data)} problems)", file=sys.stderr)
            sys.exit(1)
        problem = data[args.problem_id]
    elif isinstance(data, dict):
        problem = data
    else:
        print("Error: unexpected JSON structure", file=sys.stderr)
        sys.exit(1)

    if "tree" not in problem:
        print("Error: no 'tree' key found in the result — was the run done after tree serialization was added?", file=sys.stderr)
        sys.exit(1)

    fmt = args.format or Path(args.output).suffix.lstrip(".")
    if fmt not in ("png", "svg", "pdf"):
        fmt = "png"

    out = render_tree(problem["tree"], args.output, fmt=fmt)
    print(f"Tree rendered to {out}")


if __name__ == "__main__":
    main()
