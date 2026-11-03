"""Looking at a knowledge graph: a text summary, a picture, and one self-contained interactive page.

Only the standard library at import time; the picture needs matplotlib and networkx (both are in Colab and Kaggle).
The functions take the dict that ``GET /api/graph/data/<id>`` returns, so they can be tested without a backend.
"""

from __future__ import annotations

import html as htmllib
import json
from typing import Any, Dict


def _kind(node: Dict[str, Any]) -> str:
    return next((label for label in node.get("labels") or [] if label not in ("Entity", "Episodic")), "Entity")


def _degrees(edges: list) -> Dict[str, int]:
    degree: Dict[str, int] = {}
    for e in edges:
        for end in (e.get("source_node_uuid"), e.get("target_node_uuid")):
            degree[end] = degree.get(end, 0) + 1
    return degree


def graph_summary(data: Dict[str, Any], facts: int = 25) -> str:
    nodes, edges = data.get("nodes") or [], data.get("edges") or []
    names = {n["uuid"]: n.get("name") or "?" for n in nodes}
    degree = _degrees(edges)
    kinds: Dict[str, int] = {}
    for n in nodes:
        kinds[_kind(n)] = kinds.get(_kind(n), 0) + 1
    lines = [f"{len(nodes)} entities, {len(edges)} facts", "", "Entities by type:"]
    lines += [f"  {count:4d}  {kind}" for kind, count in sorted(kinds.items(), key=lambda kv: -kv[1])]
    hubs = [n for n in sorted(nodes, key=lambda n: -degree.get(n["uuid"], 0))[:10] if degree.get(n["uuid"], 0)]
    lines += ["", "Most connected:"] + [f"  {degree[n['uuid']]:4d}  {n.get('name')} ({_kind(n)})" for n in hubs]
    lines += ["", f"Facts (first {min(facts, len(edges))} of {len(edges)}):"]
    for e in edges[:facts]:
        source = e.get("source_node_name") or names.get(e.get("source_node_uuid"), "?")
        target = e.get("target_node_name") or names.get(e.get("target_node_uuid"), "?")
        lines.append(f"  {source} -[{e.get('name') or 'RELATES_TO'}]-> {target}: {(e.get('fact') or '')[:140]}")
    return "\n".join(lines)


def draw_graph(data: Dict[str, Any], max_drawn: int = 150) -> None:
    """A picture of the graph. When there are more than ``max_drawn`` entities, the most connected ones are drawn."""
    import matplotlib.pyplot as plt
    import networkx as nx

    nodes, edges = data.get("nodes") or [], data.get("edges") or []
    if not nodes:
        print("the graph is empty")
        return
    degree = _degrees(edges)
    shown = sorted(nodes, key=lambda n: -degree.get(n["uuid"], 0))[:max_drawn]
    ids = {n["uuid"] for n in shown}
    graph = nx.Graph()
    for n in shown:
        graph.add_node(n["uuid"], name=n.get("name") or "?", kind=_kind(n))
    for e in edges:
        if e.get("source_node_uuid") in ids and e.get("target_node_uuid") in ids:
            graph.add_edge(e["source_node_uuid"], e["target_node_uuid"])
    kinds = sorted({graph.nodes[u]["kind"] for u in graph})
    palette = plt.get_cmap("tab20")
    colour = {kind: palette(i % 20) for i, kind in enumerate(kinds)}
    layout = nx.spring_layout(graph, seed=1, k=1.2 / max(len(graph) ** 0.5, 1))
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_edges(graph, layout, alpha=0.35, width=1)
    nx.draw_networkx_nodes(graph, layout, node_color=[colour[graph.nodes[u]["kind"]] for u in graph],
                           node_size=[120 + 60 * degree.get(u, 0) for u in graph])
    above = {u: (x, y + 0.045) for u, (x, y) in layout.items()}        # labels above the dots, not on them
    nx.draw_networkx_labels(graph, above, {u: graph.nodes[u]["name"] for u in graph}, font_size=8)
    for kind in kinds:
        plt.scatter([], [], color=colour[kind], label=kind)
    plt.legend(loc="upper left", fontsize=8)
    note = f" (the {len(graph)} most connected of {len(nodes)})" if len(nodes) > len(graph) else ""
    plt.title(f"{len(nodes)} entities, {len(edges)} facts{note}")
    plt.axis("off")
    plt.show()


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>MiroFish graph</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>body{font-family:sans-serif;margin:0}#legend{padding:8px 12px}#g{height:calc(100vh - 40px)}</style></head><body>
<div id="legend">__LEGEND__</div><div id="g"></div>
<script>const d = __DATA__;
new vis.Network(document.getElementById('g'), {nodes: new vis.DataSet(d.nodes), edges: new vis.DataSet(d.edges)},
  {physics: {stabilization: {iterations: 200}}, edges: {font: {size: 9}, smooth: false}, nodes: {shape: 'dot', size: 14}});
</script></body></html>
"""


def graph_html(data: Dict[str, Any]) -> str:
    """One self-contained page (the data is inline; the drawing library loads from a CDN): drag, zoom, hover for the fact."""
    nodes, edges = data.get("nodes") or [], data.get("edges") or []
    kinds = sorted({_kind(n) for n in nodes})
    palette = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]
    colour = {kind: palette[i % len(palette)] for i, kind in enumerate(kinds)}
    vis_nodes = [{"id": n["uuid"], "label": n.get("name") or "?", "color": colour[_kind(n)],
                  "title": f"{n.get('name')} ({_kind(n)})\n{(n.get('summary') or '')[:300]}"} for n in nodes]
    vis_edges = [{"from": e["source_node_uuid"], "to": e["target_node_uuid"], "arrows": "to", "label": e.get("name") or "",
                  "title": e.get("fact") or ""} for e in edges]
    legend = f"{len(nodes)} entities, {len(edges)} facts &nbsp; " + "".join(
        f'<span style="color:{colour[k]}">&#9679;</span> {htmllib.escape(k)} &nbsp; ' for k in kinds)
    payload = json.dumps({"nodes": vis_nodes, "edges": vis_edges}, ensure_ascii=False).replace("</", "<\\/")
    return _PAGE.replace("__LEGEND__", legend).replace("__DATA__", payload)
