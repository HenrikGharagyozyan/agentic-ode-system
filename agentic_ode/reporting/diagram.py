"""Drawing of the compiled graph (PNG via matplotlib) and its Mermaid source."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from agentic_ode.graph import build_graph  # noqa: E402

_POS = {
    "__start__": (0, 2.2), "orchestrator": (0, 1), "scientist": (-2.2, -0.6),
    "general": (0, -0.6), "executor": (2.2, -0.6), "__end__": (3.1, 1.9),
    "human": (-2.6, 1.6),
    # HW3 hardware turn, in its own column on the right
    "hw_decide": (4.9, 1.0), "hw_approve": (4.9, 0.2), "hw_call": (4.9, -0.6),
}
_LABELS = {"__start__": "START\n(user message)", "__end__": "END"}
_BLUE, _GREY = "#2b5c8a", "#555555"


class _NoLLM:
    """Only used to compile the graph for drawing; never called."""

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, *a, **k):
        return self


def graph_png(workdir: Path) -> tuple[bytes, str]:
    """Draw the compiled graph's nodes and edges; also return its Mermaid source."""
    g = build_graph(_NoLLM(), workdir).get_graph()
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    for name, (x, y) in _POS.items():
        terminal = name.startswith("__")
        ax.text(x, y, _LABELS.get(name, name), ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", fc="#eeeeee" if terminal else "#dbe8f6",
                          ec=_GREY if terminal else _BLUE))
    for e in g.edges:
        (x0, y0), (x1, y1) = _POS[e.source], _POS[e.target]
        back = e.source != "orchestrator" and e.target == "orchestrator"
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                     shrinkA=26, shrinkB=26, lw=1.3,
                                     connectionstyle=f"arc3,rad={0.25 if back else 0.0}",
                                     ls="-" if (back or e.source == "__start__") else "--",
                                     color=_GREY if back else _BLUE))
    ax.text(0, -1.45, "dashed: conditional edges chosen by the orchestrator  |  solid: plain edges",
            ha="center", fontsize=9, color="#444444")
    ax.set_xlim(-3.2, 5.8)
    ax.set_ylim(-1.7, 2.7)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue(), g.draw_mermaid()
