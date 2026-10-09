"""Build report/solution.pdf from a run's transcript.json. Every number comes from that file.

Usage: python report/build_report.py runs/<timestamp>
Needs Google Chrome or Chromium for the HTML -> PDF step.
"""

import base64
import html
import io
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.graph import build_graph  # noqa: E402


def repo_url() -> str:
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return "<URL>"
    m = re.match(r"git@([^:]+):(.+?)(\.git)?$", url)
    return f"https://{m.group(1)}/{m.group(2)}" if m else url.removesuffix(".git")


class _NoLLM:
    """Only used to compile the graph for drawing; never called."""

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, *a, **k):
        return self


def graph_png(tmp: Path) -> tuple[bytes, str]:
    """Draw the compiled graph's nodes and edges; also return its Mermaid source."""
    g = build_graph(_NoLLM(), tmp).get_graph()
    pos = {"__start__": (0, 2.2), "orchestrator": (0, 1), "scientist": (-2.2, -0.6),
           "general": (0, -0.6), "executor": (2.2, -0.6), "__end__": (3.1, 1.9)}
    labels = {"__start__": "START\n(user message)", "__end__": "END"}
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for name, (x, y) in pos.items():
        terminal = name.startswith("__")
        ax.text(x, y, labels.get(name, name), ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", fc="#eeeeee" if terminal else "#dbe8f6",
                          ec="#555555" if terminal else "#2b5c8a"))
    for e in g.edges:
        (x0, y0), (x1, y1) = pos[e.source], pos[e.target]
        back = e.source != "orchestrator" and e.target == "orchestrator"
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                     shrinkA=26, shrinkB=26, lw=1.3,
                                     connectionstyle=f"arc3,rad={0.25 if back else 0.0}",
                                     ls="-" if (back or e.source == "__start__") else "--",
                                     color="#555555" if back else "#2b5c8a"))
    ax.text(0, -1.45, "dashed: conditional edges chosen by the orchestrator  |  solid: plain edges",
            ha="center", fontsize=9, color="#444444")
    ax.set_xlim(-3.2, 3.8)
    ax.set_ylim(-1.7, 2.7)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue(), g.draw_mermaid()


def field(content: str, key: str) -> str:
    for line in content.splitlines():
        if line.startswith(key + ":"):
            return line[len(key) + 1:].strip()
    return ""


def clip(s: str, n: int = 220) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def describe(i: int, turn: dict, runs: list[dict]) -> tuple[str, str]:
    """(what the turn did, orchestrator's reason) derived from the transcript text and run records."""
    role, c = turn["role"], turn["content"]
    if role == "user":
        return "Sends the Task 5 text (task5.md) to the orchestrator.", ""
    if role == "orchestrator":
        return f"Gives the turn to <b>{html.escape(field(c, 'next'))}</b>: {html.escape(clip(field(c, 'instruction')))}", \
            html.escape(field(c, "reason"))
    if role == "executor":
        mine = [r for r in runs if r["turn_index"] == i]
        files = field(c, "Files written") or "none"
        parts = [f"Wrote {html.escape(files)}."]
        for r in mine:
            code = "timeout" if r["exit_code"] is None else r["exit_code"]
            line = [m.group(0) for m in re.finditer(r"^FREE_RUN_TEST_RMSE:.*$", r["stdout"], re.M)]
            res = f"printed <code>{html.escape(line[-1])}</code>" if line else "printed no test RMSE"
            parts.append(f"Ran {html.escape(r['filename'])}: exit code {code}, {res}.")
        if not mine:
            parts.append("No script was run.")
        return " ".join(parts), ""
    return html.escape(clip(c)), ""


def main(run_dir: Path) -> Path:
    data = json.loads((run_dir / "transcript.json").read_text())
    dialogue, runs = data["dialogue"], data["runs"]

    good = [r for r in runs if r["exit_code"] == 0 and r["test_rmse"] is not None]
    final = good[-1] if good else (runs[-1] if runs else None)
    plan_idx = [i for i, t in enumerate(dialogue) if t["role"] == "scientist"
                and (final is None or i < final["turn_index"])]
    plan = dialogue[plan_idx[-1]]["content"] if plan_idx else None

    png, mermaid = graph_png(run_dir)
    img = base64.b64encode(png).decode()

    rows = []
    for i, t in enumerate(dialogue):
        what, reason = describe(i, t, runs)
        rows.append(f"<tr><td>{i}</td><td>{t['role']}</td><td>{what}</td><td>{reason}</td></tr>")

    if good:
        lines = [m.group(0) for m in re.finditer(r"^FREE_RUN_(TRAIN|TEST)_RMSE:.*$", final["stdout"], re.M)]
        result = (f"<p>The script <code>{html.escape(final['filename'])}</code> ran with exit code 0 in executor turn "
                  f"{final['turn_index']} and printed:</p><pre class='result'>{html.escape(chr(10).join(lines))}</pre>"
                  "<p>These lines are copied from the stdout stored in <code>transcript.json</code>; values are in volts. "
                  "For comparison, Task 5 states that predicting zero scores the RMS of the test output, about 0.06 V; "
                  "a test RMSE at or above that level means the free run did not track the measured output.</p>")
    else:
        result = "<p><b>The run did not reach an RMSE.</b> No executor run exited with code 0 and printed a test RMSE.</p>"

    script = ""
    if final:
        src = (run_dir / final["filename"]).read_text()
        status = "the script that printed the RMSE above" if good else "the last script the executor ran (it did not produce an RMSE)"
        script = f"<h2>5. Final script</h2><p><code>{html.escape(final['filename'])}</code>, {status}.</p><pre>{html.escape(src)}</pre>"
        plot = run_dir / "test_plot.png"
        if good and plot.exists() and final is runs[-1]:
            b = base64.b64encode(plot.read_bytes()).decode()
            script += f"<p>Plot saved by this script (measured vs free-run prediction, test segment):</p><img class='plot' src='data:image/png;base64,{b}'>"

    plan_html = f"<pre class='plan'>{html.escape(plan)}</pre>" if plan else "<p>No scientist turn.</p>"
    other_runs = "".join(
        f"<tr><td>{r['turn_index']}</td><td>{html.escape(r['filename'])}</td>"
        f"<td>{'timeout' if r['exit_code'] is None else r['exit_code']}</td>"
        f"<td>{'' if r['train_rmse'] is None else repr(r['train_rmse'])}</td>"
        f"<td>{'' if r['test_rmse'] is None else repr(r['test_rmse'])}</td></tr>" for r in runs)

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Task 6 solution</title><style>
body{{font-family:Helvetica,Arial,sans-serif;font-size:10.5pt;line-height:1.4;margin:0;color:#111}}
h1{{font-size:18pt;margin:0 0 4px}} h2{{font-size:13pt;margin:22px 0 6px;border-bottom:1px solid #ccc}}
table{{border-collapse:collapse;width:100%;font-size:9pt}} td,th{{border:1px solid #bbb;padding:3px 5px;vertical-align:top;text-align:left}}
th{{background:#eee}} pre{{background:#f6f6f6;border:1px solid #ddd;padding:6px;font-size:7.6pt;white-space:pre-wrap;word-break:break-all}}
pre.result{{font-size:11pt;font-weight:bold}} code{{font-size:9pt}} img.graph{{width:80%;display:block;margin:auto}} img.plot{{width:100%}}
.meta td{{border:none;padding:1px 8px 1px 0}} pre.plan{{border-left:3px solid #2b5c8a;font-size:8pt}}
</style></head><body>
<h1>Task 6: an agentic system that solves Task 5</h1>
<p>Repository: {html.escape(repo_url())}</p>
<table class="meta"><tr><td>Model</td><td><code>{html.escape(data['model'])}</code></td></tr>
<tr><td>Run</td><td><code>{html.escape(data['run_dir'])}</code> (full dialogue in transcript.json / transcript.md)</td></tr>
<tr><td>Orchestrator decisions</td><td>{data['decisions']} (limit 8)</td></tr>
<tr><td>Stop reason</td><td>{html.escape(data['stop_reason'])}</td></tr></table>

<h2>1. The graph</h2>
<p>A LangGraph <code>StateGraph</code> with four nodes. The state is the dialogue, a list of turns
<code>{{role, content}}</code> appended with <code>operator.add</code>, plus a counter of orchestrator decisions and the records of real
script runs. The user message enters at the orchestrator. The orchestrator makes one structured model call per turn
(<code>next</code>, <code>instruction</code>, <code>reason</code>) and a conditional edge hands the turn to the chosen role or ends the run;
each specialist returns the turn by a plain edge. Only the executor holds the tools <code>write_file</code> and <code>run_file</code>
(300 s limit). Finish is accepted only when an executor turn carries a test RMSE parsed from the stdout of a run with exit code 0;
otherwise the orchestrator model is asked again with a note that finish is not yet allowed.</p>
<img class="graph" src="data:image/png;base64,{img}">

<h2>2. Turns</h2>
<table><tr><th>#</th><th>role</th><th>what it did</th><th>orchestrator's reason</th></tr>{''.join(rows)}</table>

<h2>3. Free-run test RMSE</h2>{result}
<p>All script runs of this session (values parsed from the printed lines, volts):</p>
<table><tr><th>executor turn</th><th>file</th><th>exit code</th><th>train RMSE</th><th>test RMSE</th></tr>{other_runs or '<tr><td colspan=5>no runs</td></tr>'}</table>

<h2>4. The scientist's plan</h2>
<p>Scientist turn {plan_idx[-1] if plan_idx else '–'}, the latest plan before the final script, verbatim (raw text as the model wrote it).</p>
{plan_html}
{script}
</body></html>"""
    out_html = ROOT / "report" / "solution.html"
    out_html.write_text(page)
    (ROOT / "report" / "graph.mmd").write_text(mermaid)

    chrome = next(filter(None, map(shutil.which, ["google-chrome", "chromium", "chromium-browser"])), None)
    if chrome is None:
        sys.exit("Chrome/Chromium not found; open report/solution.html and print it to PDF.")
    pdf = ROOT / "report" / "solution.pdf"
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", out_html.as_uri()], check=True, capture_output=True)
    return pdf


if __name__ == "__main__":
    print(main(Path(sys.argv[1]).resolve()))
