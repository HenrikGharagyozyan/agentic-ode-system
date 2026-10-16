"""Build solution.pdf from a run's transcript.json. Every number in it comes from that file.

Needs Google Chrome or Chromium for the HTML -> PDF step.
"""

from __future__ import annotations

import base64
import html
import re
import shutil
import subprocess
from pathlib import Path

from agentic_ode.core.config import PROJECT_ROOT
from agentic_ode.core.dialogue import decision_field, role_label
from agentic_ode.core.metrics import last_rmse_line, rmse_lines
from agentic_ode.core.state import Role
from agentic_ode.reporting.diagram import graph_png
from agentic_ode.runtime.run_store import load_transcript

CHROMES = ("google-chrome", "chromium", "chromium-browser")


class ReportError(RuntimeError):
    pass


def repo_url() -> str:
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "<URL>"
    m = re.match(r"git@([^:]+):(.+?)(\.git)?$", url)
    return f"https://{m.group(1)}/{m.group(2)}" if m else url.removesuffix(".git")


def _clip(s: str, n: int = 220) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def describe(i: int, turn: dict, runs: list[dict]) -> tuple[str, str]:
    """(what the turn did, orchestrator's reason) derived from the transcript text and run records."""
    role, c = turn["role"], turn["content"]
    if role == Role.USER:
        return "Sends the Task 5 text (task5.md) to the orchestrator.", ""
    if role == Role.ORCHESTRATOR:
        return (f"Gives the turn to <b>{html.escape(decision_field(c, 'next'))}</b>: "
                f"{html.escape(_clip(decision_field(c, 'instruction')))}",
                html.escape(decision_field(c, "reason")))
    if role == Role.EXECUTOR:
        mine = [r for r in runs if r["turn_index"] == i]
        parts = [f"Wrote {html.escape(decision_field(c, 'Files written') or 'none')}."]
        for r in mine:
            code = "timeout" if r["exit_code"] is None else r["exit_code"]
            line = last_rmse_line(r["stdout"])
            res = f"printed <code>{html.escape(line)}</code>" if line else "printed no test RMSE"
            parts.append(f"Ran {html.escape(r['filename'])}: exit code {code}, {res}.")
        if not mine:
            parts.append("No script was run.")
        return " ".join(parts), ""
    return html.escape(_clip(c)), ""


def build_report(run_dir: Path, out_dir: Path) -> Path:
    """Write solution.html, graph.mmd and solution.pdf into out_dir; return the PDF path."""
    data = load_transcript(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    page, mermaid = render_html(run_dir, data)
    out_html = out_dir / "solution.html"
    out_html.write_text(page)
    (out_dir / "graph.mmd").write_text(mermaid)
    return _print_pdf(out_html, out_dir / "solution.pdf")


def render_html(run_dir: Path, data: dict) -> tuple[str, str]:
    dialogue, runs = data["dialogue"], data["runs"]

    good = [r for r in runs if r["exit_code"] == 0 and r["test_rmse"] is not None]
    final = good[-1] if good else (runs[-1] if runs else None)
    plan_idx = [i for i, t in enumerate(dialogue) if t["role"] == Role.SCIENTIST
                and (final is None or i < final["turn_index"])]
    plan = dialogue[plan_idx[-1]]["content"] if plan_idx else None

    png, mermaid = graph_png(run_dir)
    img = base64.b64encode(png).decode()

    rows = []
    for i, t in enumerate(dialogue):
        what, reason = describe(i, t, runs)
        rows.append(f"<tr><td>{i}</td><td>{role_label(t['role'])}</td><td>{what}</td><td>{reason}</td></tr>")

    if good:
        result = (f"<p>The script <code>{html.escape(final['filename'])}</code> ran with exit code 0 in executor turn "
                  f"{final['turn_index']} and printed:</p>"
                  f"<pre class='result'>{html.escape(chr(10).join(rmse_lines(final['stdout'])))}</pre>"
                  "<p>These lines are copied from the stdout stored in <code>transcript.json</code>; values are in volts. "
                  "For comparison, Task 5 states that predicting zero scores the RMS of the test output, about 0.06 V; "
                  "a test RMSE at or above that level means the free run did not track the measured output.</p>")
    else:
        result = "<p><b>The run did not reach an RMSE.</b> No executor run exited with code 0 and printed a test RMSE.</p>"

    script = ""
    if final:
        src = (run_dir / final["filename"]).read_text()
        status = ("the script that printed the RMSE above" if good
                  else "the last script the executor ran (it did not produce an RMSE)")
        script = (f"<h2>5. Final script</h2><p><code>{html.escape(final['filename'])}</code>, {status}.</p>"
                  f"<pre>{html.escape(src)}</pre>")
        plot = run_dir / "test_plot.png"
        if good and plot.exists() and final is runs[-1]:
            b = base64.b64encode(plot.read_bytes()).decode()
            script += ("<p>Plot saved by this script (measured vs free-run prediction, test segment):</p>"
                       f"<img class='plot' src='data:image/png;base64,{b}'>")

    plan_html = f"<pre class='plan'>{html.escape(plan)}</pre>" if plan else "<p>No scientist turn.</p>"
    run_rows = "".join(
        f"<tr><td>{r['turn_index']}</td><td>{html.escape(r['filename'])}</td>"
        f"<td>{'timeout' if r['exit_code'] is None else r['exit_code']}</td>"
        f"<td>{'' if r['train_rmse'] is None else repr(r['train_rmse'])}</td>"
        f"<td>{'' if r['test_rmse'] is None else repr(r['test_rmse'])}</td></tr>" for r in runs)

    page = _TEMPLATE.format(
        repo=html.escape(repo_url()),
        model=html.escape(data["model"]),
        run_dir=html.escape(data["run_dir"]),
        decisions=data["decisions"],
        stop_reason=html.escape(data["stop_reason"]),
        img=img,
        rows="".join(rows),
        result=result,
        run_rows=run_rows or "<tr><td colspan=5>no runs</td></tr>",
        plan_turn=plan_idx[-1] if plan_idx else "–",
        plan_html=plan_html,
        script=script,
    )
    return page, mermaid


def _print_pdf(html_file: Path, pdf: Path) -> Path:
    chrome = next(filter(None, map(shutil.which, CHROMES)), None)
    if chrome is None:
        raise ReportError(f"Chrome/Chromium not found; open {html_file} and print it to PDF.")
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html_file.as_uri()], check=True, capture_output=True)
    return pdf


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>Task 6 solution</title><style>
body{{font-family:Helvetica,Arial,sans-serif;font-size:10.5pt;line-height:1.4;margin:0;color:#111}}
h1{{font-size:18pt;margin:0 0 4px}} h2{{font-size:13pt;margin:22px 0 6px;border-bottom:1px solid #ccc}}
table{{border-collapse:collapse;width:100%;font-size:9pt}} td,th{{border:1px solid #bbb;padding:3px 5px;vertical-align:top;text-align:left}}
th{{background:#eee}} pre{{background:#f6f6f6;border:1px solid #ddd;padding:6px;font-size:7.6pt;white-space:pre-wrap;word-break:break-all}}
pre.result{{font-size:11pt;font-weight:bold}} code{{font-size:9pt}} img.graph{{width:80%;display:block;margin:auto}} img.plot{{width:100%}}
.meta td{{border:none;padding:1px 8px 1px 0}} pre.plan{{border-left:3px solid #2b5c8a;font-size:8pt}}
</style></head><body>
<h1>Task 6: an agentic system that solves Task 5</h1>
<p>Repository: {repo}</p>
<table class="meta"><tr><td>Model</td><td><code>{model}</code></td></tr>
<tr><td>Run</td><td><code>{run_dir}</code> (full dialogue in transcript.json / transcript.md)</td></tr>
<tr><td>Orchestrator decisions</td><td>{decisions} (limit 8)</td></tr>
<tr><td>Stop reason</td><td>{stop_reason}</td></tr></table>

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
<table><tr><th>#</th><th>role</th><th>what it did</th><th>orchestrator's reason</th></tr>{rows}</table>

<h2>3. Free-run test RMSE</h2>{result}
<p>All script runs of this session (values parsed from the printed lines, volts):</p>
<table><tr><th>executor turn</th><th>file</th><th>exit code</th><th>train RMSE</th><th>test RMSE</th></tr>{run_rows}</table>

<h2>4. The scientist's plan</h2>
<p>Scientist turn {plan_turn}, the latest plan before the final script, verbatim (raw text as the model wrote it).</p>
{plan_html}
{script}
</body></html>"""
