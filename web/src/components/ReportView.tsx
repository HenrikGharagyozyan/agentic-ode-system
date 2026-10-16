// The report of one experiment (built by report.ts from the saved record), with the run files to download.

import { downloadUrl } from "../api";
import { dateTime } from "../format";
import type { ExperimentView } from "../hooks/useExperiment";
import { buildReport } from "../report";
import { STATUS_TEXT } from "../status";
import { Badge, Notice } from "./ui";

const v = (x: number | null, unit = "", digits = 4) =>
  x === null ? "not measured" : `${Number(x.toPrecision(digits)).toLocaleString("en-US")}${unit}`;

export function ReportView(props: { view: ExperimentView }) {
  const s = props.view.summary;
  const r = buildReport(s, props.view.events);

  return (
    <article className="report">
      <header className="report-head">
        <h3 className="report-title">{r.title}</h3>
        <div className="report-badges">
          {r.simulated ? <Badge tone="sim">SIMULATED DATA</Badge> : <Badge tone="info">REAL CAMERA DATA</Badge>}
          {r.verified ? (
            <Badge tone="ok">verified by the finish guard</Badge>
          ) : (
            <Badge tone={r.finished ? "warn" : "muted"}>{r.finished ? "no verified result" : "in progress"}</Badge>
          )}
        </div>
        <dl className="rows">
          <Row label="Experiment">{s.id}</Row>
          <Row label="Status">{STATUS_TEXT[s.status]}{s.stop_reason ? ` (${s.stop_reason})` : ""}</Row>
          <Row label="Started / ended">
            {dateTime(s.created)} / {s.finished ? dateTime(s.finished) : "not ended"}
          </Row>
          <Row label="Model">{s.model || "not recorded"}</Row>
          <Row label="Decisions">{String(s.decisions)}</Row>
        </dl>
      </header>

      <Section n={1} title="Research question">
        <p>{r.question}</p>
      </Section>

      <Section n={2} title="Plan / hypothesis (scientist, model text)">
        {r.plan.length ? (
          r.plan.map((p, i) => (
            <pre key={i} className="turn-text">
              {p}
            </pre>
          ))
        ) : (
          <p className="muted">No scientist turn in this experiment.</p>
        )}
      </Section>

      <Section n={3} title="Procedure (camera calls through MCP, in order)">
        {r.procedure.length ? (
          <table className="table">
            <thead>
              <tr>
                <th>#</th>
                <th>Tool</th>
                <th>Arguments</th>
                <th>Approval</th>
                <th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {r.procedure.map((p) => (
                <tr key={p.n}>
                  <td className="mono">{p.n}</td>
                  <td className="mono">{p.tool}</td>
                  <td className="mono small">{JSON.stringify(p.args)}</td>
                  <td>{p.approved === null ? "not needed" : p.approved ? "approved" : "refused"}</td>
                  <td>{p.outcome}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">No camera call was made.</p>
        )}
      </Section>

      <Section n={4} title="Measurements (values exactly as the tools returned them)">
        {r.measurements.length ? (
          <table className="table">
            <thead>
              <tr>
                <th>Recording</th>
                <th>Duration</th>
                <th>Events</th>
                <th>Rate</th>
                <th>ON fraction</th>
                <th>Fundamental</th>
                <th>Best SNR</th>
              </tr>
            </thead>
            <tbody>
              {r.measurements.map((m) => (
                <tr key={m.recordingId}>
                  <td className="mono small">{m.recordingId}</td>
                  <td className="mono">{v(m.seconds, " s", 3)}</td>
                  <td className="mono">{v(m.events, "", 9)}</td>
                  <td className="mono">{v(m.rateMevS, " Mev/s")}</td>
                  <td className="mono">{v(m.onFraction, "", 3)}</td>
                  <td className="mono">
                    {m.fundamentalHz === "none"
                      ? "no peak"
                      : m.fundamentalHz === null
                        ? "no spectrum"
                        : `${v(m.fundamentalHz, " Hz", 5)} ± ${v((m.resolutionHz ?? 0) / 2, " Hz", 2)}`}
                  </td>
                  <td className="mono">{v(m.bestSnr, "", 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">No successful recording: nothing was measured.</p>
        )}
      </Section>

      <Section n={5} title="Conclusion (the orchestrator's final statement, model text)">
        {r.conclusion ? (
          <>
            <p>{r.conclusion.statement || "(no text)"}</p>
            {r.conclusion.reason && <p className="muted small">Reason given: {r.conclusion.reason}</p>}
            {!r.verified && (
              <Notice tone="warn" title="Not verified">
                The finish guard did not accept this conclusion; treat it as unconfirmed.
              </Notice>
            )}
          </>
        ) : (
          <p className="muted">No conclusion: the experiment did not finish.</p>
        )}
      </Section>

      <Section n={6} title="Limitations, warnings and uncertainty">
        <ul className="limits">
          {r.limitations.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </Section>

      <Section n={7} title="Provenance: the run files">
        <div className="actions">
          {["transcript.md", "experiment.json", "events.jsonl", "transcript.json"].map((name) => (
            <a key={name} className="btn btn-small" href={downloadUrl(s.id, name)} download>
              {name}
            </a>
          ))}
        </div>
        <p className="small muted">
          Saved in runs/{s.id}/. Recordings: runs/evcam/&lt;recording id&gt;.npz; every camera call is also in
          runs/evcam/tool_log.jsonl.
        </p>
      </Section>
    </article>
  );
}

function Section(props: { n: number; title: string; children: React.ReactNode }) {
  return (
    <section className="report-section">
      <h4>
        {props.n}. {props.title}
      </h4>
      {props.children}
    </section>
  );
}

function Row(props: { label: string; children: React.ReactNode }) {
  return (
    <div className="row">
      <dt>{props.label}</dt>
      <dd>{props.children}</dd>
    </div>
  );
}
