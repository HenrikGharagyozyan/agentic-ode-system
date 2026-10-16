// The experiment as it happened: every event of the graph, in order. Nothing here is invented: each item is one
// event from the backend (a dialogue turn, a chosen camera call, its result, a question, a reply, a status).

import { useState } from "react";

import { clock } from "../format";
import type { ExperimentEvent, TurnData } from "../types";
import { Badge, type Tone } from "./ui";

const ROLE_TONE: Record<string, Tone> = {
  user: "muted",
  orchestrator: "info",
  scientist: "info",
  general: "info",
  executor: "info",
  hardware: "info",
  human: "warn",
  system: "warn",
};

export function Timeline(props: { events: ExperimentEvent[] }) {
  const items = props.events.filter((e) => e.type !== "experiment.status" || e.data.status !== "running");
  if (!items.length) return <p className="muted small">No events yet.</p>;
  return (
    <ol className="timeline">
      {items.map((e) => (
        <li key={e.seq} className={`tl-item tl-${e.type.replace(".", "-")}`}>
          <span className="tl-time mono">{clock(e.time)}</span>
          <div className="tl-body">
            <Item event={e} />
          </div>
        </li>
      ))}
    </ol>
  );
}

function Item(props: { event: ExperimentEvent }) {
  const e = props.event;
  switch (e.type) {
    case "turn.added":
      return <Turn turn={e.data} />;
    case "tool.pending":
      return (
        <div>
          <Badge tone="muted">{e.data.kind}</Badge>{" "}
          <span className="mono">
            {e.data.name}({JSON.stringify(e.data.args)})
          </span>{" "}
          <span className="muted small">
            {e.data.needs_approval ? "chosen: waits for approval" : "chosen: runs now"}
          </span>
        </div>
      );
    case "tool.result":
      return (
        <div>
          <Badge tone={e.data.ok ? "ok" : "error"}>{e.data.ok ? "tool ok" : "tool failed"}</Badge>{" "}
          <span className="mono">{e.data.tool}</span>
          {e.data.error && <span className="small tone-text-error"> {e.data.error}</span>}
          {e.data.warnings.map((w) => (
            <div key={w} className="small tone-text-warn">
              warning: {w}
            </div>
          ))}
        </div>
      );
    case "input.required":
      return (
        <div>
          <Badge tone="warn">waiting for you</Badge> <span className="small">{e.data.question}</span>
        </div>
      );
    case "input.resolved":
      return (
        <div className="small">
          Operator replied <span className="mono">“{e.data.reply}”</span>
          {e.data.accepted !== undefined && (e.data.accepted ? " — approved" : " — refused, the step does not run")}
        </div>
      );
    case "experiment.status":
      return (
        <div className="small">
          Status: <strong>{e.data.status}</strong>
          {e.data.stop_reason ? ` (${e.data.stop_reason})` : ""}
          {e.data.error && <span className="tone-text-error"> {e.data.error}</span>}
        </div>
      );
  }
}

function Turn(props: { turn: TurnData }) {
  const t = props.turn;
  const d = t.decision;
  const [open, setOpen] = useState(false);
  const long = t.content.length > 280;

  return (
    <div className="turn">
      <div className="turn-head">
        <Badge tone={ROLE_TONE[t.role] ?? "muted"}>{t.label}</Badge>
        {d?.next && (
          <span className="small">
            → <strong>{d.next}</strong>
            {d.kind && d.kind !== "none" ? ` (${d.kind})` : ""}
          </span>
        )}
        <span className="muted small mono">turn {t.index}</span>
      </div>
      {d ? (
        <div className="decision">
          {d.instruction && <p>{d.instruction}</p>}
          {d.reason && <p className="muted small">Why: {d.reason}</p>}
          {d.question && <p className="small">Question: {d.question}</p>}
        </div>
      ) : (
        <>
          <pre className={`turn-text${long && !open ? " clipped" : ""}`}>{t.content}</pre>
          {long && (
            <button type="button" className="btn btn-ghost btn-small" onClick={() => setOpen(!open)}>
              {open ? "Show less" : "Show all"}
            </button>
          )}
        </>
      )}
    </div>
  );
}
