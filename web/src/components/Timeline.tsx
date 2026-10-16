// The experiment as it happened. Nothing here is invented: each item comes from backend events.
//
// A camera call is one card that changes in place: tool.pending (chosen) -> input.resolved (approved / refused)
// -> tool.result (the CameraRecord). The hardware turn's raw text is kept inside the card ("Raw result").

import { useState } from "react";

import { clock } from "../format";
import { timelineItems } from "../timelineItems";
import type { ExperimentEvent, TurnData } from "../types";
import { ToolCallCard } from "./ToolCallCard";
import { Badge, type Tone } from "./ui";

const ROLE_TONE: Record<string, Tone> = {
  user: "muted",
  orchestrator: "info",
  scientist: "info",
  general: "info",
  executor: "info",
  human: "warn",
  system: "warn",
};

const seconds = (from: string, to?: string) =>
  to ? Math.max(0, (new Date(to).getTime() - new Date(from).getTime()) / 1000) : null;

export function Timeline(props: { events: ExperimentEvent[]; ended: boolean }) {
  const { items, turns } = timelineItems(props.events, props.ended);
  if (!items.length) return <p className="muted small">No events yet.</p>;
  return (
    <ol className="timeline">
      {items.map((it) => (
        <li key={it.kind === "call" ? `c${it.seq}` : it.event.seq} className="tl-item">
          <span className="tl-time mono">{clock(it.kind === "call" ? it.time : it.event.time)}</span>
          <div className="tl-body">
            {it.kind === "call" ? (
              <ToolCallCard
                name={it.pending?.name ?? it.result?.tool ?? "?"}
                kind={it.pending?.kind ?? it.result?.kind ?? "read"}
                args={it.pending?.args ?? it.result?.args ?? {}}
                status={it.status}
                result={it.result}
                seconds={seconds(it.startedAt, it.resultAt)}
                rawTurn={it.result ? turns.get(it.result.turn_index) : undefined}
              />
            ) : (
              <EventItem event={it.event} />
            )}
          </div>
        </li>
      ))}
    </ol>
  );
}

function EventItem(props: { event: ExperimentEvent }) {
  const e = props.event;
  switch (e.type) {
    case "turn.added":
      return <Turn turn={e.data} />;
    case "input.required":
      return (
        <div>
          <Badge tone="warn">question to you</Badge> <span className="small">{e.data.question}</span>
        </div>
      );
    case "input.resolved":
      return (
        <div className="small">
          Operator replied <span className="mono">“{e.data.reply}”</span>
          {e.data.accepted !== undefined && (e.data.accepted ? ": approved" : ": refused, the step does not run")}
        </div>
      );
    case "experiment.status":
      return (
        <div className="small">
          Experiment <strong>{e.data.status}</strong>
          {e.data.stop_reason ? `: ${e.data.stop_reason}` : ""}
          {e.data.error && <span className="tone-text-error"> {e.data.error}</span>}
        </div>
      );
    default:
      return null;
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
        <Badge tone={ROLE_TONE[t.role] ?? "muted"}>{t.role === "system" ? "graph note" : t.label}</Badge>
        {d?.next && (
          <span className="small">
            → <strong>{d.next === "hardware" ? "camera action" : d.next}</strong>
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
