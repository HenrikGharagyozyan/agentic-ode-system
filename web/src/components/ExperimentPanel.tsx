// One experiment: the form to start one, its status, the question that waits, cancel, and its timeline.

import { useState } from "react";

import { ApiError } from "../api";
import type { ExperimentView, StreamState } from "../hooks/useExperiment";
import { type ExperimentStatus, TERMINAL } from "../types";
import { InputRequestCard } from "./InputRequestCard";
import { QuestionForm } from "./QuestionForm";
import { Timeline } from "./Timeline";
import { Badge, Dot, Notice, Panel, type Tone } from "./ui";

const STATUS_TONE: Record<ExperimentStatus, Tone> = {
  running: "info",
  awaiting_input: "warn",
  completed: "ok",
  stopped: "warn",
  failed: "error",
  cancelled: "muted",
};

export const STATUS_TEXT: Record<ExperimentStatus, string> = {
  running: "Running",
  awaiting_input: "Waiting for you",
  completed: "Completed: finish guard passed",
  stopped: "Stopped without a verified result",
  failed: "Failed",
  cancelled: "Cancelled",
};

export function ExperimentPanel(props: {
  view: ExperimentView | null;
  stream: StreamState;
  loadError: ApiError | null;
  startBlocked: string | null;
  onStarted: (id: string) => void;
  onNew: () => void;
  onReply: (requestId: string, reply: string) => Promise<unknown>;
  onCancel: () => Promise<unknown>;
}) {
  const { view } = props;
  const s = view?.summary;
  const ended = !!s && TERMINAL.includes(s.status);

  return (
    <Panel
      title="Experiment"
      className={s?.mode === "simulated" ? "panel-sim" : undefined}
      aside={
        s && (
          <>
            {props.stream === "reconnecting" && <Badge tone="warn">stream lost, reconnecting…</Badge>}
            <span className="status-pill">
              <Dot tone={STATUS_TONE[s.status]} pulse={props.stream === "live" && s.status === "running"} />
              {STATUS_TEXT[s.status]}
            </span>
            {ended ? (
              <button type="button" className="btn btn-small" onClick={props.onNew}>
                New experiment
              </button>
            ) : (
              <CancelButton onCancel={props.onCancel} />
            )}
          </>
        )
      }
    >
      {props.loadError && <Notice tone="error" title="Could not load the experiment">{props.loadError.message}</Notice>}
      {!s && !props.loadError && <QuestionForm blockedReason={props.startBlocked} onStarted={props.onStarted} />}
      {s && (
        <div className="exp-head">
          <div className="exp-question">{s.question}</div>
          <div className="exp-meta small muted mono">
            {s.id} · {s.mode === "simulated" ? "SIMULATED camera" : "real camera"} · model {s.model || "?"} ·{" "}
            {s.decisions} decision{s.decisions === 1 ? "" : "s"}
          </div>
          {s.status === "failed" && <Notice tone="error" title="The experiment failed">{s.error}</Notice>}
          {s.status === "stopped" && (
            <Notice tone="warn" title="Stopped without a verified result">
              {s.stop_reason}. The finish guard did not accept a measurement, so no result is claimed.
            </Notice>
          )}
          {s.status === "cancelled" && <Notice tone="info" title="Cancelled by the operator" />}
        </div>
      )}
      {s?.pending && !ended && (
        <InputRequestCard
          key={s.pending.request_id}
          request={s.pending}
          onReply={(reply) => props.onReply(s.pending!.request_id, reply)}
        />
      )}
      {view && <Timeline events={view.events} ended={ended} />}
    </Panel>
  );
}

function CancelButton(props: { onCancel: () => Promise<unknown> }) {
  const [state, setState] = useState<"idle" | "confirm" | "sent" | "error">("idle");
  const [error, setError] = useState("");
  if (state === "sent") return <span className="small muted">Cancel requested: stops after the current step…</span>;
  return (
    <>
      <button
        type="button"
        className={`btn btn-small ${state === "confirm" ? "btn-danger" : ""}`}
        onClick={() => {
          if (state !== "confirm") {
            setState("confirm"); // a second click confirms: no accidental cancel
            return;
          }
          props.onCancel().then(
            () => setState("sent"),
            (e) => {
              setError(e instanceof ApiError ? e.message : String(e));
              setState("error");
            },
          );
        }}
      >
        {state === "confirm" ? "Confirm cancel" : "Cancel"}
      </button>
      {state === "error" && <span className="small tone-text-error">{error}</span>}
    </>
  );
}
