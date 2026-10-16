// One experiment: the form to start one, its status, the question that waits, and its timeline.

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

const STATUS_TEXT: Record<ExperimentStatus, string> = {
  running: "Running",
  awaiting_input: "Waiting for you",
  completed: "Completed (finish guard passed)",
  stopped: "Stopped without a verified result",
  failed: "Failed",
  cancelled: "Cancelled",
};

export function ExperimentPanel(props: {
  view: ExperimentView | null;
  stream: StreamState;
  startBlocked: string | null;
  onStarted: (id: string) => void;
  onReply: (requestId: string, reply: string) => Promise<unknown>;
}) {
  const { view } = props;
  const s = view?.summary;
  const active = s && !TERMINAL.includes(s.status);

  return (
    <Panel
      title="Experiment"
      className={s?.mode === "simulated" ? "panel-sim" : undefined}
      aside={s && <StatusBadge status={s.status} stream={props.stream} />}
    >
      {!active && <QuestionForm blockedReason={props.startBlocked} onStarted={props.onStarted} />}
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
        </div>
      )}
      {s?.pending && (
        <InputRequestCard
          key={s.pending.request_id}
          request={s.pending}
          onReply={(reply) => props.onReply(s.pending!.request_id, reply)}
        />
      )}
      {view && <Timeline events={view.events} />}
    </Panel>
  );
}

function StatusBadge(props: { status: ExperimentStatus; stream: StreamState }) {
  const live = props.stream === "live";
  return (
    <>
      {props.stream === "reconnecting" && <Badge tone="warn">stream lost, reconnecting…</Badge>}
      <span className="status-pill">
        <Dot tone={STATUS_TONE[props.status]} pulse={live && props.status === "running"} />
        {STATUS_TEXT[props.status]}
      </span>
    </>
  );
}
