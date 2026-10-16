// One camera tool call (MCP), as the graph recorded it: tool, arguments, status, timing, facts, warnings, raw JSON.

import type { CallStatus } from "../timelineItems";
import { toolFacts } from "../toolSummary";
import type { Args, ToolKind, ToolResult } from "../types";
import { Badge, Spinner } from "./ui";

export type { CallStatus } from "../timelineItems";

const STATUS: Record<CallStatus, { tone: "warn" | "info" | "muted" | "ok" | "error"; text: string }> = {
  waiting_approval: { tone: "warn", text: "waiting for approval" },
  running: { tone: "info", text: "running" },
  refused: { tone: "muted", text: "refused, did not run" },
  not_run: { tone: "muted", text: "did not run (experiment ended)" },
  succeeded: { tone: "ok", text: "succeeded" },
  failed: { tone: "error", text: "failed" },
};

export const TOOL_TEXT: Record<string, string> = {
  camera_info: "Camera identity and state",
  get_settings: "Read biases, mask and ROI",
  set_bias: "Change one bias",
  find_hot_pixels: "Short recording, list pixels above 20 kHz",
  mask_pixels: "Silence pixels",
  clear_mask: "Unmask all pixels",
  set_roi: "Limit events to a window",
  clear_roi: "Remove the window",
  record: "Record events and save them",
  spectrum: "Flicker spectrum of a saved recording",
};

export function ToolCallCard(props: {
  name: string;
  kind: ToolKind;
  args: Args;
  status: CallStatus;
  result?: ToolResult;
  seconds?: number | null; // from choice (or approval) to result, measured by the backend's event times
  rawTurn?: string;
}) {
  const { result } = props;
  const st = STATUS[props.status];
  const facts = result ? toolFacts(props.name, result.result) : [];
  return (
    <div className={`tool tool-${props.status}`}>
      <div className="tool-head">
        <span className="mono tool-name">{props.name}</span>
        <Badge tone="muted">{props.kind}</Badge>
        <span className="muted small">{TOOL_TEXT[props.name] ?? "camera tool"}</span>
        <span className="tool-status">
          {props.status === "running" && <Spinner label="running" />}
          <Badge tone={st.tone}>{st.text}</Badge>
          {props.seconds != null && <span className="muted small mono">{props.seconds.toFixed(1)} s</span>}
        </span>
      </div>
      <div className="mono small tool-args">args {JSON.stringify(props.args)}</div>
      {facts.length > 0 && (
        <dl className="facts">
          {facts.map((f) => (
            <div key={f.label}>
              <dt>{f.label}</dt>
              <dd className="mono">{f.value}</dd>
            </div>
          ))}
        </dl>
      )}
      {result?.error && <div className="small tone-text-error mono wrap">{result.error}</div>}
      {result?.warnings.map((w) => (
        <div key={w} className="tool-warning small">
          <strong>warning</strong> {w}
        </div>
      ))}
      {(result || props.rawTurn) && (
        <details className="raw">
          <summary className="small muted">Raw result</summary>
          <pre className="mono">
            {result?.result ? JSON.stringify(result.result, null, 2) : (result?.result_line ?? props.rawTurn)}
          </pre>
        </details>
      )}
    </div>
  );
}
