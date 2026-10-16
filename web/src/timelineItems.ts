// Pure: the events of an experiment -> the items the timeline shows (no React, so `node --test` can test it).
//
// A camera call is one item that changes in place: tool.pending (chosen) -> input.resolved (approved / refused)
// -> tool.result (the CameraRecord). Hardware turns are not items; their text is kept for the call's "Raw result".

import type { ExperimentEvent, ToolPending, ToolResult } from "./types.ts";

export type CallStatus = "waiting_approval" | "running" | "refused" | "not_run" | "succeeded" | "failed";

export interface CallItem {
  kind: "call";
  seq: number;
  time: string;
  pending: ToolPending | null; // null: a result without a pending event (should not happen; shown anyway)
  status: CallStatus;
  startedAt: string; // when it was chosen, or approved
  approved?: boolean; // the operator's answer, for a setting change
  result?: ToolResult;
  resultAt?: string;
}

export type Item = { kind: "event"; event: ExperimentEvent } | CallItem;

/** Pure: events -> what the timeline shows. */
export function timelineItems(events: ExperimentEvent[], ended: boolean): { items: Item[]; turns: Map<number, string> } {
  const items: Item[] = [];
  const turns = new Map<number, string>(); // hardware turn index -> its text
  let open: CallItem | null = null;
  for (const e of events) {
    switch (e.type) {
      case "tool.pending":
        open = {
          kind: "call",
          seq: e.seq,
          time: e.time,
          pending: e.data,
          status: e.data.needs_approval ? "waiting_approval" : "running",
          startedAt: e.time,
        };
        items.push(open);
        break;
      case "input.resolved":
        if (open && open.status === "waiting_approval") {
          open.status = e.data.accepted ? "running" : "refused";
          open.approved = !!e.data.accepted;
          open.startedAt = e.time;
          if (!e.data.accepted) open = null;
        }
        items.push({ kind: "event", event: e });
        break;
      case "tool.result": {
        const target: CallItem =
          open && open.pending?.name === e.data.tool
            ? open
            : { kind: "call", seq: e.seq, time: e.time, pending: null, status: "running", startedAt: e.time };
        if (target !== open) items.push(target);
        target.result = e.data;
        target.resultAt = e.time;
        target.status = e.data.ok ? "succeeded" : "failed";
        open = null;
        break;
      }
      case "turn.added":
        if (e.data.role === "hardware") turns.set(e.data.index, e.data.content);
        else items.push({ kind: "event", event: e });
        break;
      case "input.required":
        if (e.data.source !== "camera_setting") items.push({ kind: "event", event: e }); // the call card shows it
        break;
      case "experiment.status":
        if (["completed", "stopped", "failed", "cancelled"].includes(e.data.status)) items.push({ kind: "event", event: e });
        break;
    }
  }
  if (ended && open && (open.status === "running" || open.status === "waiting_approval")) open.status = "not_run";
  return { items, turns };
}
