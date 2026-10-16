// The rules for applying experiment events, in one pure function (no React, no fetch), so it can be tested with
// `node --test` (web/tests/). The ".ts" import works in Vite, tsc (allowImportingTsExtensions) and Node.

import { type ExperimentEvent, type ExperimentSummary, TERMINAL } from "./types.ts";

export interface ExperimentView {
  summary: ExperimentSummary;
  events: ExperimentEvent[];
}

export type Action =
  | { type: "reset"; view: ExperimentView | null }
  | { type: "event"; event: ExperimentEvent };

export function lastSeq(view: ExperimentView | null): number {
  return view && view.events.length ? view.events[view.events.length - 1].seq : 0;
}

/** The new view after one action. An event is applied only if it is the next one (seq = last + 1): a replayed
 * event is ignored, and a gap (a lost event) is refused, so the caller reloads the snapshot instead. */
export function reduce(view: ExperimentView | null, action: Action): ExperimentView | null {
  if (action.type === "reset") return action.view;
  if (view === null) return null;
  const e = action.event;
  if (e.experiment_id !== view.summary.id || e.seq !== lastSeq(view) + 1) return view;
  const summary = { ...view.summary };
  if (e.type === "experiment.status") Object.assign(summary, e.data);
  if (e.type === "experiment.status" && TERMINAL.includes(e.data.status)) summary.pending = null;
  if (e.type === "input.required") summary.pending = e.data;
  if (e.type === "input.resolved") summary.pending = null;
  if (e.type === "turn.added" && e.data.role === "orchestrator") summary.decisions += 1;
  return { summary, events: [...view.events, e] };
}

export const isTerminal = (view: ExperimentView | null) => !!view && TERMINAL.includes(view.summary.status);
