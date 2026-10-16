// Follow one experiment: load its snapshot, then apply the live events from the Server-Sent Events stream.
//
// Recovery: the snapshot (GET /api/experiments/{id}) has every event so far; the stream starts after the last seq.
// When the stream drops, EventSource reconnects by itself and sends Last-Event-ID, so the server resumes there.
// Events with a seq we already have are ignored, so nothing is shown twice.

import { useCallback, useEffect, useReducer, useState } from "react";

import { ApiError, cancelExperiment, eventsUrl, getExperiment, isAbort, parseEvent, replyExperiment } from "../api";
import { type ExperimentEvent, type ExperimentSummary, TERMINAL } from "../types";

export interface ExperimentView {
  summary: ExperimentSummary;
  events: ExperimentEvent[];
}

type Action = { type: "reset"; view: ExperimentView | null } | { type: "event"; event: ExperimentEvent };

/** Pure: the new view after one action. Exported for tests and for reading the rules in one place. */
export function reduce(view: ExperimentView | null, action: Action): ExperimentView | null {
  if (action.type === "reset") return action.view;
  if (view === null) return null;
  const e = action.event;
  const last = view.events.length ? view.events[view.events.length - 1].seq : 0;
  if (e.seq <= last) return view; // already have it (a replay after a reconnect)
  const summary = { ...view.summary };
  if (e.type === "experiment.status") Object.assign(summary, e.data);
  if (e.type === "input.required") summary.pending = e.data;
  if (e.type === "input.resolved") summary.pending = null;
  if (e.type === "turn.added" && e.data.role === "orchestrator") summary.decisions += 1;
  return { summary, events: [...view.events, e] };
}

export type StreamState = "idle" | "loading" | "live" | "reconnecting" | "ended" | "error";

export function useExperiment(id: string | null) {
  const [view, dispatch] = useReducer(reduce, null);
  const [stream, setStream] = useState<StreamState>("idle");
  const [loadError, setLoadError] = useState<ApiError | null>(null);

  useEffect(() => {
    dispatch({ type: "reset", view: null }); // clear the old experiment at once
    setLoadError(null);
    if (!id) {
      setStream("idle");
      return;
    }
    const ctrl = new AbortController();
    let source: EventSource | null = null;
    setStream("loading");

    getExperiment(id, ctrl.signal)
      .then((detail) => {
        dispatch({ type: "reset", view: { summary: detail.experiment, events: detail.events } });
        if (TERMINAL.includes(detail.experiment.status)) {
          setStream("ended");
          return;
        }
        const after = detail.events.length ? detail.events[detail.events.length - 1].seq : 0;
        source = new EventSource(eventsUrl(id, after));
        source.onopen = () => setStream("live");
        source.onerror = () => setStream(source?.readyState === EventSource.CLOSED ? "error" : "reconnecting");
        source.onmessage = (msg) => {
          const event = parseEvent(msg.data);
          if (!event || event.experiment_id !== id) return; // never mix experiments
          dispatch({ type: "event", event });
          if (event.type === "experiment.status" && TERMINAL.includes(event.data.status)) {
            source?.close(); // the server ends the stream too; do not let EventSource reconnect
            setStream("ended");
          }
        };
      })
      .catch((e) => {
        if (isAbort(e)) return;
        setLoadError(e instanceof ApiError ? e : new ApiError("invalid", String(e)));
        setStream("error");
      });

    return () => {
      ctrl.abort();
      source?.close();
    };
  }, [id]);

  const reply = useCallback(
    (requestId: string, text: string) => (id ? replyExperiment(id, requestId, text) : Promise.reject()),
    [id],
  );
  const cancel = useCallback(() => (id ? cancelExperiment(id) : Promise.reject()), [id]);

  return { view: view && view.summary.id === id ? view : null, stream, loadError, reply, cancel };
}
