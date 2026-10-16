// Follow one experiment: load its snapshot, then apply the live events from the Server-Sent Events stream.
//
// Recovery: the snapshot (GET /api/experiments/{id}) has every event so far; the stream starts after the last seq.
// When the stream drops, EventSource reconnects by itself and sends Last-Event-ID, so the server resumes there.
// After a stream error, or a gap in the seq numbers, we reload the snapshot: it may also show that the experiment
// ended meanwhile (e.g. the backend restarted), and then the stream is closed instead of reconnecting forever.

import { useCallback, useEffect, useReducer, useState } from "react";

import { ApiError, cancelExperiment, eventsUrl, getExperiment, isAbort, parseEvent, replyExperiment } from "../api";
import { reduce } from "../experimentReducer";
import { type ExperimentEvent, TERMINAL } from "../types";

export type { ExperimentView } from "../experimentReducer";

export type StreamState = "idle" | "loading" | "live" | "reconnecting" | "ended" | "error";

const lastOf = (events: ExperimentEvent[]) => (events.length ? events[events.length - 1].seq : 0);

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
    let closed = false;
    let seen = 0; // the last seq applied

    const close = (state: StreamState) => {
      closed = true;
      source?.close();
      setStream(state);
    };

    /** Take the full snapshot; true if the experiment still runs. */
    const load = async (): Promise<boolean> => {
      const detail = await getExperiment(id, ctrl.signal);
      dispatch({ type: "reset", view: { summary: detail.experiment, events: detail.events } });
      seen = lastOf(detail.events);
      if (TERMINAL.includes(detail.experiment.status)) {
        close("ended");
        return false;
      }
      return true;
    };

    const open = () => {
      source = new EventSource(eventsUrl(id, seen));
      source.onopen = () => setStream("live");
      source.onerror = () => {
        if (closed) return;
        setStream("reconnecting");
        load().catch(() => {}); // backend away: EventSource keeps retrying by itself
      };
      source.onmessage = (msg) => {
        const event = parseEvent(msg.data);
        if (!event || event.experiment_id !== id || event.seq <= seen) return; // other experiment, or a replay
        if (event.seq > seen + 1) {
          load().catch(() => {}); // a gap: the snapshot has everything
          return;
        }
        seen = event.seq;
        dispatch({ type: "event", event });
        if (event.type === "experiment.status" && TERMINAL.includes(event.data.status)) close("ended");
      };
    };

    setStream("loading");
    load()
      .then((running) => running && !closed && open())
      .catch((e) => {
        if (isAbort(e)) return;
        setLoadError(e instanceof ApiError ? e : new ApiError("invalid", String(e)));
        setStream("error");
      });

    return () => {
      closed = true;
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
