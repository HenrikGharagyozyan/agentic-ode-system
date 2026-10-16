// Saved experiments (runs/<id>/experiment.json), newest first. Click one to open its timeline, data and report.

import { useEffect, useState } from "react";

import { ApiError, isAbort, listExperiments } from "../api";
import { dateTime } from "../format";
import { STATUS_TONE } from "../status";
import type { ExperimentListItem } from "../types";
import { Badge, Dot, Panel } from "./ui";

/** reloadKey: change it to load the list again (e.g. when the open experiment ends). */
export function HistoryPanel(props: { currentId: string | null; reloadKey: string; onOpen: (id: string) => void }) {
  const [items, setItems] = useState<ExperimentListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const ctrl = new AbortController();
    listExperiments(ctrl.signal).then(
      (list) => {
        setItems(list.experiments);
        setError(null);
      },
      (e) => !isAbort(e) && setError(e instanceof ApiError ? e.message : String(e)),
    );
    return () => ctrl.abort();
  }, [props.reloadKey, tick]);

  return (
    <Panel
      title="History"
      aside={
        <button type="button" className="btn btn-ghost btn-small" onClick={() => setTick((t) => t + 1)}>
          Reload
        </button>
      }
    >
      {error && <p className="small tone-text-error">Could not load the history: {error}</p>}
      {items === null && !error && <p className="small muted">Loading…</p>}
      {items?.length === 0 && <p className="small muted">No experiments yet.</p>}
      {items && items.length > 0 && (
        <ul className="history">
          {items.slice(0, 30).map((e) => (
            <li key={e.id}>
              <button
                type="button"
                className={`history-item${e.id === props.currentId ? " current" : ""}`}
                onClick={() => props.onOpen(e.id)}
              >
                <span className="history-top">
                  <Dot tone={STATUS_TONE[e.status] ?? "muted"} />
                  <span className="mono small">{dateTime(e.created)}</span>
                  {e.mode === "simulated" ? <Badge tone="sim">SIM</Badge> : <Badge tone="info">REAL</Badge>}
                  <span className="small muted">{e.status}</span>
                </span>
                <span className="history-q small">{e.question}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
