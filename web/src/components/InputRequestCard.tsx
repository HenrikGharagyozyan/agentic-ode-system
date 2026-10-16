// The graph waits for a person. Approve / Reject send exactly "yes" / "no" to the graph's own question; the graph
// (hw_approve or the human node) decides what happens. Other questions get a free-text answer.

import { useState } from "react";

import { ApiError } from "../api";
import type { InputRequest } from "../types";
import { Badge } from "./ui";

export function InputRequestCard(props: { request: InputRequest; onReply: (reply: string) => Promise<unknown> }) {
  const r = props.request;
  const [sending, setSending] = useState(false);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);

  const send = async (reply: string) => {
    if (sending) return;
    setSending(true);
    setError(null);
    try {
      await props.onReply(reply);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      setSending(false); // only on error: on success the card goes away with the next event
    }
  };

  const approval = r.kind === "hard";
  return (
    <div className="request" role="region" aria-label="the experiment waits for you">
      <div className="request-head">
        <Badge tone="warn">{r.source === "camera_setting" ? "Approval: camera setting change" : "Question to you"}</Badge>
        <span className="muted small mono">{r.request_id}</span>
      </div>
      {r.call && (
        <div className="call mono">
          <span className="call-name">{r.call.name}</span>
          <span className="call-args">{JSON.stringify(r.call.args)}</span>
        </div>
      )}
      <p className="request-question">{r.question || "(no question text)"}</p>
      {r.source === "orchestrator" && r.proposed_step && (
        <p className="small muted">After a yes, the graph runs: {r.proposed_step}</p>
      )}
      {approval ? (
        <div className="actions">
          <button type="button" className="btn btn-ok" disabled={sending} onClick={() => send("yes")}>
            Approve
          </button>
          <button type="button" className="btn btn-danger" disabled={sending} onClick={() => send("no")}>
            Reject
          </button>
          <span className="small muted">Applies to this one request only.</span>
        </div>
      ) : (
        <form
          className="actions"
          onSubmit={(e) => {
            e.preventDefault();
            void send(text.trim());
          }}
        >
          <input
            className="input"
            value={text}
            maxLength={2000}
            onChange={(e) => setText(e.target.value)}
            placeholder="Your answer"
            aria-label="your answer"
          />
          <button type="submit" className="btn btn-primary" disabled={sending || !text.trim()}>
            Send
          </button>
        </form>
      )}
      {error && <p className="small tone-text-error">Reply not accepted: {error}</p>}
    </div>
  );
}
