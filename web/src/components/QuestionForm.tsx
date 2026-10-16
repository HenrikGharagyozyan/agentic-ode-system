// The research question and the Start button. Starting is refused (by the backend, too) while the camera is busy.

import { useState } from "react";

import { ApiError, startExperiment } from "../api";

// the four camera use cases of the project (tasks/uc1..uc4.md), word for word
const EXAMPLES: { label: string; text: string }[] = [
  {
    label: "Dark calibration",
    text:
      "The event camera is covered, so no light reaches the sensor. Find the defect pixels and mask them, " +
      "then measure the noise level of the camera and report it in events per second.",
  },
  {
    label: "Flicker frequency",
    text: "Something in front of the event camera flickers. Find out how fast it flickers and report the frequency in Hz.",
  },
  {
    label: "Bias tuning",
    text:
      "Something in front of the event camera flickers. Lower the noise of the camera with its biases, while the " +
      "flicker stays clearly visible. Report the bias values you chose, and the noise level and the flicker before " +
      "and after.",
  },
  {
    label: "Flicker or steady?",
    text:
      "Is the light in front of the event camera flickering or steady? Measure it and answer, with the measurement " +
      "that shows it.",
  },
];

export function QuestionForm(props: { blockedReason: string | null; onStarted: (id: string) => void }) {
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const blocked = props.blockedReason;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (sending || blocked || text.trim().length < 3) return; // one click = one experiment
    setSending(true);
    setError(null);
    try {
      props.onStarted((await startExperiment(text.trim())).id);
      setText("");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setSending(false);
    }
  };

  return (
    <form className="question" onSubmit={submit}>
      <label htmlFor="question" className="label">
        Research question
      </label>
      <textarea
        id="question"
        rows={3}
        maxLength={4000}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="e.g. Something in front of the event camera flickers. How fast?"
      />
      <div className="examples" aria-label="example questions">
        {EXAMPLES.map((ex) => (
          <button key={ex.label} type="button" className="chip" onClick={() => setText(ex.text)}>
            {ex.label}
          </button>
        ))}
      </div>
      <div className="actions">
        <button type="submit" className="btn btn-primary" disabled={sending || !!blocked || text.trim().length < 3}>
          {sending ? "Starting…" : "Start experiment"}
        </button>
        {blocked && <span className="small tone-text-warn">{blocked}</span>}
        {error && <span className="small tone-text-error">Not started: {error}</span>}
      </div>
    </form>
  );
}
