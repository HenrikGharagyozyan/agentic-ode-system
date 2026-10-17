// LUMEN Dual-Vision: compare one saved Frame Lab and one saved Event Lab lumen.flicker.v1 result.
// RECORDED DATA ONLY: no camera is contacted. The two files are read in the browser only after "Compare recordings"
// and sent, as their exact bytes, to POST /api/dualvision/compare (dualVisionClient.ts). Everything shown below
// comes from that sanitized backend answer; nothing here computes, ranks or synchronises anything.

import { type ChangeEvent, useEffect, useRef, useState } from "react";

import {
  checkFileSize,
  compareRecordings,
  DualVisionError,
  type DualVisionInput,
  type DualVisionResult,
  type Eligibility,
  type LabAssessment,
  SIDE_LABEL,
  type Side,
} from "../dualVisionClient";
import { num } from "../format";
import { Badge, Notice, Panel, Row, Spinner, type Tone } from "./ui";

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "done"; result: DualVisionResult }
  | { kind: "error"; error: DualVisionError };

const ELIGIBILITY: Record<Eligibility, { tone: Tone; text: string; title: string }> = {
  valid: { tone: "ok", text: "Eligible", title: "No quality reason was found in this record." },
  provisional: { tone: "warn", text: "Provisional", title: "Kept, but not good enough for a firm comparison." },
  not_recoverable: { tone: "error", text: "Not recoverable", title: "The frequency cannot be read from this record." },
  invalid: { tone: "error", text: "Invalid record", title: "The record is malformed; see the problems." },
};

const DECLARED: Record<DualVisionInput["acquisition"]["declared"], string> = {
  real_hardware: "Real hardware (as declared by the record)",
  not_real_hardware: "Not real hardware (as declared by the record)",
  not_declared: "Not declared by the record",
};

/** A measured value, or "Not measured": a missing number is never shown as 0. */
const measured = (v: number | null, unit: string, digits = 4) =>
  v === null ? "Not measured" : `${num(v, digits)} ${unit}`;

const asText = (v: string | boolean | null) => (typeof v === "string" && v ? v : "Not declared");

function requestText(a: LabAssessment["acquisition"]): string {
  if (a.request_mode === "frame_count") {
    return a.requested_frames === null ? "Frame count (not given)" : `${a.requested_frames} frames requested`;
  }
  if (a.request_mode === "duration") {
    return a.requested_s === null ? "Duration (not given)" : `${num(a.requested_s, 4)} s requested`;
  }
  return "Not declared";
}

function earlyStopText(a: LabAssessment["acquisition"]): string {
  if (a.early_stop === null) return "Unknown";
  if (a.early_stop) return "Yes: stopped before the request was met";
  return a.request_completed === true ? "No: every requested frame was acquired" : "No";
}

function f0Text(s: LabAssessment): string {
  if (s.f0_status === "measured") return measured(s.f0_hz, "Hz", 6);
  if (s.f0_status === "not_detected") return "Fundamental unresolved (f0 unavailable)";
  return "Not measured (invalid value)";
}

function errorText(s: LabAssessment): string {
  const e = s.frequency_error;
  if (e === null) return "Not computed";
  const within =
    e.within_resolution === null ? "" : e.within_resolution ? ", within resolution" : ", outside resolution";
  return `${num(e.abs_hz, 4)} Hz${within}`;
}

function LabCard(props: { side: Side; s: LabAssessment; input: DualVisionInput; fileName: string }) {
  const { s, input } = props;
  const a = s.acquisition;
  const el = ELIGIBILITY[s.eligibility];
  return (
    <section className="subcard dv-card" aria-label={SIDE_LABEL[props.side]}>
      <div className="subcard-head">
        <h3 className="h3">{SIDE_LABEL[props.side].toUpperCase()}</h3>
        <Badge tone={el.tone} title={el.title}>
          {el.text}
        </Badge>
      </div>
      <dl className="rows">
        <Row label="File (local)" mono>
          {props.fileName}
        </Row>
        <Row label="Input" mono>
          {input.bytes} bytes · sha256 {input.sha256.slice(0, 12)}…
        </Row>
        <Row label="Recording" mono>
          {s.recording_id ?? "Not declared"}
        </Row>
        <Row label="Instrument">
          {asText(s.hardware.model)}
          {typeof s.hardware.sensor === "string" && s.hardware.sensor ? ` · ${s.hardware.sensor}` : ""}
        </Row>
        <Row label="Acquisition">{DECLARED[input.acquisition.declared]}</Row>
      </dl>

      <h4 className="label">Frequency</h4>
      <dl className="rows">
        <Row label="Reference">
          {s.reference_hz === null ? "Not given" : `${num(s.reference_hz, 6)} Hz (operator-set)`}
        </Row>
        <Row label="Measured f0">{f0Text(s)}</Row>
        <Row label="|f0 − reference|">{errorText(s)}</Row>
        <Row label="Resolution">{measured(s.resolution_hz, "Hz")}</Row>
        {s.harmonic_relation && (
          <Row label="Harmonic hint">
            {s.harmonic_relation.relation} (k = {s.harmonic_relation.k}); f0 is not replaced
          </Row>
        )}
      </dl>

      <h4 className="label">Acquisition</h4>
      <dl className="rows">
        <Row label="Request">{requestText(a)}</Row>
        <Row label="Actual duration">{measured(a.actual_s, "s")}</Row>
        <Row label={a.sample_kind === "events" ? "Event count" : "Frame count"}>
          {a.samples === null ? "Not measured" : `${a.samples.toLocaleString("en-US")} ${a.sample_kind ?? ""}`}
        </Row>
        {props.side === "frame" && (
          <>
            <Row label="Frame rate">{measured(a.fps, "fps")}</Row>
            <Row label="Nyquist">{measured(a.nyquist_hz, "Hz")}</Row>
            <Row label="Exposure">{measured(a.exposure_us, "µs")}</Row>
          </>
        )}
        <Row label="Early stop">{earlyStopText(a)}</Row>
        <Row label="Analysis status">{s.acquisition_status ?? "Not declared"}</Row>
      </dl>

      {(s.reasons.length > 0 || s.problems.length > 0) && (
        <>
          <h4 className="label">Eligibility reasons</h4>
          <ul className="dv-list">
            {s.reasons.map((r) => (
              <li key={`r-${r}`} className="mono small">
                {r}
              </li>
            ))}
            {s.problems.map((p, i) => (
              <li key={`p-${i}`} className="small tone-text-error">
                {p}
              </li>
            ))}
          </ul>
        </>
      )}

      <h4 className="label">Warnings</h4>
      {s.warnings.length === 0 ? (
        <p className="hint">No warnings in the record.</p>
      ) : (
        <ul className="dv-list">
          {s.warnings.map((w, i) => (
            <li key={i} className="dv-warning">
              <span className="dv-warning-head">
                <span className="mono small">{w.code ?? "(no code)"}</span>
                <Badge tone={w.significant ? "warn" : "muted"}>{w.significant ? "significant" : "informational"}</Badge>
              </span>
              <span className="small wrap">{w.text}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

const PAIRING_TEXT = {
  paired: "Paired: both records carry the same explicit trial_id.",
  unpaired: "Unpaired: independent recordings, not known to be the same trial.",
  trial_mismatch: "Not paired: the trial_ids differ, so these are not the same trial.",
} as const;

const REFERENCE_TEXT = {
  match: "Same reference frequency",
  mismatch: "Different reference frequencies: no frequency errors are compared",
  unknown: "A reference frequency is missing or invalid",
} as const;

function Summary(props: { result: DualVisionResult; names: Record<Side, string> }) {
  const { comparison: c, inputs, presentation } = props.result;
  return (
    <>
      <section className="subcard">
        <div className="subcard-head">
          <h3 className="h3">Comparison</h3>
          {c.comparison_valid ? (
            <Badge tone="ok">Comparison valid</Badge>
          ) : (
            <Badge tone="warn">Comparison not valid</Badge>
          )}
        </div>
        <dl className="rows">
          <Row label="Display">Recorded data, not live. {presentation.note}</Row>
          <Row label="Reference">
            {REFERENCE_TEXT[c.reference_check]}
            {c.reference_hz !== null ? ` (${num(c.reference_hz, 6)} Hz)` : ""}
          </Row>
          <Row label="Trial pairing">{PAIRING_TEXT[c.pairing.status]}</Row>
          <Row label="SNR">Not compared: the two labs define SNR differently.</Row>
        </dl>
        {c.notes.length > 0 && (
          <ul className="limits small">
            {c.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        )}
      </section>
      <div className="dv-grid">
        <LabCard side="frame" s={c.frame} input={inputs.frame} fileName={props.names.frame} />
        <LabCard side="event" s={c.event} input={inputs.event} fileName={props.names.event} />
      </div>
      <section className="subcard">
        <h3 className="h3">Scientific limitations</h3>
        <ul className="limits small">
          {c.limitations.map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      </section>
    </>
  );
}

/** Stays enabled while a comparison runs: choosing another file cancels it (see pick()). */
function FilePicker(props: {
  side: Side;
  file: File | null;
  hint: string;
  onPick: (f: File | null) => void;
}) {
  const id = `dv-file-${props.side}`;
  return (
    <div className="dv-picker">
      <label className="label" htmlFor={id}>
        {SIDE_LABEL[props.side]} result
      </label>
      <input
        id={id}
        type="file"
        accept=".json,application/json"
        onChange={(e: ChangeEvent<HTMLInputElement>) => props.onPick(e.target.files?.[0] ?? null)}
      />
      <p className="hint">
        {props.file ? `${props.file.name} · ${props.file.size.toLocaleString("en-US")} bytes` : "No file selected."}
      </p>
      <p className="hint">{props.hint}</p>
    </div>
  );
}

async function readBytes(file: File): Promise<Uint8Array> {
  return new Uint8Array(await file.arrayBuffer());
}

export function DualVisionPanel() {
  const [files, setFiles] = useState<Record<Side, File | null>>({ frame: null, event: null });
  const [state, setState] = useState<State>({ kind: "idle" });
  const ctrl = useRef<AbortController | null>(null);

  useEffect(() => () => ctrl.current?.abort(), []); // unmount: drop the request

  const pick = (side: Side) => (file: File | null) => {
    ctrl.current?.abort(); // a new file makes the running request and the shown result stale
    ctrl.current = null;
    setFiles((f) => ({ ...f, [side]: file }));
    setState({ kind: "idle" });
  };

  const compare = async () => {
    const { frame, event } = files;
    if (!frame || !event) {
      const side: Side = !frame ? "frame" : "event";
      setState({ kind: "error", error: new DualVisionError("no_file", `Select a ${SIDE_LABEL[side]} file first.`) });
      return;
    }
    try {
      checkFileSize("frame", frame.size); // before reading: an oversized file is never read
      checkFileSize("event", event.size);
    } catch (e) {
      setState({ kind: "error", error: e as DualVisionError });
      return;
    }
    ctrl.current?.abort();
    const mine = new AbortController();
    ctrl.current = mine;
    setState({ kind: "loading" });
    try {
      const [frameBytes, eventBytes] = await Promise.all([readBytes(frame), readBytes(event)]);
      if (mine.signal.aborted) return;
      const result = await compareRecordings(frameBytes, eventBytes, { signal: mine.signal });
      if (ctrl.current === mine) setState({ kind: "done", result });
    } catch (e) {
      if (ctrl.current !== mine || mine.signal.aborted) return; // replaced or cancelled: say nothing
      const error = e instanceof DualVisionError ? e : new DualVisionError("server", "The files could not be read.");
      setState({ kind: "error", error });
    } finally {
      if (ctrl.current === mine) ctrl.current = null;
    }
  };

  const loading = state.kind === "loading";
  return (
    <Panel
      title="LUMEN Dual-Vision"
      className="dv-panel"
      aside={<Badge tone="info">RECORDED DATA — NOT LIVE</Badge>}
    >
      <p className="small muted">
        Compare one saved Frame Lab result with one saved Event Lab result. No camera is contacted; the two cameras
        were not synchronised. The files are read only when you press "Compare recordings".
      </p>
      <div className="dv-grid">
        <FilePicker
          side="frame"
          file={files.frame}
          onPick={pick("frame")}
          hint="Select a normalized lumen.flicker.v1 export, not the original frame-series diagnostic JSON."
        />
        <FilePicker
          side="event"
          file={files.event}
          onPick={pick("event")}
          hint="Select an Event Lab lumen.flicker.v1 result (e.g. from report/lumen/)."
        />
      </div>
      <div className="actions">
        <button type="button" className="btn btn-primary" onClick={compare} disabled={loading}>
          {loading ? <Spinner label="comparing" /> : null}
          Compare recordings
        </button>
        <span className="hint">Each file at most 2 MiB. File names stay in this browser.</span>
      </div>
      {state.kind === "error" && (
        <Notice tone="error" title="No comparison">
          {state.error.message}
        </Notice>
      )}
      {state.kind === "done" && (
        <Summary result={state.result} names={{ frame: files.frame?.name ?? "", event: files.event?.name ?? "" }} />
      )}
    </Panel>
  );
}
