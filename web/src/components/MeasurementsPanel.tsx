// Measurements of the current experiment, from its real tool results only: the recordings (record), their spectra
// (spectrum) and hot-pixel scans. The images, the rate plot and the brightness change (delta reconstruction) are
// read from the SAVED recording file, for every recording of every experiment.

import { useEffect, useState } from "react";

import { ApiError, getRecordingPreview, getWindowDelta, isAbort } from "../api";
import { brightnessCsv } from "../csv";
import { clock } from "../format";
import { type Measurement, measurements } from "../measurements";
import type { CameraMode, ExperimentEvent, Json, RecordingPreview, WindowDelta } from "../types";
import { BrightnessPlot } from "./charts/BrightnessPlot";
import { EventImage } from "./charts/EventImage";
import { DeltaImage, TimeImage } from "./charts/MapImages";
import { PeaksChart } from "./charts/PeaksChart";
import { RatePlot } from "./charts/RatePlot";
import { Badge, Notice, Panel, Spinner } from "./ui";

const num = (v: Json | undefined) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const show = (v: number | null, unit = "", digits = 4) =>
  v === null ? "Not measured" : `${Number(v.toPrecision(digits)).toLocaleString("en-US")}${unit}`;

export function MeasurementsPanel(props: { experimentId: string | null; mode: CameraMode | null; events: ExperimentEvent[] }) {
  const { recordings, hotScans } = measurements(props.events);
  const [picked, setPicked] = useState<string | null>(null);
  const current = recordings.find((r) => r.recordingId === picked) ?? recordings[recordings.length - 1] ?? null;

  useEffect(() => setPicked(null), [props.experimentId]); // a new experiment: follow its latest recording

  return (
    <Panel
      title="Measurements"
      className={props.mode === "simulated" ? "panel-sim" : undefined}
      aside={props.mode === "simulated" ? <Badge tone="sim">SIMULATED DATA</Badge> : props.mode === "real" ? <Badge tone="info">REAL CAMERA DATA</Badge> : null}
    >
      {!props.experimentId ? (
        <p className="muted small">Start or open an experiment. Its recordings and spectra appear here.</p>
      ) : !recordings.length ? (
        <p className="muted small">Not measured yet: this experiment has no successful recording.</p>
      ) : (
        <>
          {recordings.length > 1 && (
            <div className="examples" role="tablist" aria-label="recordings">
              {recordings.map((r, i) => (
                <button
                  key={r.recordingId}
                  type="button"
                  role="tab"
                  aria-selected={r === current}
                  className={`chip${r === current ? " chip-on" : ""}`}
                  onClick={() => setPicked(r.recordingId)}
                >
                  #{i + 1} {clock(r.time)}
                </button>
              ))}
            </div>
          )}
          {current && <RecordingBlock experimentId={props.experimentId} m={current} mode={props.mode} />}
        </>
      )}
      {hotScans.length > 0 && (
        <div className="subcard">
          <div className="subcard-head">
            <strong className="small">Hot-pixel scans (find_hot_pixels)</strong>
          </div>
          {hotScans.map((s) => (
            <div key={s.time} className="small mono">
              {clock(s.time)} · {s.seconds ?? "?"} s · {s.pixels.length} pixel(s) ≥ 20 kHz
              {s.pixels.length > 0 && `: ${s.pixels.slice(0, 4).map((p) => `(${p.x}, ${p.y})`).join(" ")}`}
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

function RecordingBlock(props: { experimentId: string; m: Measurement; mode: CameraMode | null }) {
  const { m } = props;
  const s = m.summary;
  const preview = usePreview(props.experimentId, m.recordingId);
  const spectrum = m.spectra[m.spectra.length - 1];
  const sp = spectrum?.result;
  const peaks = sp && Array.isArray(sp.peaks) ? sp.peaks : [];
  const bestSnr = peaks.reduce<number | null>((b, p) => {
    const v = typeof p === "object" && p && !Array.isArray(p) ? num(p.snr) : null;
    return v !== null && (b === null || v > b) ? v : b;
  }, null);

  return (
    <div className="recording">
      <div className="rec-title">
        <span className="mono">Recording {m.recordingId}</span>
        <span className="muted small">
          {show(num(s.seconds), " s", 3)} · result at {clock(m.time)} ·{" "}
          {preview.data?.fake === true ? "fake camera" : preview.data?.fake === false ? "physical camera" : props.mode}
        </span>
      </div>
      <div className="metrics">
        <Metric label="Events" value={show(num(s.n_events), "", 9)} />
        <Metric label="Event rate" value={show(num(s.rate_mev_s), " Mev/s")} />
        <Metric label="ON fraction" value={show(num(s.on_fraction), "", 3)} />
        <Metric label="Active pixels" value={show(num(s.active_pixels), "", 9)} />
        <Metric label="Fundamental" value={sp ? (sp.fundamental_hz === null ? "No peak" : show(num(sp.fundamental_hz), " Hz", 5)) : "Not measured"} />
        <Metric label="Best SNR" value={sp ? show(bestSnr, "", 3) : "Not measured"} />
      </div>
      {m.warnings.map((w) => (
        <Notice key={w} tone="warn" title="Recording warning">
          {w}
        </Notice>
      ))}
      {preview.loading && (
        <p className="small muted">
          <Spinner label="loading" /> Reading the saved recording…
        </p>
      )}
      {preview.error && <Notice tone="error" title="Recording file not readable">{preview.error}</Notice>}
      {preview.data && (
        <>
          <h3 className="h3">Event-count image (saved recording, not live)</h3>
          <EventImage preview={preview.data} />
          <h3 className="h3">Event rate</h3>
          {preview.data.rate ? <RatePlot rate={preview.data.rate} /> : <p className="small muted">No events: no rate.</p>}
          <DeltaReconstruction experimentId={props.experimentId} preview={preview.data} />
          <p className="small muted">
            Settings at recording time: {preview.data.settings?.masked_pixels ?? "?"} masked pixel(s), ROI{" "}
            {preview.data.settings?.roi ? preview.data.settings.roi.join(", ") : "none"}, biases{" "}
            {preview.data.settings?.biases
              ? Object.entries(preview.data.settings.biases).map(([k, v]) => `${k.replace("bias_", "")} ${v}`).join(" · ")
              : "?"}
          </p>
        </>
      )}
      <h3 className="h3">Spectrum peaks</h3>
      {sp ? (
        <>
          <PeaksChart result={sp} />
          {spectrum.warnings.map((w) => (
            <Notice key={w} tone="warn" title="Spectrum warning">
              {w}
            </Notice>
          ))}
        </>
      ) : (
        <p className="small muted">Not measured: no spectrum was computed for this recording.</p>
      )}
    </div>
  );
}

function DeltaReconstruction(props: { experimentId: string; preview: RecordingPreview }) {
  const { brightness_curve: curve, delta_image: delta, time_image: time, image, recording_id } = props.preview;
  if (curve === undefined) return null; // an older backend: it does not send the delta reconstruction yet
  if (!curve || !delta || !time) return <p className="small muted">No events: no brightness change, delta or time image.</p>;

  const edges = props.preview.edge_images;
  const windows = props.preview.edge_windows ?? [];
  const firstRise = windows.find((w) => w.kind === "rise");
  const lengthMs = (props.preview.recorded_s ?? 0) * 1000;
  const download = () => {
    const url = URL.createObjectURL(new Blob([brightnessCsv(curve.t_s, curve.delta_l)], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${recording_id}_brightness.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };
  const zoomMs = Math.round(curve.zoom.t_s.length * curve.bin_s * 1000);

  return (
    <>
      <div className="h3-row">
        <h3 className="h3">Brightness change (integrated events, saved recording, not live)</h3>
        <button type="button" className="btn btn-small" onClick={download}>
          Download brightness CSV
        </button>
      </div>
      <BrightnessPlot t={curve.t_s} v={curve.delta_l} label="brightness change over the whole recording" bands={windows} />
      <h4 className="small">First {zoomMs} ms at full resolution ({curve.bin_s * 1e6} µs bins)</h4>
      <BrightnessPlot t={curve.zoom.t_s} v={curve.zoom.delta_l} label={`brightness change, first ${zoomMs} ms`} bands={windows} />
      <p className="small muted">
        Reconstructed from events: each ON = +Δ, each OFF = −Δ; leaky integration τ = {curve.leak_tau_s} s; not an
        absolute brightness. Mean change per active pixel; hot pixels left out. Full plot: {curve.step_s * 1000} ms
        steps; the CSV holds these points (t_s, delta_L).
        {windows.length > 0 && (
          <>
            {" "}
            Shaded: <span className="legend-swatch rise" /> rise windows and <span className="legend-swatch fall" /> fall
            windows, the moments the edge images below show.
          </>
        )}
      </p>

      <h3 className="h3">Where did the light change (saved recording, not live)</h3>
      {edges ? (
        <>
          <p className="small muted">
            f0 = {edges.f0_hz} Hz ({edges.f0_source}, resolution {edges.resolution_hz} Hz)
            {edges.saved_f0_hz !== null && ` · saved spectrum: ${edges.saved_f0_hz} Hz`}. A delta image over a full period
            sums to about 0, so the rising and the falling edges are shown separately, each window 1/4 period wide.
          </p>
          {edges.f0_note && (
            <Notice tone="warn" title="Saved f0 is outdated">
              {edges.f0_note}
            </Notice>
          )}
          <div className="image-pair">
            <div>
              <h4 className="small">
                LED turns ON: where the light increased (Σ over {edges.n_rise} rising edges, window {edges.window_ms} ms)
              </h4>
              <DeltaImage
                image={image}
                delta={edges.rise}
                span={`${edges.n_rise} rising-edge windows of ${edges.window_ms} ms`}
                label="delta image summed over the rising edges"
              />
            </div>
            <div>
              <h4 className="small">
                LED turns OFF: where the light decreased (Σ over {edges.n_fall} falling edges, window {edges.window_ms} ms)
              </h4>
              <DeltaImage
                image={image}
                delta={edges.fall}
                span={`${edges.n_fall} falling-edge windows of ${edges.window_ms} ms`}
                label="delta image summed over the falling edges"
              />
            </div>
          </div>
        </>
      ) : (
        <Notice tone="info" title="No edge images">
          {props.preview.edge_images_reason ?? "The backend sent no edge images."}
        </Notice>
      )}

      {lengthMs > 0 && (
        <CustomWindow
          key={recording_id}
          experimentId={props.experimentId}
          recordingId={recording_id}
          image={image}
          lengthMs={lengthMs}
          start={firstRise ? firstRise.t0_us / 1000 : 0}
          width={firstRise ? (firstRise.t1_us - firstRise.t0_us) / 1000 : Math.min(10, lengthMs)}
        />
      )}

      <div className="image-pair">
        <div>
          <h3 className="h3">Delta image, whole recording (saved recording, not live)</h3>
          <DeltaImage image={image} delta={delta} />
        </div>
        <div>
          <h3 className="h3">Time-coloured event image (saved recording, not live)</h3>
          <TimeImage image={image} time={time} />
        </div>
      </div>
    </>
  );
}

/** A delta image of any window: start + width in ms (slider and number inputs). The backend checks the range;
 * a new request waits 300 ms after the last change and cancels the one before. */
function CustomWindow(props: {
  experimentId: string;
  recordingId: string;
  image: RecordingPreview["image"];
  lengthMs: number;
  start: number;
  width: number;
}) {
  const { lengthMs } = props;
  const round = (v: number) => Math.round(v * 1000) / 1000; // µs precision
  const [width, setWidth] = useState(round(props.width));
  const [start, setStart] = useState(round(props.start));
  const [state, setState] = useState<{ data: WindowDelta | null; error: string | null; loading: boolean }>({
    data: null,
    error: null,
    loading: true,
  });
  const w = Math.min(Math.max(width, 0.1), lengthMs);
  const t0 = Math.min(Math.max(start, 0), round(lengthMs - w));
  const t1 = Math.min(round(t0 + w), lengthMs);

  useEffect(() => {
    const ctrl = new AbortController();
    setState((s) => ({ ...s, loading: true }));
    const timer = setTimeout(() => {
      getWindowDelta(props.experimentId, props.recordingId, t0, t1, ctrl.signal).then(
        (data) => setState({ data, error: null, loading: false }),
        (e) => {
          if (isAbort(e)) return;
          setState({ data: null, error: e instanceof ApiError ? e.message : String(e), loading: false });
        },
      );
    }, 300);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [props.experimentId, props.recordingId, t0, t1]);

  const num = (e: React.ChangeEvent<HTMLInputElement>) => (Number.isFinite(e.target.valueAsNumber) ? e.target.valueAsNumber : 0);
  return (
    <div className="subcard">
      <div className="subcard-head">
        <strong className="small">Custom window (saved recording, not live)</strong>
        {state.loading && <Spinner label="loading" />}
      </div>
      <div className="window-control small">
        <input
          type="range"
          min={0}
          max={round(lengthMs - w)}
          step={0.1}
          value={t0}
          onChange={(e) => setStart(num(e))}
          aria-label="window start in ms"
        />
        <label>
          start ms
          <input className="input mono" type="number" min={0} step={0.1} value={t0} onChange={(e) => setStart(num(e))} />
        </label>
        <label>
          width ms
          <input className="input mono" type="number" min={0.1} step={0.1} value={w} onChange={(e) => setWidth(num(e))} />
        </label>
        <span className="muted mono">
          {t0}–{t1} ms of {round(lengthMs)} ms
        </span>
      </div>
      {state.error && <Notice tone="error" title="Window not available">{state.error}</Notice>}
      {state.data && (
        <DeltaImage
          image={props.image}
          delta={state.data.delta_image}
          span={`the window ${state.data.t0_ms}–${state.data.t1_ms} ms`}
          label="delta image of the custom window"
        />
      )}
    </div>
  );
}

function Metric(props: { label: string; value: string }) {
  const missing = props.value === "Not measured";
  return (
    <div className="metric">
      <div className="metric-label">{props.label}</div>
      <div className={`metric-value mono${missing ? " muted" : ""}`}>{props.value}</div>
    </div>
  );
}

function usePreview(experimentId: string, recordingId: string) {
  const [state, setState] = useState<{ loading: boolean; data: RecordingPreview | null; error: string | null }>({
    loading: true,
    data: null,
    error: null,
  });
  useEffect(() => {
    const ctrl = new AbortController();
    setState({ loading: true, data: null, error: null });
    getRecordingPreview(experimentId, recordingId, ctrl.signal).then(
      (data) => setState({ loading: false, data, error: null }),
      (e) => {
        if (isAbort(e)) return;
        setState({ loading: false, data: null, error: e instanceof ApiError ? e.message : String(e) });
      },
    );
    return () => ctrl.abort();
  }, [experimentId, recordingId]);
  return state;
}
