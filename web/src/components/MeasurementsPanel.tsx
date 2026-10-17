// Measurements of the current experiment, from its real tool results only: the recordings (record), their spectra
// (spectrum) and hot-pixel scans. The images, the rate plot and the brightness change (delta reconstruction) are
// read from the SAVED recording file, for every recording of every experiment.

import { useEffect, useState } from "react";

import { ApiError, getRecordingPreview, isAbort } from "../api";
import { brightnessCsv } from "../csv";
import { clock } from "../format";
import { type Measurement, measurements } from "../measurements";
import type { CameraMode, ExperimentEvent, Json, RecordingPreview } from "../types";
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
          <DeltaReconstruction preview={preview.data} />
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

function DeltaReconstruction(props: { preview: RecordingPreview }) {
  const { brightness_curve: curve, delta_image: delta, time_image: time, image, recording_id } = props.preview;
  if (curve === undefined) return null; // an older backend: it does not send the delta reconstruction yet
  if (!curve || !delta || !time) return <p className="small muted">No events: no brightness change, delta or time image.</p>;

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
      <BrightnessPlot t={curve.t_s} v={curve.delta_l} label="brightness change over the whole recording" />
      <h4 className="small">First {zoomMs} ms at full resolution ({curve.bin_s * 1e6} µs bins)</h4>
      <BrightnessPlot t={curve.zoom.t_s} v={curve.zoom.delta_l} label={`brightness change, first ${zoomMs} ms`} />
      <p className="small muted">
        Reconstructed from events: each ON = +Δ, each OFF = −Δ; leaky integration τ = {curve.leak_tau_s} s; not an
        absolute brightness. Mean change per active pixel; hot pixels left out. Full plot: {curve.step_s * 1000} ms
        steps; the CSV holds these points (t_s, delta_L).
      </p>
      <div className="image-pair">
        <div>
          <h3 className="h3">Delta image (saved recording, not live)</h3>
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
