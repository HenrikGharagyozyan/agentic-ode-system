// The camera: what the backend knows without opening it, and the explicit "Check camera" action.

import { useEffect, useState } from "react";

import { clock, dateTime } from "../format";
import { type CheckState, useCameraCheck } from "../hooks/useCameraCheck";
import type { CameraCheck, CameraStatus } from "../types";
import { Badge, Notice, Panel, Row, Spinner } from "./ui";

const INSTRUMENT = "IDS UE-39B0XCP-E-GL · Sony IMX636 · 1280 × 720 · no lens";

export function CameraPanel(props: { camera: CameraStatus | null; online: boolean; onChanged: () => void }) {
  const { camera, online } = props;
  const { state, check } = useCameraCheck(props.onChanged);
  const running = state.kind === "running";

  const button = (
    <button
      type="button"
      className="btn btn-primary"
      onClick={check}
      disabled={running || !online || camera?.busy === true}
      title="Opens the camera once through the MCP server and reads camera_info (about 10 s on the real camera)"
    >
      {running ? <Spinner label="checking" /> : null}
      {running ? "Checking…" : "Check camera"}
    </button>
  );

  return (
    <Panel
      title="Camera"
      className={camera?.mode === "simulated" ? "panel-sim" : undefined}
      aside={camera && <ModeBadge mode={camera.mode} />}
    >
      <dl className="rows">
        <Row label="Instrument">{INSTRUMENT}</Row>
        {!online || !camera ? (
          <Row label="State">Unknown: the backend is not reachable</Row>
        ) : (
          <>
            <Row label="Mode">
              {camera.mode === "simulated" ? "Simulated (EVCAM_FAKE=1): made-up events, no hardware" : "Real hardware"}
            </Row>
            <Row label="SDK (metavision_hal)">
              {camera.sdk_available ? (
                <Badge tone="ok">found</Badge>
              ) : camera.mode === "real" ? (
                <Badge tone="error" title="Start the backend in the evcam shell">not found</Badge>
              ) : (
                <Badge tone="muted">not found (not needed in simulation)</Badge>
              )}
            </Row>
            <Row label="Camera owner">
              {camera.busy ? <Badge tone="warn">busy: in use</Badge> : <Badge tone="muted">free</Badge>}
            </Row>
          </>
        )}
      </dl>
      {online && camera && <LastCheck check={camera.last_check} />}
      <div className="actions">
        {button}
        <CheckFeedback state={state} />
      </div>
      <p className="hint">
        The SDK being found does not mean the camera works. Only a check opens it (about 10 s on the real camera);
        status updates never do.
      </p>
    </Panel>
  );
}

function ModeBadge(props: { mode: CameraStatus["mode"] }) {
  return props.mode === "simulated" ? <Badge tone="sim">SIMULATED</Badge> : <Badge tone="info">REAL HARDWARE</Badge>;
}

function LastCheck(props: { check: CameraCheck | null }) {
  const c = props.check;
  if (!c) return <p className="muted small">Not checked since the backend started.</p>;
  if (!c.ok) {
    return (
      <Notice tone="error" title={`Last check failed · ${dateTime(c.checked_at)}`}>
        <span className="mono wrap">{c.error}</span>
      </Notice>
    );
  }
  const r = c.result ?? {};
  return (
    <div className="subcard">
      <div className="subcard-head">
        <Badge tone="ok">camera answered</Badge>
        <span className="muted small mono">{dateTime(c.checked_at)}</span>
      </div>
      <dl className="rows">
        <Row label="Serial" mono>{r.serial ?? "—"}</Row>
        <Row label="Sensor" mono>{r.sensor ?? "—"}</Row>
        <Row label="Size" mono>{r.width && r.height ? `${r.width} × ${r.height}` : "—"}</Row>
        <Row label="Temperature" mono>{r.temperature_c !== undefined ? `${r.temperature_c} (°C assumed)` : "—"}</Row>
        <Row label="Biases" mono>
          {r.biases ? Object.entries(r.biases).map(([k, v]) => `${k.replace("bias_", "")} ${v}`).join(" · ") : "—"}
        </Row>
        <Row label="Source">{r.fake ? <Badge tone="sim">fake camera</Badge> : <Badge tone="info">physical camera</Badge>}</Row>
      </dl>
    </div>
  );
}

function CheckFeedback(props: { state: CheckState }) {
  const s = props.state;
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (s.kind !== "running") return;
    const t = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(t);
  }, [s.kind]);

  switch (s.kind) {
    case "idle":
      return null;
    case "running":
      return <span className="muted small mono">{Math.max(0, (now - s.startedAt) / 1000).toFixed(0)} s</span>;
    case "done":
      return s.check.ok ? (
        <span className="small tone-text-ok">Camera answered at {clock(s.check.checked_at)}</span>
      ) : (
        <span className="small tone-text-error">Check ran, camera did not answer (see above)</span>
      );
    case "busy":
      return <span className="small tone-text-warn">Camera busy: {s.message}</span>;
    case "failed":
      return <span className="small tone-text-error">Check failed: {s.error.message}</span>;
  }
}
