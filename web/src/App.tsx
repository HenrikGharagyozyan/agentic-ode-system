import { useEffect, useState } from "react";

import { listExperiments } from "./api";
import { CameraPanel } from "./components/CameraPanel";
import { ExperimentPanel } from "./components/ExperimentPanel";
import { Header } from "./components/Header";
import { Notice, Panel } from "./components/ui";
import { useBackendStatus } from "./hooks/useBackendStatus";
import { useExperiment } from "./hooks/useExperiment";

export default function App() {
  const status = useBackendStatus();
  const online = status.health !== null;
  const [currentId, setCurrentId] = useState<string | null>(null);
  const experiment = useExperiment(currentId);

  // after a reload, or when another browser starts one (the camera turns busy): follow the running experiment
  const busy = status.camera?.busy ?? false;
  useEffect(() => {
    if (!online || currentId) return;
    const ctrl = new AbortController();
    listExperiments(ctrl.signal)
      .then((list) => list.active && setCurrentId(list.active))
      .catch(() => {}); // offline: the banner already says so
    return () => ctrl.abort();
  }, [online, currentId, busy]);

  const camera = status.camera;
  const startBlocked = !online
    ? "Backend offline."
    : !camera
      ? "Camera status unknown."
      : camera.busy
        ? "The camera is busy (an experiment or a check is running)."
        : camera.mode === "real" && !camera.sdk_available
          ? "Real mode, but the camera SDK was not found: start the backend in the evcam shell."
          : null;

  return (
    <div className="app">
      <Header status={status} onRefresh={status.refresh} />
      {!online && !status.loading && (
        <Notice
          tone="error"
          title="Backend offline"
          action={
            <button type="button" className="btn" onClick={status.refresh}>
              Retry
            </button>
          }
        >
          {status.error?.kind === "offline"
            ? "The EventLab API at 127.0.0.1:8000 did not answer."
            : `The EventLab API answered with an error: ${status.error?.message ?? "unknown"}.`}{" "}
          Start it with <code>python -m agentic_ode.web</code>. No camera data is shown until it answers.
        </Notice>
      )}
      <main className="layout">
        <aside className="col-side">
          <CameraPanel camera={camera} online={online} onChanged={status.refresh} />
        </aside>
        <div className="col-main">
          <ExperimentPanel
            view={experiment.view}
            stream={experiment.stream}
            startBlocked={startBlocked}
            loadError={experiment.loadError}
            onStarted={(id) => {
              setCurrentId(id);
              status.refresh();
            }}
            onNew={() => setCurrentId(null)}
            onReply={experiment.reply}
            onCancel={experiment.cancel}
          />
          <Panel title="Recording view">
            <div className="placeholder">
              <p>No recording to show.</p>
              <p className="muted small">
                Event-count images and rates will be drawn from saved recordings (planned). This is not a live stream.
              </p>
            </div>
          </Panel>
        </div>
      </main>
    </div>
  );
}
