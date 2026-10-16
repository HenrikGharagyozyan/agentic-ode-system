import { CameraPanel } from "./components/CameraPanel";
import { Header } from "./components/Header";
import { Notice, Panel } from "./components/ui";
import { useBackendStatus } from "./hooks/useBackendStatus";

export default function App() {
  const status = useBackendStatus();
  const online = status.health !== null;

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
          <CameraPanel camera={status.camera} online={online} onChanged={status.refresh} />
        </aside>
        <div className="col-main">
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
