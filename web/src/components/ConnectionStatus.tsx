// Backend connectivity only (GET /api/health). It says nothing about the camera.

import { clock } from "../format";
import type { BackendStatus } from "../hooks/useBackendStatus";
import { Dot } from "./ui";

export function ConnectionStatus(props: { status: BackendStatus; onRefresh: () => void }) {
  const { health, error, loading, updatedAt } = props.status;
  const [tone, label] = loading && !error ? (["muted", "Connecting…"] as const)
    : health ? (["ok", "Backend connected"] as const)
    : (["error", "Backend offline"] as const);

  return (
    <div className="conn" aria-live="polite">
      <Dot tone={tone} />
      <span className="conn-label">{label}</span>
      {health && <span className="conn-meta mono">v{health.version}</span>}
      {updatedAt && <span className="conn-meta mono" title="last successful status update">{clock(updatedAt)}</span>}
      <button type="button" className="btn btn-ghost btn-small" onClick={props.onRefresh}>
        Refresh
      </button>
    </div>
  );
}
