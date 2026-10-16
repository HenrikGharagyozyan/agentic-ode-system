import type { BackendStatus } from "../hooks/useBackendStatus";
import { ConnectionStatus } from "./ConnectionStatus";
import { Badge } from "./ui";

export function Header(props: { status: BackendStatus; onRefresh: () => void }) {
  const mode = props.status.camera?.mode;
  return (
    <header className="app-header">
      <div className="brand">
        <svg className="brand-mark" viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="10" />
          <circle cx="7" cy="9" r="1.3" className="ev-on" />
          <circle cx="15" cy="7" r="1.3" className="ev-off" />
          <circle cx="11" cy="15" r="1.3" className="ev-on" />
          <circle cx="17" cy="14" r="1.3" className="ev-on" />
        </svg>
        <div>
          <h1>EventLab</h1>
          <p className="subtitle">Agentic Event Camera Research System</p>
        </div>
        <span className="tag">Research instrument</span>
      </div>
      <div className="header-right">
        {mode === "simulated" && <Badge tone="sim">Simulated camera</Badge>}
        {mode === "real" && <Badge tone="info">Real hardware mode</Badge>}
        <ConnectionStatus status={props.status} onRefresh={props.onRefresh} />
      </div>
    </header>
  );
}
