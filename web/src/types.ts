// The shapes of the API responses (agentic_ode/web/app.py). Keep them in sync with the backend.

/** GET /api/health */
export interface Health {
  ok: boolean;
  version: string;
}

export type CameraMode = "simulated" | "real";

/** The camera_info result of a camera check (MCP tool camera_info, without "ok"). */
export interface CameraInfo {
  serial?: string;
  sensor?: string;
  width?: number;
  height?: number;
  temperature_c?: number;
  biases?: Record<string, number>;
  fake?: boolean;
}

/** POST /api/camera/check, and status.last_check. Exactly one of result / error is set. */
export interface CameraCheck {
  checked_at: string;
  ok: boolean;
  result?: CameraInfo;
  error?: string;
}

/** GET /api/camera/status: what the backend knows WITHOUT opening the camera. */
export interface CameraStatus {
  mode: CameraMode;
  sdk_available: boolean; // the SDK module was found; it does not mean the camera works
  busy: boolean;
  last_check: CameraCheck | null;
}
