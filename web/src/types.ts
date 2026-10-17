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

// ---- experiments (agentic_ode/web/experiments.py, agentic_ode/web/events.py) ----

export type ExperimentStatus = "running" | "awaiting_input" | "completed" | "stopped" | "failed" | "cancelled";
export const TERMINAL: readonly ExperimentStatus[] = ["completed", "stopped", "failed", "cancelled"];

export type Json = string | number | boolean | null | Json[] | { [key: string]: Json };
export type Args = Record<string, Json>;

/** One real camera tool call of a hardware turn, as the graph recorded it (what the finish guard trusts). */
export interface CameraRecord {
  turn_index: number;
  tool: string;
  args: Args;
  ok: boolean;
  result_line: string;
}

/** The graph paused and waits for a person. camera_setting: hw_approve; orchestrator: the human node. */
export interface InputRequest {
  request_id: string;
  source: "camera_setting" | "orchestrator";
  kind: "hard" | "ambiguous" | "none";
  question: string;
  call?: { name: string; args: Args };
  proposed_step?: string | null;
}

export interface ExperimentSummary {
  id: string;
  question: string;
  mode: CameraMode;
  model: string;
  created: string;
  finished: string | null;
  status: ExperimentStatus;
  stop_reason: string | null;
  error: string | null;
  decisions: number;
  camera: CameraRecord[];
  refused_calls: { name: string; args: Args }[];
  pending: InputRequest | null;
}

export type ExperimentListItem = Pick<
  ExperimentSummary,
  "id" | "question" | "mode" | "created" | "finished" | "status" | "stop_reason" | "error" | "decisions"
>;

export interface Decision {
  next?: string;
  instruction?: string;
  reason?: string;
  kind?: string;
  proposed_next?: string;
  question?: string;
}

export interface TurnData {
  index: number;
  role: string;
  label: string;
  content: string;
  decision?: Decision;
}

export type ToolKind = "setting" | "measurement" | "read";

export interface ToolPending {
  name: string;
  args: Args;
  ignored: string[];
  kind: ToolKind;
  needs_approval: boolean;
}

export interface ToolResult {
  turn_index: number;
  tool: string;
  kind: ToolKind;
  args: Args;
  ok: boolean;
  result: Record<string, Json> | null; // the tool's JSON result; null if it was not JSON (see result_line)
  result_line: string | null;
  warnings: string[];
  error: string | null;
}

interface EventBase {
  seq: number;
  experiment_id: string;
  time: string;
}

export type ExperimentEvent = EventBase &
  (
    | {
        type: "experiment.status";
        data: { status: ExperimentStatus; stop_reason?: string; error?: string; finished?: string };
      }
    | { type: "turn.added"; data: TurnData }
    | { type: "tool.pending"; data: ToolPending }
    | { type: "tool.result"; data: ToolResult }
    | { type: "input.required"; data: InputRequest }
    | { type: "input.resolved"; data: { request_id: string; reply: string; accepted?: boolean } }
  );

// ---- plot data of a saved recording (agentic_ode/web/recordings.py) ----

export interface RecordingPreview {
  recording_id: string;
  created: string | null;
  note: string;
  fake: boolean | null;
  seconds_asked: number | null;
  recorded_s: number | null;
  n_events: number;
  settings: { masked_pixels?: number; biases?: Record<string, number>; roi?: number[] | null } | null;
  image: {
    width: number;
    height: number;
    bin_px: number;
    sensor_width: number;
    sensor_height: number;
    scale: string;
    on: { data: string; scale_top: number };
    off: { data: string; scale_top: number };
  };
  rate: { bin_s: number; on_hz: number[]; off_hz: number[] } | null;
  hot_pixels: { x: number; y: number; count: number; rate_hz: number }[];
  // delta reconstruction, in units of Δ (contrast steps); null when the recording has no events. Absent from an
  // older backend, so the panel checks them before use.
  brightness_curve?: {
    bin_s: number;
    step_s: number;
    leak_tau_s: number;
    unit: string;
    t_s: number[];
    delta_l: number[];
    zoom: { t_s: number[]; delta_l: number[] };
  } | null;
  delta_image?: { data: string; scale_top: number; min: number; max: number; scale: string } | null; // int8, signed
  time_image?: { data: string; seconds: number; scale: string } | null; // uint8: 0 = no events, 1..255 = start..end
  // "where did the light change": delta images summed over the rise and the fall windows of the first periods
  edge_images?: {
    f0_hz: number;
    f0_source: string;
    saved_f0_hz: number | null;
    f0_note: string | null; // set when a saved spectrum f0 differs (older analysis code)
    resolution_hz: number;
    window_ms: number;
    n_periods: number;
    n_rise: number;
    n_fall: number;
    rise: SignedImage;
    fall: SignedImage;
  };
  edge_images_reason?: string; // instead of edge_images: why there are none (e.g. no flicker frequency)
  edge_windows?: EdgeWindow[];
}

export type SignedImage = NonNullable<RecordingPreview["delta_image"]>;
export interface EdgeWindow {
  t0_us: number;
  t1_us: number;
  kind: "rise" | "fall";
}

/** GET .../preview?t0_ms=&t1_ms=: the delta image of one custom window of a saved recording. */
export interface WindowDelta {
  recording_id: string;
  t0_ms: number;
  t1_ms: number;
  delta_image: SignedImage;
}
