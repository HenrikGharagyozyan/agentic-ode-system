// Every request to the backend goes through this file. Components never call fetch() themselves.

import type {
  CameraCheck,
  CameraStatus,
  ExperimentEvent,
  ExperimentListItem,
  ExperimentSummary,
  Health,
  RecordingPreview,
  WindowDelta,
} from "./types";

export const API_BASE = "/api";

/** offline: the backend did not answer; http: it answered with an error status; invalid: unexpected body. */
export type ApiErrorKind = "offline" | "http" | "invalid";

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;

  constructor(kind: ApiErrorKind, message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

export function isAbort(e: unknown): boolean {
  return e instanceof DOMException && e.name === "AbortError";
}

export function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** One request: JSON in, a checked JSON body out, or an ApiError. An abort is passed on unchanged. */
export async function request<T>(path: string, check: (body: unknown) => body is T, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API_BASE + path, { ...init, headers: { Accept: "application/json", ...init?.headers } });
  } catch (e) {
    if (isAbort(e)) throw e;
    throw new ApiError("offline", "The backend could not be reached.");
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null; // not JSON, e.g. an HTML error page
  }
  if (!res.ok) {
    // 502/503/504: the dev proxy (vite.config.ts) could not reach the backend
    if (res.status >= 502 && res.status <= 504) {
      throw new ApiError("offline", "The backend could not be reached.", res.status);
    }
    const detail = isObject(body) && typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`;
    throw new ApiError("http", detail, res.status);
  }
  if (!check(body)) throw new ApiError("invalid", "The backend sent an unexpected response.", res.status);
  return body;
}

// ---- checks of the response shapes (small on purpose: they catch a wrong server, not every field) ----

function isHealth(v: unknown): v is Health {
  return isObject(v) && v.ok === true && typeof v.version === "string";
}

function isCameraCheck(v: unknown): v is CameraCheck {
  return isObject(v) && typeof v.checked_at === "string" && typeof v.ok === "boolean";
}

function isCameraStatus(v: unknown): v is CameraStatus {
  return (
    isObject(v) &&
    (v.mode === "simulated" || v.mode === "real") &&
    typeof v.sdk_available === "boolean" &&
    typeof v.busy === "boolean" &&
    (v.last_check === null || isCameraCheck(v.last_check))
  );
}

// ---- the endpoints ----

export const getHealth = (signal?: AbortSignal) => request("/health", isHealth, { signal });

export const getCameraStatus = (signal?: AbortSignal) => request("/camera/status", isCameraStatus, { signal });

/** Opens the camera through MCP (~10 s on the real camera). Only ever called from a button click. */
export const checkCamera = (signal?: AbortSignal) =>
  request("/camera/check", isCameraCheck, { method: "POST", signal });

// ---- experiments ----

function isSummary(v: unknown): v is ExperimentSummary {
  return isObject(v) && typeof v.id === "string" && typeof v.status === "string" && typeof v.question === "string";
}

function isEvent(v: unknown): v is ExperimentEvent {
  return isObject(v) && typeof v.seq === "number" && typeof v.type === "string" && isObject(v.data);
}

export interface ExperimentDetail {
  experiment: ExperimentSummary;
  events: ExperimentEvent[];
}

function isDetail(v: unknown): v is ExperimentDetail {
  return isObject(v) && isSummary(v.experiment) && Array.isArray(v.events) && v.events.every(isEvent);
}

export interface ExperimentList {
  active: string | null;
  experiments: ExperimentListItem[];
}

function isList(v: unknown): v is ExperimentList {
  return isObject(v) && (v.active === null || typeof v.active === "string") && Array.isArray(v.experiments);
}

function isStarted(v: unknown): v is { id: string; status: string } {
  return isObject(v) && typeof v.id === "string";
}

function isOk(v: unknown): v is { ok: true } {
  return isObject(v) && v.ok === true;
}

const json = (method: string, body?: unknown, signal?: AbortSignal): RequestInit => ({
  method,
  signal,
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const startExperiment = (question: string) => request("/experiments", isStarted, json("POST", { question }));

export const listExperiments = (signal?: AbortSignal) => request("/experiments", isList, { signal });

export const getExperiment = (id: string, signal?: AbortSignal) =>
  request(`/experiments/${encodeURIComponent(id)}`, isDetail, { signal });

/** The answer to the graph's question. For an approval, reply is exactly "yes" or "no". */
export const replyExperiment = (id: string, requestId: string, reply: string) =>
  request(`/experiments/${encodeURIComponent(id)}/reply`, isOk, json("POST", { request_id: requestId, reply }));

export const cancelExperiment = (id: string) =>
  request(`/experiments/${encodeURIComponent(id)}/cancel`, isOk, json("POST"));

/** A run file to download (the backend allows only transcript.md/.json, experiment.json, events.jsonl). */
export const downloadUrl = (id: string, name: string) =>
  `${API_BASE}/experiments/${encodeURIComponent(id)}/download/${encodeURIComponent(name)}`;

/** The URL of the Server-Sent Events stream: the events after seq `after`, then live until the experiment ends. */
export const eventsUrl = (id: string, after: number) =>
  `${API_BASE}/experiments/${encodeURIComponent(id)}/events?after=${after}`;

export function parseEvent(text: string): ExperimentEvent | null {
  try {
    const v: unknown = JSON.parse(text);
    return isEvent(v) ? v : null;
  } catch {
    return null;
  }
}

function isPreview(v: unknown): v is RecordingPreview {
  return isObject(v) && typeof v.recording_id === "string" && isObject(v.image) && Array.isArray(v.hot_pixels);
}

function isWindowDelta(v: unknown): v is WindowDelta {
  return isObject(v) && typeof v.t0_ms === "number" && typeof v.t1_ms === "number" && isObject(v.delta_image);
}

/** The delta image of the window t0_ms .. t1_ms of a saved recording (the backend checks the range). */
export const getWindowDelta = (experimentId: string, recordingId: string, t0_ms: number, t1_ms: number, signal?: AbortSignal) =>
  request(
    `/experiments/${encodeURIComponent(experimentId)}/recordings/${encodeURIComponent(recordingId)}/preview?t0_ms=${t0_ms}&t1_ms=${t1_ms}`,
    isWindowDelta,
    { signal },
  );

/** Image and rate of a recording that this experiment saved (read from the file; never opens the camera). */
export const getRecordingPreview = (experimentId: string, recordingId: string, signal?: AbortSignal) =>
  request(
    `/experiments/${encodeURIComponent(experimentId)}/recordings/${encodeURIComponent(recordingId)}/preview`,
    isPreview,
    { signal },
  );
