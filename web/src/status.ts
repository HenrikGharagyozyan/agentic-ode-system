// How each experiment status is shown.

import type { Tone } from "./components/ui";
import type { ExperimentStatus } from "./types";

export const STATUS_TONE: Record<ExperimentStatus, Tone> = {
  running: "info",
  awaiting_input: "warn",
  completed: "ok",
  stopped: "warn",
  failed: "error",
  cancelled: "muted",
};

export const STATUS_TEXT: Record<ExperimentStatus, string> = {
  running: "Running",
  awaiting_input: "Waiting for you",
  completed: "Completed: finish guard passed",
  stopped: "Stopped without a verified result",
  failed: "Failed",
  cancelled: "Cancelled",
};
