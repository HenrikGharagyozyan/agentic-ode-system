// Pure: the measurements of an experiment, taken from its tool.result events (the CameraRecords). Each recording
// gets its record summary, its warnings and, if one was computed, the spectrum of that recording.

import type { ExperimentEvent, Json, ToolResult } from "./types.ts";

type Obj = Record<string, Json>;
const isObj = (v: Json | undefined): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);

export interface Measurement {
  recordingId: string;
  time: string; // when the record call's result arrived
  summary: Obj; // the record tool's "summary"
  warnings: string[];
  spectra: { result: Obj; warnings: string[]; time: string }[]; // every spectrum computed for this recording
}

export interface HotPixelScan {
  time: string;
  seconds: number | null;
  pixels: Obj[];
}

export function measurements(events: ExperimentEvent[]): { recordings: Measurement[]; hotScans: HotPixelScan[] } {
  const recordings: Measurement[] = [];
  const hotScans: HotPixelScan[] = [];
  const byId = new Map<string, Measurement>();
  for (const e of events) {
    if (e.type !== "tool.result" || !e.data.ok || !e.data.result) continue;
    const r: ToolResult = e.data;
    const result = r.result!;
    if (r.tool === "record" && typeof result.recording_id === "string") {
      const m: Measurement = {
        recordingId: result.recording_id,
        time: e.time,
        summary: isObj(result.summary) ? result.summary : {},
        warnings: r.warnings,
        spectra: [],
      };
      recordings.push(m);
      byId.set(m.recordingId, m);
    } else if (r.tool === "spectrum" && typeof result.recording_id === "string") {
      byId.get(result.recording_id)?.spectra.push({ result, warnings: r.warnings, time: e.time });
    } else if (r.tool === "find_hot_pixels") {
      hotScans.push({
        time: e.time,
        seconds: typeof result.seconds === "number" ? result.seconds : null,
        pixels: Array.isArray(result.hot_pixels) ? result.hot_pixels.filter(isObj) : [],
      });
    }
  }
  return { recordings, hotScans };
}
