// Short, readable facts from a camera tool's JSON result, and what a setting change will do. Pure functions: every
// value shown comes from the result itself; a missing value is shown as missing, never as 0.

import type { Args, Json } from "./types.ts";

export interface Fact {
  label: string;
  value: string;
}

type Obj = Record<string, Json>;

const isObj = (v: Json | undefined): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const num = (v: Json | undefined): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const fmt = (v: number | null, unit = "", digits = 4) =>
  v === null ? "not reported" : `${Number(v.toPrecision(digits)).toLocaleString("en-US")}${unit}`;
const pixels = (v: Json | undefined) =>
  Array.isArray(v) ? v.map((p) => (Array.isArray(p) ? `(${p.join(", ")})` : isObj(p) ? `(${p.x}, ${p.y})` : "?")) : [];

export function toolFacts(tool: string, r: Obj | null): Fact[] {
  if (!r) return [];
  switch (tool) {
    case "record": {
      const s = isObj(r.summary) ? r.summary : {};
      const top = Array.isArray(s.top_pixels) ? s.top_pixels.filter(isObj) : [];
      return [
        { label: "recording", value: String(r.recording_id ?? "not reported") },
        { label: "duration", value: fmt(num(s.seconds), " s") },
        { label: "events", value: fmt(num(s.n_events), "", 9) },
        { label: "rate", value: fmt(num(s.rate_mev_s), " Mev/s") },
        { label: "ON fraction", value: fmt(num(s.on_fraction)) },
        { label: "active pixels", value: fmt(num(s.active_pixels), "", 9) },
        ...(top.length
          ? [{ label: "busiest pixel", value: `(${top[0].x}, ${top[0].y}) ${fmt(num(top[0].rate_hz), " ev/s")}` }]
          : []),
      ];
    }
    case "spectrum": {
      const peaks = Array.isArray(r.peaks) ? r.peaks.filter(isObj) : [];
      const best = peaks.length ? Math.max(...peaks.map((p) => num(p.snr) ?? 0)) : null;
      return [
        { label: "recording", value: String(r.recording_id ?? "not reported") },
        { label: "fundamental", value: r.fundamental_hz === null ? "no peak above the noise" : fmt(num(r.fundamental_hz), " Hz") },
        { label: "dominant peak", value: r.dominant_hz === null ? "none" : fmt(num(r.dominant_hz), " Hz") },
        { label: "peaks", value: String(peaks.length) },
        { label: "best SNR", value: best === null ? "—" : fmt(best, "", 3) },
        { label: "resolution", value: fmt(num(r.resolution_hz), " Hz") },
      ];
    }
    case "find_hot_pixels": {
      const hot = Array.isArray(r.hot_pixels) ? r.hot_pixels.filter(isObj) : [];
      return [
        { label: "hot pixels", value: String(hot.length) },
        { label: "listed", value: hot.slice(0, 4).map((h) => `(${h.x}, ${h.y}) ${fmt(num(h.rate_hz), " ev/s", 3)}`).join(" · ") || "none" },
        { label: "recording", value: `${fmt(num(r.seconds), " s")}, not saved` },
      ];
    }
    case "mask_pixels":
      return [
        { label: "newly masked", value: pixels(r.newly_masked).join(" ") || "none" },
        { label: "mask slots", value: `${r.slots_used ?? "?"} of ${r.slots_total ?? "?"} used` },
      ];
    case "set_bias":
      return [{ label: String(r.name ?? "bias"), value: `read back ${r.value ?? "?"}` }];
    case "set_roi":
      return [{ label: "ROI", value: Array.isArray(r.roi) ? `x, y, w, h = ${r.roi.join(", ")}` : "?" }];
    case "clear_roi":
      return [{ label: "ROI", value: "removed: the whole sensor sends events" }];
    case "clear_mask":
      return [{ label: "mask", value: "all pixels unmasked" }];
    case "camera_info":
      return [
        { label: "serial", value: String(r.serial ?? "?") },
        { label: "sensor", value: String(r.sensor ?? "?") },
        { label: "size", value: `${r.width ?? "?"} × ${r.height ?? "?"}` },
        { label: "source", value: r.fake === true ? "fake camera" : r.fake === false ? "physical camera" : "?" },
      ];
    case "get_settings": {
      const biases = isObj(r.biases) ? Object.entries(r.biases).map(([k, v]) => `${k.replace("bias_", "")} ${v}`) : [];
      return [
        { label: "biases", value: biases.join(" · ") || "?" },
        { label: "masked pixels", value: String(Array.isArray(r.mask) ? r.mask.length : "?") },
        { label: "ROI", value: Array.isArray(r.roi) ? r.roi.join(", ") : "none" },
      ];
    }
    default:
      return [];
  }
}

/** What a camera setting change will do, and what to watch out for (for the approval card). */
export function settingChange(name: string, args: Args): { what: string; risk: string } {
  switch (name) {
    case "mask_pixels": {
      const list = pixels(args.pixels);
      return {
        what: `Silence ${list.length} pixel(s): ${list.slice(0, 8).join(" ")}${list.length > 8 ? " …" : ""}`,
        risk: "Masked pixels send no events until the mask is cleared or the camera is reopened (64 slots).",
      };
    }
    case "clear_mask":
      return { what: "Unmask every masked pixel.", risk: "Hot pixels come back and can dominate the event rate." };
    case "set_bias":
      return {
        what: `Set ${String(args.name)} to ${String(args.value)} (an offset; 0 = default).`,
        risk: "Changes the sensitivity and noise of the whole sensor; a start-up burst follows a bias change.",
      };
    case "set_roi":
      return {
        what: `Only the window x=${args.x}, y=${args.y}, ${args.width} × ${args.height} sends events.`,
        risk: "Events outside the window are lost for every later recording until clear_roi.",
      };
    case "clear_roi":
      return { what: "Remove the ROI: the whole sensor sends events again.", risk: "The event rate can rise sharply." };
    default:
      return { what: `${name}(${JSON.stringify(args)})`, risk: "Unknown setting: read the call before approving." };
  }
}
