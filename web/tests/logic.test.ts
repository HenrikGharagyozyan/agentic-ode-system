// Tests of the pure frontend logic with Node's own test runner (no packages): `npm test` in web/.
// Node runs the TypeScript directly (it strips the types).

import assert from "node:assert/strict";
import { test } from "node:test";

import { reduce, type ExperimentView } from "../src/experimentReducer.ts";
import { timelineItems } from "../src/timelineItems.ts";
import { settingChange, toolFacts } from "../src/toolSummary.ts";
import type { ExperimentEvent, ExperimentSummary } from "../src/types.ts";

const summary = (over: Partial<ExperimentSummary> = {}): ExperimentSummary => ({
  id: "20261016-120000",
  question: "q",
  mode: "simulated",
  model: "fake",
  created: "2026-10-16T12:00:00+04:00",
  finished: null,
  status: "running",
  stop_reason: null,
  error: null,
  decisions: 0,
  camera: [],
  refused_calls: [],
  pending: null,
  ...over,
});

let seq = 0;
const ev = (type: string, data: object, time = "2026-10-16T12:00:00.000+04:00"): ExperimentEvent =>
  ({ seq: ++seq, experiment_id: "20261016-120000", time, type, data }) as ExperimentEvent;

const view = (events: ExperimentEvent[] = []): ExperimentView => ({ summary: summary(), events });

test("reduce applies the next event, ignores replays, refuses gaps and other experiments", () => {
  seq = 0;
  const a = ev("turn.added", { index: 0, role: "orchestrator", label: "orchestrator", content: "next: hardware" });
  const v1 = reduce(view(), { type: "event", event: a })!;
  assert.equal(v1.events.length, 1);
  assert.equal(v1.summary.decisions, 1);
  assert.equal(reduce(v1, { type: "event", event: a }), v1); // replay: same object
  const gap = { ...ev("turn.added", { index: 1, role: "scientist", label: "scientist", content: "x" }), seq: 5 };
  assert.equal(reduce(v1, { type: "event", event: gap }), v1);
  const other = { ...ev("turn.added", { index: 1, role: "scientist", label: "s", content: "x" }), experiment_id: "x" };
  assert.equal(reduce(v1, { type: "event", event: { ...other, seq: 2 } }), v1);
});

test("reduce tracks the pending question and the final status", () => {
  seq = 0;
  const req = { request_id: "r1", source: "camera_setting", kind: "hard", question: "ok?", call: { name: "mask_pixels", args: {} } };
  let v = reduce(view(), { type: "event", event: ev("input.required", req) })!;
  assert.equal(v.summary.pending?.request_id, "r1");
  v = reduce(v, { type: "event", event: ev("experiment.status", { status: "cancelled" }) })!;
  assert.equal(v.summary.status, "cancelled");
  assert.equal(v.summary.pending, null); // a finished experiment never shows an open question
});

test("a camera call is one card: chosen, approved, then its result and duration", () => {
  seq = 0;
  const events = [
    ev("tool.pending", { name: "mask_pixels", args: { pixels: [[1, 2]] }, ignored: [], kind: "setting", needs_approval: true }, "2026-10-16T12:00:00.000Z"),
    ev("input.required", { request_id: "r1", source: "camera_setting", kind: "hard", question: "?" }),
    ev("input.resolved", { request_id: "r1", reply: "yes", accepted: true }, "2026-10-16T12:00:05.000Z"),
    ev("turn.added", { index: 4, role: "hardware", label: "executor (hardware)", content: "camera call: ..." }),
    ev("tool.result", { turn_index: 4, tool: "mask_pixels", kind: "setting", args: {}, ok: true, result: { ok: true }, result_line: null, warnings: [], error: null }, "2026-10-16T12:00:05.400Z"),
  ];
  const { items, turns } = timelineItems(events, false);
  const calls = items.filter((i) => i.kind === "call");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].kind === "call" && calls[0].status, "succeeded");
  assert.equal(calls[0].kind === "call" && calls[0].startedAt, "2026-10-16T12:00:05.000Z"); // timed from the approval
  assert.equal(turns.get(4), "camera call: ...");
  assert.ok(!items.some((i) => i.kind === "event" && i.event.type === "input.required")); // the card shows it
});

test("a refused call and a call cut off by the end never look like they ran", () => {
  seq = 0;
  const pending = { name: "set_bias", args: { name: "bias_diff_off", value: 20 }, ignored: [], kind: "setting", needs_approval: true };
  const refused = timelineItems([ev("tool.pending", pending), ev("input.resolved", { request_id: "r", reply: "no", accepted: false })], false);
  assert.equal(refused.items[0].kind === "call" && refused.items[0].status, "refused");
  const cut = timelineItems([ev("tool.pending", { ...pending, name: "record", needs_approval: false })], true);
  assert.equal(cut.items[0].kind === "call" && cut.items[0].status, "not_run");
});

test("tool facts come from the result; missing values are said to be missing, never 0", () => {
  const facts = toolFacts("record", { recording_id: "20261016-120000-abcdef", summary: { seconds: 2, n_events: 12345 } });
  const get = (label: string) => facts.find((f) => f.label === label)?.value;
  assert.equal(get("recording"), "20261016-120000-abcdef");
  assert.equal(get("events"), "12,345");
  assert.equal(get("rate"), "not reported");
  const spec = toolFacts("spectrum", { fundamental_hz: null, dominant_hz: null, peaks: [] });
  assert.equal(spec.find((f) => f.label === "fundamental")?.value, "no peak above the noise");
  assert.deepEqual(toolFacts("record", null), []);
});

test("the approval card says what a setting change does", () => {
  assert.match(settingChange("mask_pixels", { pixels: [[1053, 704], [926, 151]] }).what, /Silence 2 pixel/);
  assert.match(settingChange("set_bias", { name: "bias_diff_off", value: 40 }).what, /bias_diff_off to 40/);
  assert.match(settingChange("unknown_tool", {}).risk, /Unknown setting/);
});

test("the report claims nothing the record does not hold", async () => {
  const { buildReport } = await import("../src/report.ts");
  seq = 0;
  const pending = { name: "mask_pixels", args: { pixels: [[1, 2]] }, ignored: [], kind: "setting", needs_approval: true };
  const events = [
    ev("turn.added", { index: 1, role: "scientist", label: "scientist", content: "Plan: record 2 s." }),
    ev("input.resolved", { request_id: "q0", reply: "no", accepted: false }), // a human-node answer, not a call
    ev("tool.pending", pending),
    ev("input.resolved", { request_id: "q1", reply: "no", accepted: false }),
    ev("experiment.status", { status: "stopped", stop_reason: "decision limit (20) reached" }),
  ];
  const r = buildReport(summary({ status: "stopped", stop_reason: "decision limit (20) reached" }), events);
  assert.equal(r.simulated, true);
  assert.equal(r.verified, false);
  assert.equal(r.conclusion, null); // no finish decision: no conclusion
  assert.deepEqual(r.measurements, []);
  assert.equal(r.procedure[0].approved, false);
  assert.match(r.procedure[0].outcome, /refused/);
  assert.ok(r.limitations.some((l) => l.startsWith("SIMULATED")));
  assert.ok(r.limitations.some((l) => l.includes("No verified result") && l.includes("decision limit")));
  assert.ok(r.limitations.some((l) => l.includes("The operator refused mask_pixels")));
  assert.deepEqual(r.plan, ["Plan: record 2 s."]);
});

test("a finish approved by the operator still gives the conclusion text", async () => {
  const { buildReport } = await import("../src/report.ts");
  seq = 0;
  const decision = { next: "human", instruction: "Report 240.0 Hz.", reason: "spectrum done", kind: "hard", proposed_next: "finish" };
  const events = [
    ev("turn.added", { index: 9, role: "orchestrator", label: "orchestrator", content: "...", decision }),
    ev("input.resolved", { request_id: "q4", reply: "yes", accepted: true }),
    ev("experiment.status", { status: "completed", stop_reason: "finish" }),
  ];
  const r = buildReport(summary({ status: "completed", stop_reason: "finish" }), events);
  assert.deepEqual(r.conclusion, { statement: "Report 240.0 Hz.", reason: "spectrum done" });
  assert.equal(r.verified, true);
});

test("the report says when the operator refused a step or steered the model", async () => {
  const { buildReport } = await import("../src/report.ts");
  seq = 0;
  const events = [
    ev("turn.added", { index: 3, role: "human", label: "human", content: "no" }),
    ev("turn.added", { index: 4, role: "system", label: "system", content: "The human refused the step 'executor'. It did NOT run." }),
    ev("turn.added", { index: 6, role: "human", label: "human", content: "Use the spectrum tool instead." }),
  ];
  const r = buildReport(summary(), events);
  assert.ok(r.limitations.includes('The operator refused the proposed step "executor"; it did not run.'));
  assert.ok(r.limitations.some((l) => l.includes('free-text answer: "Use the spectrum tool instead."')));
  assert.ok(!r.limitations.some((l) => l.includes('free-text answer: "no"')));
});
