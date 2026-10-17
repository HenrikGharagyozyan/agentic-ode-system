// Tests of the Dual-Vision client (DV-UI-02) with Node's own test runner: `npm test` in web/, or from the repo root
// `node --test web/tests/dualVisionClient.test.ts`. fetch is always a mock: nothing goes to the network.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

// fetch must not be touched while the module loads: any call before a test injects its own mock fails here
let importFetchCalls = 0;
const realFetch = globalThis.fetch;
globalThis.fetch = (async () => {
  importFetchCalls += 1;
  throw new Error("no network in tests");
}) as typeof fetch;
const client = await import("../src/dualVisionClient.ts");
const { COMPARE_PATH, DualVisionError, MAX_FILE_BYTES, bytesToBase64, compareRecordings, isDualVisionResult } = client;

const SOURCE = readFileSync(new URL("../src/dualVisionClient.ts", import.meta.url), "utf8");
const enc = new TextEncoder();

// The exact shape of agentic_ode.dualvision.ingest.compare_json_bytes() for two SYNTHETIC records (generated once
// from the real backend function; no hardware data).
const RESULT = {
  schema: "lumen.dualvision.ingest.v1",
  presentation: {
    mode: "recorded_data",
    live_session: false,
    note: "saved results shown later: no camera was contacted, and the two cameras were not synchronised",
  },
  inputs: {
    frame: { bytes: 768, sha256: "a4e5fc0dfdcc3113", acquisition: { declared: "real_hardware", basis: "hardware.real as written in the record" } },
    event: { bytes: 614, sha256: "06e02da6e1f76edc", acquisition: { declared: "real_hardware", basis: "hardware.real as written in the record" } },
  },
  comparison: {
    schema: "lumen.dualvision.comparison.v1",
    reference_hz: 10.0,
    reference_check: "match",
    pairing: { status: "unpaired", frame_trial_id: null, event_trial_id: null, basis: "explicit trial_id only" },
    frame: {
      lab: "frame", run_id: "F-RUN", recording_id: "F-REC", trial_id: null, local_time: "2026-10-16T12:00:00+04:00",
      hardware: { model: "SYNTH-FRAME", sensor: null, real: true }, source: { type: "LED_PWM", set_by: "manual" },
      reference_hz: 10.0, acquisition_status: "ok",
      acquisition: {
        request_mode: "frame_count", requested_s: null, requested_frames: 128, request_completed: true, actual_s: 0.78,
        samples: 128, sample_kind: "frames", fps: 164.0, exposure_us: 5000.0, early_stop: false, nyquist_hz: 82.0,
        expected_alias_hz: 10.0,
      },
      f0_hz: 10.05, f0_status: "measured", resolution_hz: 1.28,
      frequency_error: { signed_hz: 0.05000000000000071, abs_hz: 0.05000000000000071, relative: 0.005000000000000071, within_resolution: true },
      harmonic_relation: null, snr: { value: 100.0, definition: "spectral power ratio, Frame Lab DV-01" },
      eligibility: "provisional", reasons: ["SHORT_RECORDING", "SIGNIFICANT_WARNING"], problems: [],
      warnings: [
        { code: "HOST_TIMESTAMPS", text: "host arrival times", significant: true },
        { code: "ALIASING_NOT_EXCLUDED", text: "one frame rate", significant: false },
      ],
      analysis: "synthetic",
    },
    event: {
      lab: "event", run_id: "E-RUN", recording_id: "E-REC", trial_id: null, local_time: "2026-10-16T12:00:00+04:00",
      hardware: { model: "SYNTH-EVENT", sensor: "SYNTH", real: true }, source: { type: "LED_PWM", set_by: "manual" },
      reference_hz: 10.0, acquisition_status: "ok",
      acquisition: {
        request_mode: "duration", requested_s: 2.0, requested_frames: null, request_completed: null, actual_s: 1.5,
        samples: 1000, sample_kind: "events", fps: null, exposure_us: null, early_stop: true, nyquist_hz: null,
        expected_alias_hz: null,
      },
      f0_hz: 10.3, f0_status: "measured", resolution_hz: 0.65,
      frequency_error: { signed_hz: 0.3000000000000007, abs_hz: 0.3000000000000007, relative: 0.030000000000000072, within_resolution: true },
      harmonic_relation: null,
      snr: { value: 30.0, definition: "spectral amplitude ratio (peak amplitude / median amplitude), Event Lab" },
      eligibility: "provisional", reasons: ["EARLY_STOP", "SIGNIFICANT_WARNING"], problems: [],
      warnings: [{ code: null, text: "stopped early: synthetic cap", significant: true }],
      analysis: "synthetic",
    },
    error_evidence: "unpaired_evidence",
    snr_comparable: false,
    comparison_valid: false,
    notes: ["no explicit trial_id on both records: they are not known to be the same trial"],
    limitations: ["reference_hz is the generator setting entered by hand, not a calibrated optical measurement."],
  },
};

const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v)) as T;
const frameBytes = () => enc.encode(JSON.stringify({ schema: "lumen.flicker.v1", lab: "frame", note: "synthetic" }));
const eventBytes = () => enc.encode(JSON.stringify({ schema: "lumen.flicker.v1", lab: "event", note: "synthetic" }));

interface Call {
  url: string;
  init: RequestInit;
}

/** A fetch mock that records each call and answers with status and body (an object, or raw text). */
function mockFetch(status: number, body: unknown = RESULT, calls: Call[] = []) {
  const fn = (async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    const text = typeof body === "string" ? body : JSON.stringify(body);
    return new Response(text, { status, headers: { "Content-Type": "application/json" } });
  }) as unknown as typeof fetch;
  return { fn, calls };
}

async function rejection(p: Promise<unknown>): Promise<InstanceType<typeof DualVisionError>> {
  try {
    await p;
  } catch (e) {
    assert.ok(e instanceof DualVisionError, `expected a DualVisionError, got ${String(e)}`);
    return e;
  }
  assert.fail("expected a rejection");
}

test("importing the module makes no network request", () => {
  assert.equal(importFetchCalls, 0);
  globalThis.fetch = realFetch;
});

// ---- base64 ----

test("binary bytes are encoded as standard padded base64", () => {
  const cases: [number[], string][] = [
    [[], ""], [[0x00], "AA=="], [[0xff], "/w=="], [[0x00, 0xff], "AP8="], [[0xff, 0xff, 0xff], "////"],
    [[0xfb, 0xff], "+/8="], [[0x00, 0x00, 0x00, 0x00], "AAAAAA=="],
  ];
  for (const [bytes, expected] of cases) assert.equal(bytesToBase64(new Uint8Array(bytes)), expected);
  const all = Uint8Array.from({ length: 256 }, (_, i) => i);
  assert.equal(bytesToBase64(all), Buffer.from(all).toString("base64"));
});

test("UTF-8 JSON bytes survive the round trip unchanged", () => {
  const bytes = enc.encode('{"lab": "frame", "note": "µs · 10 Hz · ✓", "x": 1}\n');
  const b64 = bytesToBase64(bytes);
  assert.deepEqual(new Uint8Array(Buffer.from(b64, "base64")), bytes);
});

test("a 2 MiB array is encoded without a stack overflow and exactly", () => {
  const big = new Uint8Array(MAX_FILE_BYTES);
  for (let i = 0; i < big.length; i++) big[i] = (i * 7919) & 0xff;
  const b64 = bytesToBase64(big);
  assert.equal(b64.length, 4 * Math.ceil(MAX_FILE_BYTES / 3));
  assert.equal(b64, Buffer.from(big).toString("base64"));
});

// ---- the request ----

test("one POST to the same-origin endpoint with exactly the two base64 fields", async () => {
  const { fn, calls } = mockFetch(200);
  const f = frameBytes();
  const e = eventBytes();
  await compareRecordings(f, e, { fetch: fn });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "/api/dualvision/compare");
  assert.equal(COMPARE_PATH, "/api/dualvision/compare");
  assert.equal(calls[0].init.method, "POST");
  assert.equal(new Headers(calls[0].init.headers).get("Content-Type"), "application/json");
  const body = JSON.parse(String(calls[0].init.body));
  assert.deepEqual(Object.keys(body).sort(), ["event_b64", "frame_b64"]);
  assert.deepEqual(new Uint8Array(Buffer.from(body.frame_b64, "base64")), f);
  assert.deepEqual(new Uint8Array(Buffer.from(body.event_b64, "base64")), e);
});

test("no file name or path is sent", async () => {
  const { fn, calls } = mockFetch(200);
  await compareRecordings(frameBytes(), eventBytes(), { fetch: fn });
  const body = JSON.parse(String(calls[0].init.body)) as Record<string, unknown>;
  assert.deepEqual(Object.keys(body).sort(), ["event_b64", "frame_b64"]); // no name, path or url field
  for (const v of Object.values(body)) assert.match(String(v), /^[A-Za-z0-9+/]*={0,2}$/); // only the bytes
});

test("the input arrays are not changed", async () => {
  const f = frameBytes();
  const e = eventBytes();
  const before = [Uint8Array.from(f), Uint8Array.from(e)];
  await compareRecordings(f, e, { fetch: mockFetch(200).fn });
  assert.deepEqual([f, e], before);
});

test("exactly 2 MiB is sent; one byte more is refused before fetch is called", async () => {
  const { fn, calls } = mockFetch(200);
  await compareRecordings(new Uint8Array(MAX_FILE_BYTES).fill(32), eventBytes(), { fetch: fn });
  assert.equal(calls.length, 1);
  for (const [f, e, side] of [
    [new Uint8Array(MAX_FILE_BYTES + 1), eventBytes(), "frame"],
    [frameBytes(), new Uint8Array(MAX_FILE_BYTES + 1), "event"],
  ] as const) {
    const err = await rejection(compareRecordings(f, e, { fetch: fn }));
    assert.equal(err.kind, "too_large");
    assert.equal(err.side, side);
  }
  assert.equal(calls.length, 1); // never sent
});

test("an empty file is refused before fetch is called", async () => {
  const { fn, calls } = mockFetch(200);
  const err = await rejection(compareRecordings(new Uint8Array(0), eventBytes(), { fetch: fn }));
  assert.equal(err.kind, "empty");
  assert.equal(calls.length, 0);
});

// ---- the response ----

test("a valid response is returned unchanged, with no value computed by the client", async () => {
  const r = await compareRecordings(frameBytes(), eventBytes(), { fetch: mockFetch(200, RESULT).fn });
  assert.deepEqual(r, RESULT);
  assert.equal(r.comparison.frame.frequency_error?.abs_hz, RESULT.comparison.frame.frequency_error.abs_hz);
  assert.equal(r.comparison.frame.acquisition.request_mode, "frame_count");
  assert.equal(r.comparison.event.acquisition.request_mode, "duration");
});

test("wrong schemas and malformed shapes are refused", async () => {
  const bad: unknown[] = [];
  const set = (path: string[], value: unknown) => {
    const v = clone(RESULT) as Record<string, unknown>;
    let o = v;
    for (const k of path.slice(0, -1)) o = o[k] as Record<string, unknown>;
    o[path[path.length - 1]] = value;
    bad.push(v);
  };
  set(["schema"], "lumen.dualvision.ingest.v2");
  set(["comparison", "schema"], "other");
  set(["presentation", "live_session"], true); // never shown as live
  set(["presentation", "mode"], "live");
  set(["comparison", "snr_comparable"], true);
  set(["comparison", "comparison_valid"], "yes");
  set(["comparison", "pairing", "status"], "synchronised");
  set(["comparison", "frame", "lab"], "event");
  set(["comparison", "frame", "f0_hz"], "10.05");
  set(["comparison", "frame", "eligibility"], "great");
  set(["comparison", "frame", "warnings"], [{ code: 1, text: "x", significant: true }]);
  set(["comparison", "event", "acquisition"], null);
  set(["comparison", "event", "frequency_error"], { abs_hz: 0.3 });
  set(["inputs", "frame", "acquisition", "declared"], "live");
  set(["comparison", "limitations"], [1, 2]);
  bad.push(null, [], "text", {});
  for (const body of bad) {
    assert.equal(isDualVisionResult(body), false);
    const err = await rejection(compareRecordings(frameBytes(), eventBytes(), { fetch: mockFetch(200, body).fn }));
    assert.equal(err.kind, "invalid_response");
  }
});

test("a response that is not JSON is refused", async () => {
  const err = await rejection(compareRecordings(frameBytes(), eventBytes(), { fetch: mockFetch(200, "<html>").fn }));
  assert.equal(err.kind, "invalid_response");
});

// ---- HTTP errors: fixed texts, never the server's or the file's content ----

const SECRET = "SECRET-MARKER-9876";

test("HTTP 400, 413 and 415 give their own kinds and fixed messages", async () => {
  for (const [status, kind] of [[400, "bad_request"], [413, "payload_too_large"], [415, "unsupported_media"]] as const) {
    const fn = mockFetch(status, { detail: `bad ${SECRET}` }).fn;
    const err = await rejection(compareRecordings(frameBytes(), eventBytes(), { fetch: fn }));
    assert.equal(err.kind, kind);
    assert.equal(err.status, status);
    assert.ok(!err.message.includes(SECRET));
  }
});

test("HTTP 422 names the side and gives a clear lumen.flicker.v1 message", async () => {
  const fn = mockFetch(422, { detail: `frame input ${SECRET}`, side: "frame", code: "wrong_schema" }).fn;
  const err = await rejection(compareRecordings(frameBytes(), eventBytes(), { fetch: fn }));
  assert.equal(err.kind, "invalid_input");
  assert.equal(err.side, "frame");
  assert.equal(err.code, "wrong_schema");
  assert.match(err.message, /Frame Lab file is not a lumen\.flicker\.v1 record/);
  assert.match(err.message, /lumen\.frame_series_probe\.v1/);
  assert.ok(!err.message.includes(SECRET));
  const odd = mockFetch(422, { side: SECRET, code: `<${SECRET}>` }).fn; // unexpected fields are not shown
  const err2 = await rejection(compareRecordings(frameBytes(), eventBytes(), { fetch: odd }));
  assert.equal(err2.side, null);
  assert.equal(err2.code, null);
  assert.ok(!err2.message.includes(SECRET));
});

test("a network failure, a proxy 502 and a server error", async () => {
  const down = (async () => {
    throw new TypeError("Failed to fetch");
  }) as unknown as typeof fetch;
  assert.equal((await rejection(compareRecordings(frameBytes(), eventBytes(), { fetch: down }))).kind, "offline");
  const proxy = mockFetch(502, { detail: "backend unreachable" }).fn;
  assert.equal((await rejection(compareRecordings(frameBytes(), eventBytes(), { fetch: proxy }))).kind, "offline");
  const err = await rejection(compareRecordings(frameBytes(), eventBytes(), { fetch: mockFetch(500, "Traceback").fn }));
  assert.equal(err.kind, "server");
  assert.ok(!err.message.includes("Traceback"));
});

test("a cancelled request is reported as aborted", async () => {
  const ctrl = new AbortController();
  const fn = (async (_url: string, init: RequestInit) => {
    ctrl.abort();
    assert.ok(init.signal?.aborted);
    throw new DOMException("The operation was aborted.", "AbortError");
  }) as unknown as typeof fetch;
  const err = await rejection(compareRecordings(frameBytes(), eventBytes(), { fetch: fn, signal: ctrl.signal }));
  assert.equal(err.kind, "aborted");
});

// ---- the source itself ----

test("the client has one endpoint, no other URL, and no React, file or hardware API", () => {
  assert.deepEqual(SOURCE.match(/"\/[^"]*"/g), ['"/api/dualvision/compare"']); // the only path string
  for (const word of ["http://", "https://", "ws://", "from \"react\"", "node:", "require(", "readFile", "camera",
    "console.", "localStorage", "XMLHttpRequest", "EventSource"]) {
    assert.ok(!SOURCE.includes(word), word);
  }
});

test("the client does no frequency science", () => {
  // it never subtracts, divides or rounds a measured value: errors, Nyquist and eligibility come from the backend
  const code = SOURCE.replace(/\/\/.*$/gm, ""); // without line comments
  for (const word of ["Math.", "f0_hz -", "- reference", "/ 2", "nyquist_hz =", "abs_hz =", "toFixed"]) {
    assert.ok(!code.includes(word), word);
  }
});
