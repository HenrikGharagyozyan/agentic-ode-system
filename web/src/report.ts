// Pure: the report of one experiment, built ONLY from its saved record (experiment.json summary + its events, which
// carry the dialogue and the CameraRecords). Nothing is filled in: a missing section says that it is missing.

import { measurements } from "./measurements.ts";
import { timelineItems } from "./timelineItems.ts";
import { settingChange } from "./toolSummary.ts";
import { type Args, type ExperimentEvent, type ExperimentSummary, type Json, TERMINAL } from "./types.ts";

const num = (v: Json | undefined) => (typeof v === "number" && Number.isFinite(v) ? v : null);

export interface ProcedureStep {
  n: number;
  tool: string;
  args: Args;
  kind: string;
  outcome: string; // "ok", "failed: ...", "refused by the operator", "did not run (experiment ended)", ...
  approved: boolean | null; // setting changes: the operator's answer; null for tools without approval
}

export interface MeasurementRow {
  recordingId: string;
  seconds: number | null;
  events: number | null;
  rateMevS: number | null;
  onFraction: number | null;
  fundamentalHz: number | null | "none";
  bestSnr: number | null;
  resolutionHz: number | null;
}

export interface Report {
  title: string;
  question: string;
  simulated: boolean;
  verified: boolean; // the finish guard accepted a real measurement (status "completed")
  finished: boolean;
  plan: string[]; // the scientist's turns, word for word
  procedure: ProcedureStep[];
  measurements: MeasurementRow[];
  conclusion: { statement: string; reason: string } | null; // the orchestrator's final "finish" decision
  limitations: string[];
}

export function buildReport(summary: ExperimentSummary, events: ExperimentEvent[]): Report {
  const finished = TERMINAL.includes(summary.status);
  const turns = events.flatMap((e) => (e.type === "turn.added" ? [e.data] : []));
  const plan = turns.filter((t) => t.role === "scientist").map((t) => t.content);
  // the final statement: the last orchestrator decision that proposed to finish, directly (next: finish) or through
  // the operator's approval (next: human, proposed_next: finish), as the real model does
  const finish = [...turns]
    .reverse()
    .find((t) => t.role === "orchestrator" && (t.decision?.next === "finish" || t.decision?.proposed_next === "finish"));

  const procedure: ProcedureStep[] = [];
  for (const item of timelineItems(events, finished).items) {
    if (item.kind !== "call") continue;
    const tool = item.pending?.name ?? item.result?.tool ?? "?";
    const needsApproval = item.pending?.needs_approval ?? false;
    const approved = needsApproval ? (item.approved ?? null) : null; // null: no answer (yet)
    const outcome =
      item.status === "succeeded"
        ? "ok"
        : item.status === "failed"
          ? `failed: ${item.result?.error ?? "no error text"}`
          : item.status === "refused"
            ? "refused by the operator: it did not run"
            : item.status === "not_run"
              ? "did not run (the experiment ended)"
              : "still running";
    procedure.push({
      n: procedure.length + 1,
      tool,
      args: item.pending?.args ?? item.result?.args ?? {},
      kind: item.pending?.kind ?? item.result?.kind ?? "read",
      outcome,
      approved,
    });
  }

  const { recordings } = measurements(events);
  const rows: MeasurementRow[] = recordings.map((m) => {
    const sp = m.spectra[m.spectra.length - 1]?.result;
    const peaks = sp && Array.isArray(sp.peaks) ? sp.peaks : [];
    const snrs = peaks.map((p) => (typeof p === "object" && p && !Array.isArray(p) ? num(p.snr) : null));
    const best = snrs.filter((v): v is number => v !== null);
    return {
      recordingId: m.recordingId,
      seconds: num(m.summary.seconds),
      events: num(m.summary.n_events),
      rateMevS: num(m.summary.rate_mev_s),
      onFraction: num(m.summary.on_fraction),
      fundamentalHz: sp ? (sp.fundamental_hz === null ? "none" : num(sp.fundamental_hz)) : null,
      bestSnr: best.length ? Math.max(...best) : null,
      resolutionHz: sp ? num(sp.resolution_hz) : null,
    };
  });

  const limitations: string[] = [];
  if (summary.mode === "simulated") {
    limitations.push(
      "SIMULATED: every measurement comes from the fake camera (EVCAM_FAKE=1). It says nothing about the physical camera.",
    );
  }
  if (!finished) limitations.push("The experiment has not ended yet; this report is incomplete.");
  else if (summary.status !== "completed") {
    limitations.push(
      `No verified result: the experiment ${summary.status}` +
        (summary.stop_reason ? ` (${summary.stop_reason})` : "") +
        (summary.error ? `: ${summary.error}` : "") +
        ". The finish guard did not accept a measurement.",
    );
  }
  for (const step of procedure) {
    if (step.outcome.startsWith("refused")) {
      limitations.push(`The operator refused ${step.tool}: ${settingChange(step.tool, step.args).what}`);
    }
    if (step.outcome.startsWith("failed")) limitations.push(`${step.tool} ${step.outcome}`);
  }
  for (const t of turns) {
    const refusedStep = /^The human refused the step '([^']+)'/.exec(t.content);
    if (t.role === "system" && refusedStep) {
      limitations.push(`The operator refused the proposed step "${refusedStep[1]}"; it did not run.`);
    }
    if (t.role === "human" && !/^(yes|y|no|n)\b/i.test(t.content.trim())) {
      limitations.push(`The operator steered the model with a free-text answer: "${t.content.trim()}"`);
    }
  }
  const warnings = new Set(events.flatMap((e) => (e.type === "tool.result" ? e.data.warnings : [])));
  for (const w of warnings) limitations.push(`Tool warning: ${w}`);
  if (rows.some((r) => r.fundamentalHz !== null)) {
    limitations.push(
      "The spectrum tool reports only the peaks above its detection threshold, not the full spectrum; a frequency is " +
        "known to about ± half the resolution.",
    );
  }
  limitations.push("The camera has no lens: it measures changes of the total light, not an image of a scene.");
  limitations.push(
    "The plan and the final statement are text written by the language model. Check them against the measurements " +
      "and the procedure above.",
  );

  return {
    title: summary.question.length > 90 ? `${summary.question.slice(0, 89)}…` : summary.question,
    question: summary.question,
    simulated: summary.mode === "simulated",
    verified: summary.status === "completed",
    finished,
    plan,
    procedure,
    measurements: rows,
    conclusion: finish
      ? { statement: finish.decision?.instruction ?? "", reason: finish.decision?.reason ?? "" }
      : null,
    limitations,
  };
}
