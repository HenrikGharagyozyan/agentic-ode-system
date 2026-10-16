// The DETECTED PEAKS of the spectrum tool, drawn as stems. The tool returns only its peaks and harmonics (not the
// full curve), so this chart shows exactly those: height = amplitude relative to the strongest peak.
// Dashed lines: the noise floor and the detection threshold (min_snr x noise floor), both from the tool's own SNR.

import type { Json } from "../../types";
import { ticks } from "./axes";

type Obj = Record<string, Json>;
const isObj = (v: Json | undefined): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const n = (v: Json | undefined) => (typeof v === "number" ? v : null);

const W = 640;
const H = 222;
const M = { l: 40, r: 16, t: 24, b: 42 };

export function PeaksChart(props: { result: Obj }) {
  const r = props.result;
  const fmin = n(r.fmin_hz) ?? 0;
  const fmax = n(r.fmax_hz) ?? 2000;
  const peaks = (Array.isArray(r.peaks) ? r.peaks.filter(isObj) : []).filter((p) => n(p.hz) !== null);
  const harmonics = (Array.isArray(r.harmonics) ? r.harmonics.filter(isObj) : []).filter((h) => n(h.hz) !== null);
  const f0 = n(r.fundamental_hz);
  const dominant = peaks.find((p) => n(p.rel_amp) === 1);
  const domSnr = dominant ? n(dominant.snr) : null;
  const floor = domSnr ? 1 / domSnr : null; // noise floor in units of the dominant peak
  const threshold = floor !== null && n(r.min_snr) !== null ? floor * n(r.min_snr)! : null;

  const x = (hz: number) => M.l + ((hz - fmin) / (fmax - fmin)) * (W - M.l - M.r);
  const y = (rel: number) => H - M.b - Math.min(rel, 1.05) / 1.05 * (H - M.t - M.b);

  if (!peaks.length) {
    return (
      <p className="small muted">
        No peak stood out of the noise (no peak above {n(r.min_snr) ?? "min_snr"} × noise floor between {fmin} and{" "}
        {fmax} Hz). This is a valid result: no flicker was detected in this band.
      </p>
    );
  }
  return (
    <figure className="figure">
      <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="detected spectrum peaks">
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className="grid" />
            <text x={M.l - 6} y={y(v) + 4} className="tick" textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        {ticks(fmin, fmax, 6).map((f) => (
          <text key={f} x={x(f)} y={H - M.b + 16} className="tick" textAnchor="middle">
            {f}
          </text>
        ))}
        <text x={(M.l + W - M.r) / 2} y={H - 4} className="tick" textAnchor="middle">
          frequency (Hz)
        </text>
        {floor !== null && (
          <line x1={M.l} x2={W - M.r} y1={y(floor)} y2={y(floor)} className="ref-line">
            <title>{`noise floor (median amplitude in the band) = 1/${domSnr} of the dominant peak`}</title>
          </line>
        )}
        {threshold !== null && threshold <= 1.05 && (
          <g>
            <line x1={M.l} x2={W - M.r} y1={y(threshold)} y2={y(threshold)} className="ref-line threshold" />
            <text x={W - M.r} y={y(threshold) - 4} className="tick" textAnchor="end">
              threshold (SNR {n(r.min_snr)})
            </text>
          </g>
        )}
        {harmonics.map((h) => (
          <circle key={`h${h.k}`} cx={x(n(h.hz)!)} cy={y(n(h.rel_amp) ?? 0)} r={5} className="harmonic">
            <title>{`harmonic ${h.k}×f0: ${h.hz} Hz, rel. amplitude ${h.rel_amp}, SNR ${h.snr ?? "—"}`}</title>
          </circle>
        ))}
        {peaks.map((p) => {
          const hz = n(p.hz)!;
          const isF0 = f0 !== null && Math.abs(hz - f0) < 1e-6;
          return (
            <g key={`p${hz}`} className={isF0 ? "stem fundamental" : "stem"}>
              <line x1={x(hz)} x2={x(hz)} y1={y(0)} y2={y(n(p.rel_amp) ?? 0)} />
              <circle cx={x(hz)} cy={y(n(p.rel_amp) ?? 0)} r={4} />
              <rect x={x(hz) - 8} y={M.t} width={16} height={H - M.t - M.b} fill="transparent">
                <title>{`peak ${hz} Hz · rel. amplitude ${p.rel_amp} · SNR ${p.snr ?? "—"}${isF0 ? " · fundamental" : ""}`}</title>
              </rect>
              {isF0 && (
                <text x={x(hz)} y={y(n(p.rel_amp) ?? 0) - 8} className="direct-label" textAnchor="middle">
                  f₀ {hz} Hz
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <figcaption className="small muted">
        Detected peaks only, not the full spectrum · height = amplitude relative to the strongest peak · ○ harmonics
        2f₀…5f₀ · dashed: noise floor and detection threshold · signal: ON − OFF events per {String(r.bin_us)} µs
      </figcaption>
    </figure>
  );
}
