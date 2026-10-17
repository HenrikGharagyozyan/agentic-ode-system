// Brightness change over the time of ONE saved recording, reconstructed from its events (each ON = +Δ, each
// OFF = −Δ, leaky integral). One series, one y axis in contrast steps Δ; a dashed line marks 0. Optional bands
// shade the edge windows: light red = rise windows, light blue = fall windows (the moments the edge images show).

import { useState } from "react";

import type { EdgeWindow } from "../../types";
import { short, ticks } from "./axes";

const W = 640;
const H = 212;
const M = { l: 52, r: 16, t: 14, b: 42 };

export function BrightnessPlot(props: { t: number[]; v: number[]; label: string; bands?: EdgeWindow[] }) {
  const { t, v } = props;
  const n = t.length;
  const [hover, setHover] = useState<number | null>(null);
  const tMax = t[n - 1] ?? 1;
  const lo = Math.min(0, ...v);
  const hi = Math.max(0, ...v);
  const pad = (hi - lo || 1) * 0.08;
  const [yLo, yHi] = [lo - pad, hi + pad];
  const x = (s: number) => M.l + (s / tMax) * (W - M.l - M.r);
  const y = (d: number) => H - M.b - ((d - yLo) / (yHi - yLo)) * (H - M.t - M.b);
  const path = v.map((d, i) => `${i ? "L" : "M"}${x(t[i]).toFixed(1)},${y(d).toFixed(1)}`).join("");

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const s = ((e.clientX - box.left) / box.width) * tMax;
    let best = 0; // the point nearest in time
    for (let i = 1; i < n; i++) if (Math.abs(t[i] - s) < Math.abs(t[best] - s)) best = i;
    setHover(best);
  };

  return (
    <figure className="figure">
      <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label={props.label}>
        {ticks(yLo, yHi, 4).map((d) => (
          <g key={d}>
            <line x1={M.l} x2={W - M.r} y1={y(d)} y2={y(d)} className="grid" />
            <text x={M.l - 6} y={y(d) + 4} className="tick" textAnchor="end">
              {short(d)}
            </text>
          </g>
        ))}
        {(props.bands ?? [])
          .filter((b) => b.t0_us / 1e6 < tMax)
          .map((b) => (
            <rect
              key={`${b.kind}${b.t0_us}`}
              x={x(b.t0_us / 1e6)}
              y={M.t}
              width={x(Math.min(b.t1_us / 1e6, tMax)) - x(b.t0_us / 1e6)}
              height={H - M.t - M.b}
              className={`band band-${b.kind}`}
            />
          ))}
        <line x1={M.l} x2={W - M.r} y1={y(0)} y2={y(0)} className="ref-line" />
        {ticks(0, tMax, 5).map((s) => (
          <text key={s} x={x(s)} y={H - M.b + 16} className="tick" textAnchor="middle">
            {s}
          </text>
        ))}
        <text x={(M.l + W - M.r) / 2} y={H - 4} className="tick" textAnchor="middle">
          time in the recording (s)
        </text>
        <text x={M.l - 6} y={M.t - 1} className="tick" textAnchor="end">
          ΔL (contrast steps Δ)
        </text>
        <path d={path} className="series brightness" />
        {hover !== null && (
          <g pointerEvents="none">
            <line x1={x(t[hover])} x2={x(t[hover])} y1={M.t} y2={H - M.b} className="crosshair" />
            <circle cx={x(t[hover])} cy={y(v[hover])} r={4} className="dot-brightness" />
          </g>
        )}
        <rect
          x={M.l}
          y={M.t}
          width={W - M.l - M.r}
          height={H - M.t - M.b}
          fill="transparent"
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        />
      </svg>
      <figcaption className="small muted mono">
        {hover !== null ? `t ${t[hover].toFixed(4)} s · ΔL ${v[hover].toFixed(4)} Δ` : "hover for values"}
      </figcaption>
    </figure>
  );
}
