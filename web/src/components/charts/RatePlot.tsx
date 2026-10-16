// Event rate over the time of ONE saved recording: ON and OFF events per second in time bins. One y axis (ev/s).

import { useState } from "react";

import type { RecordingPreview } from "../../types";
import { short, ticks } from "./axes";

const W = 640;
const H = 212;
const M = { l: 48, r: 44, t: 14, b: 42 };

export function RatePlot(props: { rate: NonNullable<RecordingPreview["rate"]> }) {
  const { on_hz, off_hz, bin_s } = props.rate;
  const n = on_hz.length;
  const [hover, setHover] = useState<number | null>(null);
  const tMax = n * bin_s;
  const yMax = Math.max(1, ...on_hz, ...off_hz) * 1.08;
  const x = (i: number) => M.l + ((i + 0.5) / n) * (W - M.l - M.r);
  const y = (v: number) => H - M.b - (v / yMax) * (H - M.t - M.b);
  // direct labels at the line ends; when the ends are closer than 13 px, push the labels apart
  const endOn = y(on_hz[n - 1]) + 4;
  const endOff = y(off_hz[n - 1]) + 4;
  const push = Math.max(0, 13 - Math.abs(endOn - endOff)) / 2;
  const onAbove = endOn <= endOff;
  const labelY = { on: onAbove ? endOn - push : endOn + push, off: onAbove ? endOff + push : endOff - push };
  const path = (vs: number[]) => vs.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - box.left) / box.width;
    setHover(Math.min(n - 1, Math.max(0, Math.floor(frac * n))));
  };

  return (
    <figure className="figure">
      <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="event rate over time, ON and OFF">
        {ticks(0, yMax, 4).map((v) => (
          <g key={v}>
            <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} className="grid" />
            <text x={M.l - 6} y={y(v) + 4} className="tick" textAnchor="end">
              {short(v)}
            </text>
          </g>
        ))}
        {ticks(0, tMax, 5).map((t) => (
          <text key={t} x={M.l + (t / tMax) * (W - M.l - M.r)} y={H - M.b + 16} className="tick" textAnchor="middle">
            {t}
          </text>
        ))}
        <text x={(M.l + W - M.r) / 2} y={H - 4} className="tick" textAnchor="middle">
          time in the recording (s)
        </text>
        <text x={M.l - 6} y={M.t - 1} className="tick" textAnchor="end">
          ev/s
        </text>
        <path d={path(on_hz)} className="series on" />
        <path d={path(off_hz)} className="series off" />
        <text x={x(n - 1) + 6} y={labelY.on} className="direct-label">
          ON
        </text>
        <text x={x(n - 1) + 6} y={labelY.off} className="direct-label">
          OFF
        </text>
        {hover !== null && (
          <g pointerEvents="none">
            <line x1={x(hover)} x2={x(hover)} y1={M.t} y2={H - M.b} className="crosshair" />
            <circle cx={x(hover)} cy={y(on_hz[hover])} r={4} className="dot-on" />
            <circle cx={x(hover)} cy={y(off_hz[hover])} r={4} className="dot-off" />
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
      <figcaption className="small muted">
        {hover !== null ? (
          <span className="mono">
            t {(hover * bin_s).toFixed(3)}–{((hover + 1) * bin_s).toFixed(3)} s · ON {short(on_hz[hover])} ev/s · OFF{" "}
            {short(off_hz[hover])} ev/s
          </span>
        ) : (
          <>
            <span className="legend-swatch on" /> ON <span className="legend-swatch off" /> OFF · events per second in{" "}
            {bin_s * 1000} ms bins (hover for values)
          </>
        )}
      </figcaption>
    </figure>
  );
}
