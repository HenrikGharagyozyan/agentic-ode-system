// Event-count image of ONE saved recording: each canvas pixel is a 4 x 4 sensor-pixel bin; ON events in blue,
// OFF events in orange, log scale. Sensor coordinates: x 0..1279 left to right, y 0..719 top to bottom.
// Hot pixels (above 20 kHz) are ringed. This is a picture of a recording, not a live view.

import { useEffect, useRef } from "react";

import type { RecordingPreview } from "../../types";

function cssColor(name: string): [number, number, number] {
  const hex = getComputedStyle(document.documentElement).getPropertyValue(name).trim().replace("#", "");
  const n = parseInt(hex.length === 3 ? hex.replace(/./g, "$&$&") : hex, 16);
  return Number.isNaN(n) ? [255, 255, 255] : [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

const bytes = (b64: string) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

export function EventImage(props: { preview: RecordingPreview }) {
  const { image, hot_pixels } = props.preview;
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx) return;
    const on = bytes(image.on.data);
    const off = bytes(image.off.data);
    const [onC, offC] = [cssColor("--on"), cssColor("--off")];
    const bg = cssColor("--bg");
    const img = ctx.createImageData(image.width, image.height);
    for (let i = 0; i < on.length; i++) {
      const a = on[i] / 255;
      const b = off[i] / 255;
      for (let c = 0; c < 3; c++) img.data[i * 4 + c] = Math.min(255, bg[c] + a * onC[c] + b * offC[c]);
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [image]);

  const w = image.sensor_width;
  const h = image.sensor_height;
  return (
    <figure className="figure">
      <div className="event-image" style={{ aspectRatio: `${w} / ${h}` }}>
        <canvas ref={canvas} width={image.width} height={image.height} aria-label="event-count image" />
        <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
          {hot_pixels.map((p) => (
            <circle key={`${p.x},${p.y}`} cx={p.x} cy={p.y} r={14} className="hot-ring">
              <title>{`hot pixel (${p.x}, ${p.y}): ${Math.round(p.rate_hz).toLocaleString("en-US")} ev/s`}</title>
            </circle>
          ))}
        </svg>
      </div>
      <figcaption className="small muted">
        <span className="legend-swatch on" /> ON <span className="legend-swatch off" /> OFF · events per {image.bin_px}×
        {image.bin_px} px bin, log scale · x 0–{w - 1} →, y 0–{h - 1} ↓ (sensor coordinates)
        {hot_pixels.length > 0 && (
          <>
            {" "}
            · <span className="tone-text-warn">○ {hot_pixels.length} hot pixel(s) above 20 kHz</span>
          </>
        )}
      </figcaption>
    </figure>
  );
}
