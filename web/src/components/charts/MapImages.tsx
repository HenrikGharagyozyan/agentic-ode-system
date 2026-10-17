// Two colour-map images of ONE saved recording, on the same 4 x 4 px bins as the event-count image (sensor
// coordinates: x 0..1279 left to right, y 0..719 top to bottom). Hot pixels are already left out by the backend.
// - DeltaImage: Σ polarity per bin (each ON = +Δ, each OFF = −Δ). Blue = darker, white = 0, red = brighter.
// - TimeImage: the mean time of the events in each bin. Blue = start, red = end of the recording, black = no events.

import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";

import type { RecordingPreview } from "../../types";
import { bytes, cssColor } from "./EventImage";

type Rgb = [number, number, number];
const mix = (a: Rgb, b: Rgb, f: number): Rgb => [0, 1, 2].map((c) => a[c] + (b[c] - a[c]) * f) as Rgb;

/** A canvas of image.width x image.height bins; colour(i) colours bin i; describe(i) is the hover text. */
function MapImage(props: {
  image: RecordingPreview["image"];
  colour: (i: number) => Rgb;
  describe: (i: number) => string;
  label: string;
  legend: ReactNode;
  caption: ReactNode;
}) {
  const { image, colour } = props;
  const canvas = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(image.width, image.height);
    for (let i = 0; i < image.width * image.height; i++) {
      const rgb = colour(i);
      for (let c = 0; c < 3; c++) img.data[i * 4 + c] = rgb[c];
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [image, colour]);

  const onMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const col = Math.min(image.width - 1, Math.floor(((e.clientX - box.left) / box.width) * image.width));
    const row = Math.min(image.height - 1, Math.floor(((e.clientY - box.top) / box.height) * image.height));
    setHover(row * image.width + col);
  };

  const b = image.bin_px;
  return (
    <figure className="figure">
      <div
        className="event-image"
        style={{ aspectRatio: `${image.sensor_width} / ${image.sensor_height}` }}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        <canvas ref={canvas} width={image.width} height={image.height} aria-label={props.label} />
      </div>
      {props.legend}
      <figcaption className="small muted">
        {hover !== null ? (
          <span className="mono">
            x {(hover % image.width) * b}–{(hover % image.width) * b + b - 1}, y {Math.floor(hover / image.width) * b}–
            {Math.floor(hover / image.width) * b + b - 1} · {props.describe(hover)}
          </span>
        ) : (
          props.caption
        )}
      </figcaption>
    </figure>
  );
}

function Colourbar(props: { from: string; via?: string; to: string; left: string; mid?: string; right: string }) {
  const stops = [props.from, props.via, props.to].filter(Boolean).map((v) => `var(${v})`).join(", ");
  return (
    <div className="colourbar">
      <div className="colourbar-ramp" style={{ background: `linear-gradient(to right, ${stops})` }} />
      <div className="colourbar-labels small mono muted">
        <span>{props.left}</span>
        {props.mid && <span>{props.mid}</span>}
        <span>{props.right}</span>
      </div>
    </div>
  );
}

export function DeltaImage(props: { image: RecordingPreview["image"]; delta: NonNullable<RecordingPreview["delta_image"]> }) {
  const { image, delta } = props;
  const top = delta.scale_top;
  const codes = useMemo(() => new Int8Array(bytes(delta.data).buffer), [delta.data]); // signed: -127..127
  const colourOf = useMemo(() => {
    const [neg, zero, pos] = [cssColor("--dl-neg"), cssColor("--dl-zero"), cssColor("--dl-pos")];
    return (i: number): Rgb => {
      const f = codes[i] / 127;
      return f >= 0 ? mix(zero, pos, f) : mix(zero, neg, -f);
    };
  }, [codes]);
  const value = (i: number) => (codes[i] / 127) * Math.max(top, 1);
  const describe = (i: number) =>
    `≈ ${value(i) >= 0 ? "+" : ""}${value(i).toFixed(1)} Δ${Math.abs(codes[i]) === 127 ? " (at the colour limit)" : ""}`;
  const t = Number(Math.max(top, 1).toPrecision(3));
  return (
    <MapImage
      image={image}
      colour={colourOf}
      describe={describe}
      label="delta image: sum of event polarity per bin"
      legend={<Colourbar from="--dl-neg" via="--dl-zero" to="--dl-pos" left={`−${t} Δ darker`} mid="0" right={`+${t} Δ brighter`} />}
      caption={
        <>
          Reconstructed from events: each ON = +Δ, each OFF = −Δ, summed per {image.bin_px}×{image.bin_px} px bin over
          the whole recording; linear scale, ±{t} Δ = full colour (range {delta.min} … +{delta.max} Δ); not an absolute
          brightness · x 0–{image.sensor_width - 1} →, y 0–{image.sensor_height - 1} ↓ (sensor coordinates)
        </>
      }
    />
  );
}

export function TimeImage(props: { image: RecordingPreview["image"]; time: NonNullable<RecordingPreview["time_image"]> }) {
  const { image, time } = props;
  const codes = useMemo(() => bytes(time.data), [time.data]); // 0 = no events, 1..255 = start..end
  const colourOf = useMemo(() => {
    const [start, end] = [cssColor("--time-start"), cssColor("--time-end")];
    return (i: number): Rgb => (codes[i] === 0 ? [0, 0, 0] : mix(start, end, (codes[i] - 1) / 254));
  }, [codes]);
  const describe = (i: number) =>
    codes[i] === 0 ? "no events" : `mean event time ≈ ${(((codes[i] - 1) / 254) * time.seconds).toFixed(3)} s`;
  return (
    <MapImage
      image={image}
      colour={colourOf}
      describe={describe}
      label="time-coloured event image: mean event time per bin"
      legend={<Colourbar from="--time-start" to="--time-end" left="0 s (start)" right={`${time.seconds.toFixed(3)} s (end)`} />}
      caption={
        <>
          Mean time of the events in each {image.bin_px}×{image.bin_px} px bin: blue = start, red = end of the
          recording, <span className="legend-swatch none" /> black = no events; hot pixels left out · x 0–
          {image.sensor_width - 1} →, y 0–{image.sensor_height - 1} ↓ (sensor coordinates)
        </>
      }
    />
  );
}
