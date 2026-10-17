// Pure: the brightness curve of a recording as CSV text, the same kind of table as the Frame Lab's mean brightness
// per frame. Columns t_s (time in the recording, s) and delta_L (brightness change in contrast steps Δ).

export function brightnessCsv(t_s: number[], delta_l: number[]): string {
  const rows = t_s.map((t, i) => `${t},${delta_l[i]}`);
  return ["t_s,delta_L", ...rows].join("\n") + "\n";
}
