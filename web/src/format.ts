// Small display helpers.

/** "14:03:22" from an ISO time or a Date. */
export function clock(t: string | Date): string {
  const d = typeof t === "string" ? new Date(t) : t;
  return Number.isNaN(d.getTime()) ? String(t) : d.toLocaleTimeString([], { hour12: false });
}

/** "2026-10-16 14:03:22" from an ISO time. */
export function dateTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return `${d.toLocaleDateString("sv-SE")} ${clock(d)}`; // sv-SE gives YYYY-MM-DD
}

/** A number with a fixed count of significant digits, or "—" when there is no value. */
export function num(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Math.abs(v) >= 1e5 ? v.toExponential(2) : Number(v.toPrecision(digits)).toLocaleString("en-US");
}
