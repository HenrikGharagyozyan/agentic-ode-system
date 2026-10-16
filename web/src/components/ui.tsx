// Small building blocks shared by the panels.

import type { ReactNode } from "react";

export type Tone = "ok" | "warn" | "error" | "info" | "muted" | "sim";

export function Panel(props: { title: string; aside?: ReactNode; className?: string; children: ReactNode }) {
  return (
    <section className={`panel ${props.className ?? ""}`}>
      <header className="panel-head">
        <h2>{props.title}</h2>
        {props.aside && <div className="panel-aside">{props.aside}</div>}
      </header>
      <div className="panel-body">{props.children}</div>
    </section>
  );
}

export function Badge(props: { tone: Tone; children: ReactNode; title?: string }) {
  return (
    <span className={`badge tone-${props.tone}`} title={props.title}>
      {props.children}
    </span>
  );
}

/** A status dot; pulse only while something really runs. */
export function Dot(props: { tone: Tone; pulse?: boolean }) {
  return <span className={`dot tone-${props.tone}${props.pulse ? " pulse" : ""}`} aria-hidden="true" />;
}

export function Spinner(props: { label: string }) {
  return (
    <span className="spinner" role="status">
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <circle cx="8" cy="8" r="6" />
      </svg>
      <span className="sr-only">{props.label}</span>
    </span>
  );
}

/** A definition list row: label on the left, value on the right. */
export function Row(props: { label: string; children: ReactNode; mono?: boolean }) {
  return (
    <div className="row">
      <dt>{props.label}</dt>
      <dd className={props.mono ? "mono" : undefined}>{props.children}</dd>
    </div>
  );
}

export function Notice(props: { tone: Tone; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className={`notice tone-${props.tone}`} role={props.tone === "error" ? "alert" : undefined}>
      <div>
        <strong>{props.title}</strong>
        {props.children && <div className="notice-text">{props.children}</div>}
      </div>
      {props.action}
    </div>
  );
}
