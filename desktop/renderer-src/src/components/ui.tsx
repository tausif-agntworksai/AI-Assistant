/**
 * The shared primitives. Small on purpose: the point of a component layer here
 * is that every screen gets the same focus ring, the same disabled state and
 * the same bilingual caption treatment without three copies of the CSS.
 */
import type { ReactNode } from "react";
import "./ui.css";

export function Button({
  children,
  onClick,
  variant = "primary",
  disabled,
  type = "button",
  full,
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "ghost" | "danger";
  disabled?: boolean;
  type?: "button" | "submit";
  full?: boolean;
  title?: string;
}) {
  return (
    <button
      type={type}
      className={`btn ${variant}${full ? " full" : ""}`}
      onClick={onClick}
      disabled={disabled}
      title={title}
    >
      {children}
    </button>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

/** A label with its Hindi twin underneath, used everywhere both are shown. */
export function Bilingual({ en, hi }: { en: ReactNode; hi?: string }) {
  return (
    <>
      {en}
      {hi && (
        <span className="hi bilingual-hi" lang="hi">
          {hi}
        </span>
      )}
    </>
  );
}

export type NoteKind = "info" | "ok" | "warn" | "danger";

export function Note({
  kind = "info",
  children,
  action,
}: {
  kind?: NoteKind;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className={`note ${kind}`} role={kind === "danger" ? "alert" : undefined}>
      <span className="grow">{children}</span>
      {action}
    </div>
  );
}

export function Badge({ kind = "info", children }: { kind?: NoteKind; children: ReactNode }) {
  return <span className={`badge ${kind}`}>{children}</span>;
}

export function Switch({
  checked,
  onChange,
  disabled,
  label,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  disabled?: boolean;
  label: string;
}) {
  return (
    <span className="switch">
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        aria-label={label}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span aria-hidden="true" />
    </span>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="spinner-wrap">
      <span className="spinner" aria-hidden="true" />
      {label && <span className="muted">{label}</span>}
    </span>
  );
}

/** The draggable title strip of a frameless window. */
export function TitleBar({ children }: { children?: ReactNode }) {
  return (
    <header className="titlebar">
      <span className="brand">JARVIS</span>
      <span className="grow" />
      {children}
    </header>
  );
}

export function IconButton({
  children,
  onClick,
  title,
  active,
  danger,
}: {
  children: ReactNode;
  onClick?: () => void;
  title: string;
  active?: boolean;
  danger?: boolean;
}) {
  return (
    <button
      className={`icon-btn${active ? " active" : ""}${danger ? " danger" : ""}`}
      onClick={onClick}
      title={title}
      aria-label={title}
    >
      {children}
    </button>
  );
}
