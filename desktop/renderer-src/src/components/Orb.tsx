/**
 * The listening indicator, and the app's only piece of real ornament.
 *
 * It is driven by the engine's live level stream rather than a canned
 * animation, which is the difference between decoration and a diagnostic: if
 * the ring is not moving while you talk, the microphone is not hearing you, and
 * that is worth being able to see at a glance.
 */
import "./orb.css";

export type OrbState =
  | "starting"
  | "idle"
  | "listening"
  | "thinking"
  | "acting"
  | "speaking"
  | "confirming"
  | "error";

export function Orb({
  state,
  level = 0,
  size = 76,
  onClick,
  followUp,
}: {
  state: OrbState;
  /** 0..1 microphone loudness. Scales the ring. */
  level?: number;
  size?: number;
  onClick?: () => void;
  followUp?: boolean;
}) {
  const clamped = Math.max(0, Math.min(1, level));
  return (
    <div
      className={`orb${onClick ? " clickable" : ""}${followUp ? " followup" : ""}`}
      data-orb-state={state}
      style={{
        width: size,
        height: size,
        // Read by the stylesheet so the ring tracks your voice without a
        // re-render per animation frame.
        ["--level" as string]: clamped.toFixed(3),
      }}
      onClick={onClick}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={(event) => {
        if (onClick && (event.key === "Enter" || event.key === " ")) onClick();
      }}
      title={onClick ? "Click to talk" : undefined}
      aria-label={onClick ? "Talk to Jarvis" : undefined}
    >
      <span className="orb-ring" aria-hidden="true" />
      <span className="orb-core" aria-hidden="true" />
    </div>
  );
}
