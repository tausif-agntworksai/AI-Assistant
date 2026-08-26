/**
 * Minimise, tuck away, turn off — on every screen, including the gates.
 *
 * This window is frameless, so it has no OS titlebar and therefore none of the
 * buttons a window normally comes with. The sign-in and permission screens had
 * a draggable strip and nothing else, which meant someone who could not get
 * past sign-in — no authenticator to hand, a forgotten password — had no way to
 * minimise or quit except Task Manager. A gate you cannot back out of is a trap.
 *
 * Icons are inline SVG rather than emoji. `🗕` in particular is an uncommon
 * codepoint that renders as a hollow box on plenty of systems, which is a poor
 * showing for the button that gets the window off your screen.
 */
import { useEffect, useState } from "react";
import { jarvis } from "../bridge";
import "./windowcontrols.css";

function Minimise() {
  return (
    <svg viewBox="0 0 12 12" aria-hidden="true">
      <line x1="2.5" y1="6" x2="9.5" y2="6" />
    </svg>
  );
}

function Tray() {
  return (
    <svg viewBox="0 0 12 12" aria-hidden="true">
      <polyline points="3,4.5 6,7.5 9,4.5" />
    </svg>
  );
}

function Power() {
  return (
    <svg viewBox="0 0 12 12" aria-hidden="true">
      <path d="M3.6 3.2a3.9 3.9 0 1 0 4.8 0" />
      <line x1="6" y1="1.5" x2="6" y2="5.5" />
    </svg>
  );
}

export function WindowControls({
  /** The HUD offers "keep running in the tray"; the gate screens have no use for it. */
  showTray = false,
}: {
  showTray?: boolean;
}) {
  const [armed, setArmed] = useState(false);

  // Turning the assistant off sits a few pixels from the button that merely
  // gets it out of the way, and the outcomes are very different — so it asks
  // once, and disarms itself rather than staying dangerous.
  useEffect(() => {
    if (!armed) return;
    const timer = window.setTimeout(() => setArmed(false), 3500);
    return () => window.clearTimeout(timer);
  }, [armed]);

  return (
    <div className="wincontrols">
      <button
        className="wc"
        title="Minimise"
        aria-label="Minimise"
        onClick={() => void jarvis.minimize()}
      >
        <Minimise />
      </button>

      {showTray && (
        <button
          className="wc"
          title="Keep running in the tray — Jarvis stays awake and listening"
          aria-label="Hide to the tray"
          onClick={() => void jarvis.hide()}
        >
          <Tray />
        </button>
      )}

      {armed ? (
        <button className="wc-armed" onClick={() => void jarvis.quit()}>
          Turn off?
        </button>
      ) : (
        <button
          className="wc danger"
          title="Turn Jarvis off — stops listening completely"
          aria-label="Turn off"
          onClick={() => setArmed(true)}
        >
          <Power />
        </button>
      )}
    </div>
  );
}
