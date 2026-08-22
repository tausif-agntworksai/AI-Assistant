/**
 * "Can Jarvis hear you?" — a setup step, not a diagnostic buried in settings.
 *
 * The single most common way a voice assistant is broken is that it cannot hear
 * you, and almost always for a reason nothing reports: the OS privacy switch is
 * off, the default input is a virtual device with no signal, or the level is so
 * low that recognition returns confident nonsense. All three look identical from
 * the outside — you talk, nothing happens.
 *
 * So this asks you to say the wake word once and shows what arrived. A moving
 * ring means audio is reaching the engine; a transcript means it understood.
 * Getting that wrong here is a two-minute fix, and finding out later is an hour
 * of wondering whether the app works at all.
 */
import { useEffect, useState } from "react";
import { consent } from "../bridge";
import { useEngine } from "../useEngine";
import { Button, Note, Spinner } from "./ui";
import { Orb } from "./Orb";
import "./miccheck.css";

type Verdict = "waiting" | "hearing" | "heard" | "silent";

/** −48 dBFS. Below this the local model guesses rather than transcribes. */
const QUIET_PEAK = 0.004;
const SILENT_AFTER_MS = 12_000;

export function MicCheck({ blocked }: { blocked: boolean }) {
  const { state, level, turns, status, connected } = useEngine();
  const [peak, setPeak] = useState(0);
  const [elapsed, setElapsed] = useState(0);

  // Track the loudest thing we have seen, so a brief word still registers.
  useEffect(() => {
    setPeak((previous) => Math.max(previous, level));
  }, [level]);

  useEffect(() => {
    const timer = window.setInterval(() => setElapsed((n) => n + 1000), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const heard = turns.filter((turn) => turn.kind === "you");
  const mic = status.microphone;

  const verdict: Verdict = heard.length
    ? "heard"
    : peak > QUIET_PEAK
      ? "hearing"
      : elapsed >= SILENT_AFTER_MS
        ? "silent"
        : "waiting";

  return (
    <div className="miccheck">
      <Orb
        state={state === "starting" ? "starting" : state}
        level={level}
        size={92}
        followUp={false}
      />

      <div className="miccheck-meter" aria-hidden="true">
        <i style={{ width: `${Math.min(100, level * 100)}%` }} />
      </div>

      {!connected && <Spinner label="Waiting for the engine…" />}

      {blocked && (
        <Note
          kind="danger"
          action={
            <Button variant="secondary" onClick={() => void consent.openOsSettings("microphone")}>
              Open settings
            </Button>
          }
        >
          Windows is blocking microphone access for apps. Nothing will be heard
          until that is changed.
        </Note>
      )}

      {!blocked && mic && !mic.open && (
        <Note kind="warn">Microphone closed — {mic.reason ?? "unavailable"}.</Note>
      )}

      {verdict === "waiting" && connected && (
        <Note kind="info">Say “hey jarvis”, then a command like “what time is it”.</Note>
      )}

      {verdict === "hearing" && (
        <Note kind="ok">Audio is reaching Jarvis. Now say “hey jarvis”.</Note>
      )}

      {verdict === "heard" && (
        <>
          <Note kind="ok">Heard you clearly.</Note>
          <div className="miccheck-heard">
            {heard.slice(-3).map((turn) => (
              <p key={turn.id}>“{turn.text}”</p>
            ))}
          </div>
        </>
      )}

      {verdict === "silent" && !blocked && (
        <Note kind="warn">
          Nothing is arriving. Check that the right microphone is the Windows
          default and that it is not muted — then reopen this screen.
        </Note>
      )}

      {mic?.too_quiet && (
        <Note kind="warn">
          What is arriving is very quiet. Raise the input level in Windows sound
          settings, or move closer — recognition gets much worse below this.
        </Note>
      )}

      {mic && !mic.wake_word && (
        <Note kind="info">
          The wake word model is not loaded, so “hey jarvis” will not trigger.
          Use the hotkey or click the orb instead.
        </Note>
      )}
    </div>
  );
}
