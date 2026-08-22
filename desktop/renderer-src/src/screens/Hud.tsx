/**
 * The HUD: the orb, the conversation, the activity log, and settings.
 *
 * A frameless always-on-top window, so it is deliberately small and quiet. The
 * one thing it insists on saying out loud is why it cannot hear you — a muted
 * microphone, a Windows privacy block or a level too low to recognise all look
 * identical from the outside, and all three are one sentence away from being
 * obvious.
 */
import { useEffect, useRef, useState } from "react";
import { jarvis } from "../bridge";
import { Badge, IconButton, Note, TitleBar } from "../components/ui";
import { Orb } from "../components/Orb";
import { WindowControls } from "../components/WindowControls";
import { useEngine } from "../useEngine";
import { Settings } from "./Settings";
import "./hud.css";

type Tab = "chat" | "skills" | "activity";

const STATE_LABEL: Record<string, string> = {
  starting: "starting…",
  idle: "listening for “hey jarvis”",
  listening: "listening…",
  thinking: "thinking…",
  acting: "working…",
  speaking: "speaking…",
  confirming: "waiting for your answer",
  error: "something went wrong",
};

export function Hud() {
  const engine = useEngine();
  const [tab, setTab] = useState<Tab>("chat");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [pinned, setPinned] = useState(true);
  const [draft, setDraft] = useState("");

  // The engine asks for settings when something needed the model and no key was
  // set. Opening it beats leaving the user to work out why a question went
  // unanswered.
  useEffect(() => {
    if (engine.needsKey) {
      setSettingsOpen(true);
      engine.clearNeedsKey();
    }
  }, [engine.needsKey, engine.clearNeedsKey]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      if (settingsOpen) setSettingsOpen(false);
      else void jarvis.hide();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [settingsOpen]);

  // Distinguish "the engine is booting" from "we cannot reach the engine".
  // Both used to read "starting…", which is how a refused WebSocket managed to
  // look like a slow launch for as long as anyone cared to wait.
  const label = !engine.connected
    ? engine.state === "error"
      ? "engine unavailable — see the log"
      : "connecting to the engine…"
    : engine.state === "listening" && engine.followUp
      ? "listening — go ahead"
      : (STATE_LABEL[engine.state] ?? engine.state);

  const mic = engine.status.microphone;
  const account = engine.status.session?.account;
  const llm = engine.status.llm;

  return (
    <div className="hud" data-state={engine.state}>
      <TitleBar>
        <span
          className={`dot ${engine.connected ? "ready" : "starting"}`}
          title={engine.connected ? "Connected to the engine" : "Connecting…"}
        />
        <IconButton
          title="Settings"
          active={settingsOpen}
          onClick={() => setSettingsOpen((open) => !open)}
        >
          ⚙
        </IconButton>
        <IconButton
          title={pinned ? "Always on top" : "Not pinned"}
          active={pinned}
          onClick={() => {
            const next = !pinned;
            setPinned(next);
            void jarvis.pin(next);
          }}
        >
          📌
        </IconButton>
        <IconButton title="Open engine log" onClick={() => void jarvis.openLog()}>
          🗒
        </IconButton>
        <WindowControls showTray />
      </TitleBar>

      <div className="stage">
        <Orb
          state={engine.state}
          level={engine.level}
          followUp={engine.followUp}
          onClick={() => void jarvis.listen()}
        />
        <div className="state-label">{label}</div>
      </div>

      {mic && !mic.open && (
        <div className="hud-alert">
          <Note kind="danger">Microphone closed — {mic.reason ?? "unavailable"}.</Note>
        </div>
      )}
      {mic?.open && mic.too_quiet && (
        <div className="hud-alert">
          <Note kind="warn">
            Your microphone is very quiet. Raise its level in Windows sound
            settings — recognition suffers badly below this.
          </Note>
        </div>
      )}

      <nav className="tabs hud-tabs" role="tablist">
        {(
          [
            ["chat", "Conversation"],
            ["skills", "Skills"],
            ["activity", "Activity"],
          ] as const
        ).map(([id, text]) => (
          <button
            key={id}
            role="tab"
            aria-selected={tab === id}
            className={tab === id ? "active" : ""}
            onClick={() => setTab(id)}
          >
            {text}
          </button>
        ))}
      </nav>

      <main className="hud-main">
        {tab === "chat" && <Conversation engine={engine} />}
        {tab === "skills" && <Skills engine={engine} onPick={setDraft} />}
        {tab === "activity" && <Activity engine={engine} />}
      </main>

      <footer className="hud-foot">
        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            void engine.send(draft);
            setDraft("");
          }}
        >
          <input
            className="text grow"
            placeholder="chrome kholo / what's the battery…"
            autoComplete="off"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
          />
          <button className="btn primary" type="submit">
            Run
          </button>
        </form>
        <div className="hint">
          <span>
            Wake word <b>hey jarvis</b>
          </span>
          <span>
            Hotkey <b>{engine.info?.hotkey ?? "Ctrl+Alt+J"}</b>
          </span>
          {(engine.localTurns > 0 || engine.aiTurns > 0) && (
            <span
              className="tally"
              title="How this session split between offline handling and the AI model"
            >
              {engine.localTurns} offline · {engine.aiTurns} AI
            </span>
          )}
          <span className="who truncate" title={account ?? ""}>
            {llm && !llm.has_key ? <Badge kind="warn">no AI key</Badge> : account}
          </span>
        </div>
      </footer>

      {settingsOpen && <Settings onClose={() => setSettingsOpen(false)} />}
    </div>
  );
}

/* ── tabs ─────────────────────────────────────────────────────────────── */

function Conversation({ engine }: { engine: ReturnType<typeof useEngine> }) {
  const endRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Follow the tail only when the reader is already at the bottom, so scrolling
  // back through a conversation is not yanked away mid-read.
  useEffect(() => {
    const box = scrollRef.current;
    if (!box) return;
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 90;
    if (atBottom) endRef.current?.scrollIntoView({ block: "end" });
  }, [engine.turns]);

  if (!engine.turns.length) {
    return (
      <div className="empty">
        Say <b>“hey jarvis”</b>, press <kbd>{engine.info?.hotkey ?? "Ctrl+Alt+J"}</kbd>,
        <br />
        or type a command below.
      </div>
    );
  }

  return (
    <div className="chat" ref={scrollRef}>
      {engine.turns.map((turn) => (
        <div key={turn.id} className={`msg ${turn.kind}`}>
          <div className="who">{WHO[turn.kind]}</div>
          <div className="body">
            {turn.text}
            {turn.meta && <span className="lang">{turn.meta}</span>}
            {turn.corrected && (
              <span className="lang" title="Re-heard with the accurate model">
                re-heard
              </span>
            )}
            <ViaBadge via={turn.via} />
          </div>
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}

/**
 * Says how a turn was resolved. The point is not decoration: "does this thing
 * call an AI for everything?" is a fair question about a voice assistant, and
 * the honest answer is a label on each turn rather than a paragraph in a README.
 */
function ViaBadge({ via }: { via?: string }) {
  if (!via) return null;
  if (via === "llm") {
    return (
      <span className="via ai" title="Answered by the AI model — this used your API key">
        AI
      </span>
    );
  }
  if (via === "local") {
    return (
      <span className="via" title="Answered by the engine itself — no model, no network">
        local
      </span>
    );
  }
  return (
    <span
      className="via"
      title={
        via === "rule"
          ? "Matched an offline rule — no model, no network, no cost"
          : "Matched a known phrasing offline — no model, no network, no cost"
      }
    >
      offline
    </span>
  );
}

const WHO: Record<string, string> = {
  you: "you",
  jarvis: "jarvis",
  action: "does",
  confirm: "asks",
  error: "error",
};

interface SkillSpec {
  name: string;
  description: string;
  risk: string;
  examples: string[];
}

function Skills({
  engine,
  onPick,
}: {
  engine: ReturnType<typeof useEngine>;
  onPick: (text: string) => void;
}) {
  const [categories, setCategories] = useState<Record<string, SkillSpec[]> | null>(null);

  useEffect(() => {
    const info = engine.info;
    if (!info) return;
    void fetch(`${info.url}/skills`, {
      headers: { authorization: `Bearer ${info.token}` },
    })
      .then((response) => response.json())
      .then((data) => setCategories(data.categories as Record<string, SkillSpec[]>))
      .catch(() => setCategories({}));
  }, [engine.info]);

  if (!categories) return <div className="empty">Loading skills…</div>;

  return (
    <div className="skills">
      {Object.entries(categories).map(([category, specs]) => (
        <div className="cat" key={category}>
          <h3>{category}</h3>
          {specs.map((spec) => (
            <button
              key={spec.name}
              className="skill"
              title={spec.description}
              onClick={() => onPick(spec.examples[0] ?? "")}
            >
              <Badge kind={spec.risk === "safe" ? "info" : spec.risk === "confirm" ? "warn" : "danger"}>
                {spec.risk === "safe" ? "auto" : spec.risk}
              </Badge>
              <span className="ex truncate">{spec.examples[0] ?? spec.name}</span>
            </button>
          ))}
        </div>
      ))}
    </div>
  );
}

interface AuditRow {
  iso: string;
  action: string;
  args: Record<string, unknown>;
  allowed: boolean;
  ok: boolean | null;
  detail: string;
}

function Activity({ engine }: { engine: ReturnType<typeof useEngine> }) {
  const [rows, setRows] = useState<AuditRow[] | null>(null);

  useEffect(() => {
    const info = engine.info;
    if (!info) return;
    void fetch(`${info.url}/audit?limit=80`, {
      headers: { authorization: `Bearer ${info.token}` },
    })
      .then((response) => response.json())
      .then((data) => setRows(data.entries as AuditRow[]))
      .catch(() => setRows([]));
    // Re-read whenever a turn lands, so the log follows what just happened.
  }, [engine.info, engine.turns.length]);

  if (!rows) return <div className="empty">Loading…</div>;
  if (!rows.length) return <div className="empty">No activity yet.</div>;

  return (
    <div className="activity">
      {rows.map((row, index) => (
        <div className="row" key={`${row.iso}-${index}`} title={row.detail}>
          <span className="time">{row.iso.slice(11, 16)}</span>
          <span className="what truncate">{row.action}</span>
          <span
            className={`res ${!row.allowed ? "blocked" : row.ok ? "ok" : "no"}`}
          >
            {!row.allowed ? "blocked" : row.ok ? "ok" : "failed"}
          </span>
        </div>
      ))}
    </div>
  );
}
