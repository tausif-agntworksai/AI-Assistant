/**
 * The live connection to the engine.
 *
 * One WebSocket, one reconnect loop, one event reducer — shared by the HUD and
 * by the microphone check, because both want the same two things: what state the
 * engine is in, and how loud the room is.
 *
 * The token rides in the subprotocol list rather than a query string. A browser
 * WebSocket cannot set an Authorization header, and a query string ends up in
 * logs and referrers.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { jarvis, type EngineInfo } from "./bridge";

export type EngineState =
  | "starting"
  | "idle"
  | "listening"
  | "thinking"
  | "acting"
  | "speaking"
  | "confirming"
  | "error";

export interface Turn {
  id: number;
  kind: "you" | "jarvis" | "action" | "error" | "confirm";
  text: string;
  meta?: string;
  /** True when a re-decode replaced the words for this same utterance. */
  corrected?: boolean;
  /**
   * How this turn was resolved: "rule"/"example" offline, "llm" if a model was
   * asked, "local" for the engine's own canned words. Shown as a badge, so the
   * split between free local work and paid calls is visible per turn instead of
   * being something the documentation asserts.
   */
  via?: string;
}

export interface MicHealth {
  open: boolean;
  reason: string | null;
  recent_peak: number;
  too_quiet: boolean;
  wake_word: boolean;
}

export interface EngineStatus {
  microphone?: MicHealth;
  session?: { account: string };
  llm?: { provider: string; label: string; model: string; has_key: boolean };
  speech?: { provider: string; label: string; has_key: boolean };
  skills?: number;
}

const RECONNECT_MS = 1500;
const LEVEL_DECAY_MS = 400;
const MAX_TURNS = 200;

export function useEngine() {
  const [info, setInfo] = useState<EngineInfo | null>(null);
  const [connected, setConnected] = useState(false);
  const [state, setState] = useState<EngineState>("starting");
  const [followUp, setFollowUp] = useState(false);
  const [level, setLevel] = useState(0);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [status, setStatus] = useState<EngineStatus>({});
  const [needsKey, setNeedsKey] = useState(false);

  const socket = useRef<WebSocket | null>(null);
  const retryTimer = useRef<number | undefined>(undefined);
  const decayTimer = useRef<number | undefined>(undefined);
  const nextId = useRef(1);

  const addTurn = useCallback((turn: Omit<Turn, "id">) => {
    setTurns((prev) => {
      const next = [...prev, { ...turn, id: nextId.current++ }];
      return next.length > MAX_TURNS ? next.slice(-MAX_TURNS) : next;
    });
  }, []);

  /** GET /status — the microphone diagnosis, the account, the chosen model. */
  const refreshStatus = useCallback(async (current: EngineInfo) => {
    try {
      const response = await fetch(`${current.url}/status`, {
        headers: { authorization: `Bearer ${current.token}` },
      });
      if (response.ok) setStatus((await response.json()) as EngineStatus);
    } catch {
      /* the reconnect loop comes back around */
    }
  }, []);

  useEffect(() => {
    void jarvis.info().then(setInfo);
  }, []);

  useEffect(() => {
    if (!info) return;
    let disposed = false;

    const handle = (event: Record<string, unknown>) => {
      switch (event.type) {
        case "state":
          setState(event.state as EngineState);
          setFollowUp(Boolean(event.follow_up));
          break;

        case "level": {
          setLevel(Number(event.level ?? 0));
          // Decay to silence if the stream stops, so the ring cannot freeze
          // mid-pulse and imply we are still listening.
          window.clearTimeout(decayTimer.current);
          decayTimer.current = window.setTimeout(() => setLevel(0), LEVEL_DECAY_MS);
          break;
        }

        case "transcript":
          setTurns((prev) => {
            // A re-decode replaces the words of the same utterance rather than
            // adding a second bubble. The user said it once.
            const last = prev[prev.length - 1];
            if (event.corrected && last && last.kind === "you") {
              return [
                ...prev.slice(0, -1),
                { ...last, text: String(event.text ?? ""), corrected: true },
              ];
            }
            const next = [
              ...prev,
              {
                id: nextId.current++,
                kind: "you" as const,
                text: String(event.text ?? ""),
                meta: String(event.language ?? ""),
              },
            ];
            return next.length > MAX_TURNS ? next.slice(-MAX_TURNS) : next;
          });
          void refreshStatus(info);
          break;

        case "reply":
          addTurn({
            kind: "jarvis",
            text: String(event.text ?? ""),
            via: String(event.via ?? ""),
          });
          break;

        case "action":
          addTurn({
            kind: "action",
            text: `${String(event.skill)}(${formatArgs(event.args)})`,
            via: String(event.via ?? ""),
          });
          break;

        case "confirm_request":
          addTurn({ kind: "confirm", text: String(event.prompt ?? "") });
          break;

        case "confirm_result":
          addTurn({ kind: "confirm", text: event.approved ? "Approved." : "Cancelled." });
          break;

        case "needs_key":
          setNeedsKey(true);
          break;

        case "error":
          addTurn({ kind: "error", text: String(event.message ?? "unknown error") });
          void refreshStatus(info);
          break;

        default:
          break;
      }
    };

    const connect = () => {
      if (disposed) return;
      let ws: WebSocket;
      try {
        ws = new WebSocket(info.ws, ["jarvis.bearer", info.token]);
      } catch {
        retryTimer.current = window.setTimeout(connect, RECONNECT_MS);
        return;
      }
      socket.current = ws;

      ws.onopen = () => {
        setConnected(true);
        void refreshStatus(info);
      };
      ws.onclose = () => {
        setConnected(false);
        if (!disposed) retryTimer.current = window.setTimeout(connect, RECONNECT_MS);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (message) => {
        try {
          handle(JSON.parse(String(message.data)) as Record<string, unknown>);
        } catch {
          /* a malformed frame is not worth tearing the socket down for */
        }
      };
    };

    connect();
    return () => {
      disposed = true;
      window.clearTimeout(retryTimer.current);
      window.clearTimeout(decayTimer.current);
      socket.current?.close();
      socket.current = null;
    };
  }, [info, addTurn, refreshStatus]);

  // Engine lifecycle messages arrive from the main process, not the socket.
  useEffect(() => {
    const offStatus = jarvis.onStatus((payload) => {
      if ((payload as { state?: string })?.state === "failed") setState("error");
    });
    const offError = jarvis.onError((message) => addTurn({ kind: "error", text: message }));
    return () => {
      offStatus();
      offError();
    };
  }, [addTurn]);

  const send = useCallback(
    async (text: string) => {
      const value = text.trim();
      if (!info || !value) return;
      addTurn({ kind: "you", text: value });
      try {
        const response = await fetch(`${info.url}/command`, {
          method: "POST",
          headers: {
            authorization: `Bearer ${info.token}`,
            "content-type": "application/json",
          },
          body: JSON.stringify({ text: value }),
        });
        if (!response.ok) {
          const body = (await response.json().catch(() => ({}))) as { error?: string };
          addTurn({ kind: "error", text: body.error ?? `Engine said ${response.status}.` });
        }
      } catch (err) {
        addTurn({ kind: "error", text: `Engine unreachable: ${(err as Error).message}` });
      }
    },
    [info, addTurn]
  );

  const refresh = useCallback(() => {
    if (info) void refreshStatus(info);
  }, [info, refreshStatus]);

  return {
    info,
    connected,
    state,
    followUp,
    level,
    turns,
    status,
    needsKey,
    clearNeedsKey: useCallback(() => setNeedsKey(false), []),
    send,
    refresh,
  };
}

export function formatArgs(args: unknown): string {
  if (!args || typeof args !== "object") return "";
  return Object.entries(args as Record<string, unknown>)
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
    .join(", ");
}
