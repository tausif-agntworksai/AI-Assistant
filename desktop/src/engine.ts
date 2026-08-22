/**
 * Supervises the Python engine process.
 *
 * The engine owns the microphone, the models, and every Windows action; this
 * process only draws the HUD. They talk over a localhost HTTP + WebSocket API,
 * so the UI is a pure view of engine state and can reconnect freely without
 * disturbing anything the engine is doing.
 */
import { spawn, type ChildProcess } from "node:child_process";
import { randomBytes } from "node:crypto";
import { createWriteStream, existsSync, mkdirSync, type WriteStream } from "node:fs";
import { createServer } from "node:net";
import path from "node:path";
import { app } from "electron";

/**
 * Fixed rather than random: the renderer connects to this origin, and a
 * moving port would mean re-plumbing the URL into the page on every launch.
 * We only walk off it if something else already holds it.
 */
const PREFERRED_PORT = 8756;
const PORT_SCAN_LIMIT = 10;

/**
 * Generous on purpose. A first run downloads the Whisper weights (~500 MB)
 * before the API answers, and failing at 45 seconds would look like a crash
 * when it is really just a download.
 */
const HEALTH_TIMEOUT_MS = 15 * 60 * 1000;

const HOST = "127.0.0.1";

export interface EnginePaths {
  /** Directory to run the engine from. */
  cwd: string;
  /** Executable to launch. */
  command: string;
  /** Arguments for it. */
  args: string[];
  /** True when running the PyInstaller build rather than the source tree. */
  packaged: boolean;
}

/**
 * In development the engine runs from the repo's virtualenv; in a packaged
 * build it is a PyInstaller folder copied next to the app (see
 * `extraResources` in package.json).
 */
export function resolvePaths(): EnginePaths {
  if (app.isPackaged) {
    const dir = path.join(process.resourcesPath, "engine");
    return {
      cwd: dir,
      command: path.join(dir, "jarvis-engine.exe"),
      args: [],
      packaged: true,
    };
  }

  const engineDir = path.resolve(app.getAppPath(), "..", "engine");
  return {
    cwd: engineDir,
    command: path.join(engineDir, ".venv", "Scripts", "python.exe"),
    args: ["-m", "jarvis"],
    packaged: false,
  };
}

/** Lists what a launch needs but doesn't have yet, in human words. */
export function missingPieces(paths: EnginePaths): string[] {
  const missing: string[] = [];
  if (!existsSync(paths.command)) {
    missing.push(
      paths.packaged
        ? "the bundled engine (resources/engine)"
        : "the Python virtualenv — run engine\\setup.ps1 first"
    );
  }
  return missing;
}

function isPortFree(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const probe = createServer();
    probe.once("error", () => resolve(false));
    probe.once("listening", () => probe.close(() => resolve(true)));
    probe.listen(port, HOST);
  });
}

async function pickPort(): Promise<number> {
  for (let port = PREFERRED_PORT; port < PREFERRED_PORT + PORT_SCAN_LIMIT; port++) {
    if (await isPortFree(port)) return port;
  }
  throw new Error(
    `Ports ${PREFERRED_PORT}-${PREFERRED_PORT + PORT_SCAN_LIMIT - 1} are all in use.`
  );
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Shapes of the credentials that can turn up in engine output. The log file is
 * what people attach to bug reports, and a provider that rejects a key likes to
 * quote it back.
 */
const SECRET_PATTERNS: RegExp[] = [
  /\bsk-ant-[A-Za-z0-9_-]{12,}/g,
  /\bsk-[A-Za-z0-9_-]{16,}/g,
  /\bAIza[A-Za-z0-9_-]{20,}/g,
  /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}/g,
];

export class Engine {
  private child: ChildProcess | null = null;
  private log: WriteStream | null = null;
  /** Tail of the engine output, so a failed boot can show why. */
  private recent: string[] = [];
  private stopping = false;

  readonly paths = resolvePaths();
  port = PREFERRED_PORT;

  /**
   * Minted per launch and handed to the engine through its environment.
   *
   * The engine's API can shut the machine down, and it listens on a fixed
   * loopback port that every browser on this machine can also reach — a page
   * on any website can issue `fetch("http://127.0.0.1:8756/command")`. It
   * cannot, however, read this string. That is what makes the port safe.
   */
  readonly apiToken = randomBytes(32).toString("base64url");

  /** Called with each line the engine prints, for the HUD's log view. */
  onOutput: ((line: string) => void) | null = null;

  get url(): string {
    return `http://${HOST}:${this.port}`;
  }

  get wsUrl(): string {
    return `ws://${HOST}:${this.port}/ws`;
  }

  get logPath(): string {
    return path.join(app.getPath("logs"), "engine.log");
  }

  get tail(): string {
    return this.recent.join("").trim();
  }

  get running(): boolean {
    return this.child !== null;
  }

  /**
   * Boots the engine and resolves once /health answers. Rejects with the
   * engine's own output if it dies or never comes up.
   */
  async start(): Promise<string> {
    this.port = await pickPort();
    mkdirSync(path.dirname(this.logPath), { recursive: true });
    this.log = createWriteStream(this.logPath, { flags: "a" });
    this.write(`\n=== launch ${new Date().toISOString()} — port ${this.port} ===\n`);

    this.child = spawn(this.paths.command, this.paths.args, {
      cwd: this.paths.cwd,
      env: {
        ...process.env,
        JARVIS_PORT: String(this.port),
        JARVIS_API_TOKEN: this.apiToken,
        // Launched by the app, so the engine holds its microphone shut until a
        // signed-in session unlocks it. A bare `python -m jarvis` sets nothing
        // and keeps working as before.
        JARVIS_REQUIRE_SESSION: "1",
        // Devanagari transcripts crash a cp1252 stdout, which would take the
        // engine down mid-utterance.
        PYTHONIOENCODING: "utf-8",
        PYTHONUNBUFFERED: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });

    this.child.stdout?.on("data", (chunk) => this.write(String(chunk)));
    this.child.stderr?.on("data", (chunk) => this.write(String(chunk)));

    let exited: string | null = null;
    this.child.once("exit", (code, signal) => {
      this.child = null;
      if (this.stopping) return;
      exited = `The engine stopped unexpectedly (code ${code ?? signal}).`;
    });
    this.child.once("error", (err) => {
      exited = `The engine could not be started: ${err.message}`;
    });

    const deadline = Date.now() + HEALTH_TIMEOUT_MS;
    while (Date.now() < deadline) {
      if (exited) throw new Error(exited);
      if (await this.healthy()) return this.url;
      await sleep(400);
    }
    throw new Error("The engine did not respond in time.");
  }

  private async healthy(): Promise<boolean> {
    try {
      const res = await fetch(`${this.url}/health`);
      return res.ok;
    } catch {
      return false;
    }
  }

  /** Stops the engine. Safe to call more than once. */
  stop(): void {
    this.stopping = true;
    this.child?.kill();
    this.child = null;
    this.log?.end();
    this.log = null;
  }

  /* ── the authenticated API ────────────────────────────────────────────── */

  /** Every call the app makes to the engine goes through here, with the token. */
  async call<T>(route: string, body?: unknown): Promise<T> {
    const response = await fetch(`${this.url}${route}`, {
      method: body === undefined ? "GET" : "POST",
      headers: {
        Authorization: `Bearer ${this.apiToken}`,
        ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    if (!response.ok) {
      throw new Error(`${route} failed (${response.status})`);
    }
    return (await response.json()) as T;
  }

  /**
   * Tells the engine a verified person is present, and for how long. The TTL
   * matches the Firebase ID token that vouched for them, so an account revoked
   * upstream stops the microphone within one refresh cycle rather than at the
   * next restart.
   */
  async unlock(account: string, uid: string, ttlSec: number): Promise<void> {
    await this.call("/session/unlock", { account, uid, ttl_sec: ttlSec });
  }

  async lock(): Promise<void> {
    await this.call("/session/lock", {});
  }

  /** Mirrors the permission screen's decision into the engine that enforces it. */
  async pushConsent(granted: Record<string, boolean>): Promise<void> {
    await this.call("/consent", { granted });
  }

  /**
   * Lends the engine the user's model key for as long as it runs.
   *
   * "Lends" is the accurate word: the engine holds it in memory and writes it
   * nowhere. This process is where it actually lives, encrypted by the OS.
   */
  async pushLlm(provider: string, model: string, apiKey: string): Promise<void> {
    await this.call("/llm", { provider, model, api_key: apiKey });
  }

  async pushSpeech(provider: string, apiKey: string): Promise<void> {
    await this.call("/speech", { provider, api_key: apiKey });
  }

  /** What the settings screen renders: providers, models, current choice. */
  async llmInfo<T>(): Promise<T> {
    return this.call<T>("/llm");
  }

  /**
   * Proves a key works before it is saved. The engine makes the live call, so
   * the key travels main → loopback → provider and never through a page.
   */
  async validateKey(provider: string, apiKey: string, model = ""): Promise<void> {
    const response = await fetch(`${this.url}/llm/validate`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ provider, api_key: apiKey, model }),
    });
    const body = (await response.json().catch(() => ({}))) as { error?: string };
    if (!response.ok) throw new Error(body.error ?? "That key could not be verified.");
  }

  async listModels(provider: string, apiKey: string): Promise<unknown[]> {
    const response = await fetch(`${this.url}/llm/models`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ provider, api_key: apiKey }),
    });
    const body = (await response.json().catch(() => ({}))) as {
      models?: unknown[];
      error?: string;
    };
    if (!response.ok) throw new Error(body.error ?? "Could not list models.");
    return body.models ?? [];
  }

  private write(text: string): void {
    // The token isn't key-shaped, so scrub it by value as well.
    const safe = redactSecrets(text).split(this.apiToken).join("[redacted]");
    this.log?.write(safe);
    this.recent.push(safe);
    if (this.recent.length > 60) this.recent.shift();
    if (!app.isPackaged) process.stdout.write(safe);
    for (const line of safe.split(/\r?\n/)) {
      if (line.trim()) this.onOutput?.(line);
    }
  }
}

/** Replaces anything credential-shaped with a marker. Safe on any string. */
export function redactSecrets(text: string): string {
  return SECRET_PATTERNS.reduce((out, pattern) => out.replace(pattern, "[redacted]"), text);
}
