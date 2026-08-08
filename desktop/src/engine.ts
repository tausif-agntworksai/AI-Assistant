/**
 * Supervises the Python engine process.
 *
 * The engine owns the microphone, the models, and every Windows action; this
 * process only draws the HUD. They talk over a localhost HTTP + WebSocket API,
 * so the UI is a pure view of engine state and can reconnect freely without
 * disturbing anything the engine is doing.
 */
import { spawn, type ChildProcess } from "node:child_process";
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

export class Engine {
  private child: ChildProcess | null = null;
  private log: WriteStream | null = null;
  /** Tail of the engine output, so a failed boot can show why. */
  private recent: string[] = [];
  private stopping = false;

  readonly paths = resolvePaths();
  port = PREFERRED_PORT;

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

  private write(text: string): void {
    this.log?.write(text);
    this.recent.push(text);
    if (this.recent.length > 60) this.recent.shift();
    if (!app.isPackaged) process.stdout.write(text);
    for (const line of text.split(/\r?\n/)) {
      if (line.trim()) this.onOutput?.(line);
    }
  }
}
