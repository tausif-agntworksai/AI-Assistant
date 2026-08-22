/**
 * Electron main process: the HUD window, the tray icon, and the global hotkey.
 *
 * The window is deliberately frameless and always-on-top — an assistant you
 * summon with a hotkey shouldn't make you hunt for it in the taskbar.
 *
 * This process is also the security boundary. It holds the Firebase refresh
 * token (the renderer never sees it), it holds the token that authenticates
 * every call to the engine, and it is what tells the engine whether it may
 * open the microphone at all. The order on every launch is:
 *
 *     boot the engine (locked) → permissions → sign in → unlock → listening
 */
import path from "node:path";
import {
  app,
  BrowserWindow,
  dialog,
  globalShortcut,
  ipcMain,
  Menu,
  nativeImage,
  screen,
  session as electronSession,
  shell,
  systemPreferences,
  Tray,
} from "electron";
import { Engine, missingPieces } from "./engine";
import { sessionManager, type AuthStatus } from "./auth/session";
import {
  CAPABILITIES,
  consentStore,
  openOsSettings,
  osAccess,
  type Grants,
} from "./consent";
import { keyStore } from "./llmKeys";
import { log, logPath } from "./log";
import { firebaseConfigured, requireAuth, twoFactorRequired } from "./config";

const HOTKEY = "Control+Alt+J";
const WIDTH = 400;
const HEIGHT = 560;
const MARGIN = 24;

/** The sign-in and permission screens need more room than the HUD. */
const GATE_WIDTH = 460;
const GATE_HEIGHT = 660;

const engine = new Engine();
let win: BrowserWindow | null = null;
let tray: Tray | null = null;
let quitting = false;
let engineReady = false;
/** True once a screen has painted at least once. See render-process-gone. */
let painted = false;

// A single instance owns the microphone and the port; a second would fight it
// for both and fail in confusing ways.
//
// `app.quit()` is a request, not a return: without this flag the rest of
// startup still ran, built a window, and began loading a page that was then
// torn down underneath it — surfacing as a bare `ERR_FAILED` with no
// indication that a second instance was the reason.
const isPrimaryInstance = app.requestSingleInstanceLock();
if (!isPrimaryInstance) {
  app.quit();
} else {
  app.on("second-instance", () => showWindow());
}

// Renderers get the OS sandbox. Nothing in this app's pages needs Node, and a
// page bug should not be able to reach for it.
app.enableSandbox();

function iconPath(): string {
  return path.join(app.getAppPath(), "assets", "icon.png");
}

/* ── which screen the window should be on ───────────────────────────────── */

type Screen = "auth" | "consent" | "hud";

function requiredScreen(): Screen {
  if (consentStore.needsAsking) return "consent";
  if (requireAuth && !sessionManager.ready) return "auth";
  return "hud";
}

let currentScreen: Screen | null = null;

/**
 * Navigations are serialised through here.
 *
 * Sign-in state arrives asynchronously and can settle twice in quick
 * succession — restore() emits a change, then the caller re-checks — and two
 * overlapping `loadFile` calls make Chromium abort the first one, which
 * surfaces as an unhandled ERR_ABORTED rejection. Queuing them means the last
 * decision wins and the superseded load is simply a no-op.
 */
let navigating: Promise<void> = Promise.resolve();

function showScreen(target: Screen, force = false): Promise<void> {
  if (!win || win.isDestroyed()) return Promise.resolve();
  if (target === currentScreen && !force) return navigating;
  currentScreen = target;

  navigating = navigating.then(async () => {
    if (!win || win.isDestroyed() || currentScreen !== target) return;
    const page = { auth: "auth.html", consent: "consent.html", hud: "index.html" }[target];
    const gate = target !== "hud";
    win.setMinimumSize(gate ? 380 : 340, gate ? 520 : 420);
    win.setSize(gate ? GATE_WIDTH : WIDTH, gate ? GATE_HEIGHT : HEIGHT);
    try {
      await win.loadFile(path.join(app.getAppPath(), "renderer-dist", page));
    } catch (err) {
      // A load superseded by a newer one is the queue working, not a failure.
      if (!String(err).includes("ERR_ABORTED")) throw err;
      return;
    }
    // A screen that needs an answer has to be visible, even at login.
    if (gate) win.show();
  });
  return navigating;
}

/** Re-evaluates the gate and moves the window if the answer changed. */
async function syncScreen(): Promise<void> {
  await showScreen(requiredScreen());
}

/* ── window ─────────────────────────────────────────────────────────────── */

function createWindow(): BrowserWindow {
  const display = screen.getPrimaryDisplay().workArea;

  const window = new BrowserWindow({
    width: WIDTH,
    height: HEIGHT,
    x: display.x + display.width - WIDTH - MARGIN,
    y: display.y + display.height - HEIGHT - MARGIN,
    frame: false,
    resizable: true,
    minWidth: 340,
    minHeight: 420,
    transparent: false,
    backgroundColor: "#0b0f17",
    alwaysOnTop: true,
    skipTaskbar: false,
    show: false,
    icon: iconPath(),
    title: "Jarvis",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      // The preload uses only `ipcRenderer`, which is available in a sandboxed
      // renderer — so the sandbox costs nothing and removes a whole class of
      // escalation from a page bug.
      sandbox: true,
      webSecurity: true,
      allowRunningInsecureContent: false,
      webviewTag: false,
      spellcheck: false,
    },
  });

  // `--hidden` is what the run-at-login entry passes: start listening in the
  // tray without taking focus at sign-in.
  const startHidden = process.argv.includes("--hidden");
  window.once("ready-to-show", () => {
    // A gate screen has to be seen — starting hidden with an un-answered
    // permission prompt would leave the assistant permanently mute and
    // silently so.
    if (!startHidden || requiredScreen() !== "hud") window.show();
  });

  // The window is frameless and has no menu, so there is no way to open dev
  // tools by hand — a broken sign-in screen would otherwise fail in complete
  // silence. Errors from a page land in the same place as everything else.
  window.webContents.on("console-message", (event) => {
    if (event.level !== "error" && event.level !== "warning") return;
    log(
      event.level === "error" ? "error" : "warn",
      `[renderer] ${event.message} (${event.sourceId}:${event.lineNumber})`
    );
  });

  // A page that fails to load at all never gets as far as a console message.
  window.webContents.on("did-fail-load", (_event, code, description, url) => {
    if (code === -3) return; // ERR_ABORTED: a navigation we superseded ourselves
    log("error", `[renderer] failed to load ${url}: ${description} (${code})`);
  });

  // Positive confirmation that a screen actually painted. Worth logging even
  // when nothing is wrong: in a packaged build this line is the only evidence
  // that the renderer is alive, and its absence is what a white screen looks
  // like from the outside.
  window.webContents.on("did-finish-load", () => {
    painted = true;
    log("info", `[renderer] ${currentScreen ?? "page"} painted`);
  });

  window.webContents.on("render-process-gone", (_event, details) => {
    // Shutting the app down kills the renderer, and Chromium reports that as a
    // crash — so this fired on every clean quit and cried wolf. What actually
    // matters is a renderer that dies *before* painting anything, because that
    // is a white screen with no other symptom.
    if (quitting || painted) {
      log("info", `[renderer] process ended during shutdown (${details.reason})`);
      return;
    }
    log("error", `[renderer] died before painting: ${details.reason}`);
  });

  // Closing hides to the tray; the assistant is meant to keep listening.
  window.on("close", (event) => {
    if (quitting) return;
    event.preventDefault();
    window.hide();
  });

  // Links open in the real browser, never inside the HUD, and only ever http(s)
  // — `openExternal` on an arbitrary scheme launches whatever is registered for
  // it, which is a lot of trust to place in a string that came from a page.
  window.webContents.setWindowOpenHandler(({ url }) => {
    if (isWebUrl(url)) void shell.openExternal(url);
    return { action: "deny" };
  });
  window.webContents.on("will-navigate", (event, url) => {
    if (url.startsWith("file://")) return;
    event.preventDefault();
    if (isWebUrl(url)) void shell.openExternal(url);
  });

  return window;
}

function isWebUrl(url: string): boolean {
  try {
    const scheme = new URL(url).protocol;
    return scheme === "http:" || scheme === "https:";
  } catch {
    return false;
  }
}

function showWindow(): void {
  if (!win) return;
  if (win.isMinimized()) win.restore();
  win.show();
  win.focus();
}

function send(channel: string, payload: unknown): void {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
}

/* ── permissions asked of Chromium ──────────────────────────────────────── */

/**
 * The HUD is a local page that needs nothing from this list, so the honest
 * default is to refuse everything. Without a handler Electron approves some
 * requests silently, which would make the permission screen a decoration.
 */
function lockDownPermissions(): void {
  const ses = electronSession.defaultSession;
  ses.setPermissionRequestHandler((_contents, permission, callback) => {
    console.warn(`Refused a renderer permission request: ${permission}`);
    callback(false);
  });
  ses.setPermissionCheckHandler(() => false);
}

/* ── tray ───────────────────────────────────────────────────────────────── */

/** True when Windows launches the app at sign-in. */
function launchesAtLogin(): boolean {
  return app.getLoginItemSettings().openAtLogin;
}

function setLaunchAtLogin(enabled: boolean): void {
  app.setLoginItemSettings({
    openAtLogin: enabled,
    // Start hidden — an assistant that steals focus every morning is worse
    // than one you have to click the tray icon for.
    args: ["--hidden"],
  });
  buildTrayMenu();
}

function buildTrayMenu(): void {
  if (!tray) return;
  const status = sessionManager.status();
  const account =
    status.state === "unconfigured"
      ? "Sign-in not configured"
      : status.email
        ? `Signed in as ${status.email}`
        : "Not signed in";

  tray.setContextMenu(
    Menu.buildFromTemplate([
      { label: account, enabled: false },
      { type: "separator" },
      { label: "Show Jarvis", click: () => showWindow() },
      {
        label: `Listen now (${HOTKEY})`,
        enabled: currentScreen === "hud",
        click: () => void triggerListen(),
      },
      { type: "separator" },
      { label: "Permissions…", click: () => void showScreen("consent", true) },
      {
        label: "Start with Windows",
        type: "checkbox",
        checked: launchesAtLogin(),
        click: (item) => setLaunchAtLogin(item.checked),
      },
      { type: "separator" },
      { label: "Open engine log", click: () => shell.openPath(engine.logPath) },
      { label: "Open app log", click: () => shell.openPath(logPath()) },
      { label: "Open data folder", click: () => shell.openPath(app.getPath("logs")) },
      { type: "separator" },
      {
        label: "Sign out",
        enabled: Boolean(status.email),
        click: () => void doSignOut(),
      },
      {
        label: "Quit",
        click: () => {
          quitting = true;
          app.quit();
        },
      },
    ])
  );
}

function createTray(): void {
  const image = nativeImage.createFromPath(iconPath());
  tray = new Tray(image.isEmpty() ? nativeImage.createEmpty() : image);
  tray.setToolTip("Jarvis — say “hey jarvis”");
  buildTrayMenu();
  tray.on("click", () => (win?.isVisible() ? win.hide() : showWindow()));
}

/* ── the engine ─────────────────────────────────────────────────────────── */

async function triggerListen(): Promise<void> {
  try {
    await engine.call("/listen", {});
    showWindow();
  } catch (err) {
    send("engine:error", `Could not reach the engine: ${(err as Error).message}`);
  }
}

/**
 * Pushes the current permission and sign-in state down to the engine.
 *
 * This is the join between the two processes: the engine enforces both, and
 * until this runs it holds the microphone closed. Called after boot, after
 * every permission change, and on every sign-in or sign-out.
 */
async function syncEngineState(): Promise<void> {
  if (!engineReady) return;
  try {
    // Nothing is pushed until the user has actually answered the permission
    // screen — the engine treats "no record" as command-line use and allows
    // everything, and writing the defaults early would turn a screen the user
    // is still reading into a decision they never made.
    if (!consentStore.needsAsking) {
      await engine.pushConsent(consentStore.all() as unknown as Record<string, boolean>);
    }
    const status = sessionManager.status();
    const allowed = !requireAuth || status.state === "ready";

    if (allowed) {
      // Keys are stored per account, so they are loaded here rather than at
      // startup: we don't know whose they are until someone has signed in.
      keyStore.load(status.uid);
      const llm = keyStore.llmChoice();
      const speech = keyStore.speechChoice();
      await engine.pushLlm(llm.provider, llm.model, llm.apiKey);
      await engine.pushSpeech(speech.provider, speech.apiKey);
      await engine.unlock(status.email, status.uid, sessionManager.unlockTtlSec || 3600);
    } else {
      // Signing out takes the key back out of the engine's memory, not just
      // the microphone away from it.
      await engine.pushLlm("anthropic", "", "");
      await engine.pushSpeech("", "");
      await engine.lock();
    }
  } catch (err) {
    send("engine:error", `Could not update the engine: ${(err as Error).message}`);
  }
}

async function boot(): Promise<void> {
  const missing = missingPieces(engine.paths);
  if (missing.length) {
    dialog.showErrorBox(
      "Jarvis can't start",
      `Missing ${missing.join(" and ")}.\n\n` +
        `Expected: ${engine.paths.command}\n\n` +
        (engine.paths.packaged
          ? "Try reinstalling."
          : "Run this once from the engine folder:\n" +
            "  powershell -ExecutionPolicy Bypass -File setup.ps1")
    );
    app.quit();
    return;
  }

  engine.onOutput = (line) => send("engine:log", line);
  send("engine:status", { state: "starting" });

  try {
    await engine.start();
    engineReady = true;
    send("engine:status", { state: "ready", url: engine.url, ws: engine.wsUrl });
    await syncEngineState();
  } catch (err) {
    const message = (err as Error).message;
    send("engine:status", { state: "failed", message });
    dialog.showErrorBox(
      "Jarvis engine failed to start",
      `${message}\n\nLast output:\n${engine.tail || "(nothing)"}\n\nLog: ${engine.logPath}`
    );
  }
}

/* ── sign-in ────────────────────────────────────────────────────────────── */

async function afterAuthChange(status: AuthStatus): Promise<AuthStatus> {
  await syncEngineState();
  buildTrayMenu();
  await syncScreen();
  send("auth:status", status);
  return status;
}

async function doSignOut(): Promise<AuthStatus> {
  const status = await sessionManager.signOut();
  return afterAuthChange(status);
}

/** Turns a thrown auth error into something a person can act on. */
function authError(err: unknown): never {
  const message = err instanceof Error ? err.message : "Something went wrong.";
  throw new Error(message);
}

/**
 * Renders the enrolment QR as inline SVG, here in the main process.
 *
 * Generated locally rather than through a QR-code web service, because the
 * thing being encoded is the TOTP shared secret — handing it to a third party
 * to draw would defeat the second factor it is meant to create. Optional: if
 * the library isn't installed, the screen falls back to the setup key, which
 * every authenticator app can take by hand.
 */
async function qrCode(text: string): Promise<string | null> {
  try {
    const { toString } = await import("qrcode");
    return await toString(text, { type: "svg", margin: 1, width: 190 });
  } catch {
    return null;
  }
}

/* ── lifecycle ──────────────────────────────────────────────────────────── */

app.whenReady().then(async () => {
  if (!isPrimaryInstance) return; // another copy already owns the microphone
  lockDownPermissions();
  consentStore.load();

  win = createWindow();
  createTray();

  if (!globalShortcut.register(HOTKEY, () => void triggerListen())) {
    send("engine:error", `Could not register the ${HOTKEY} hotkey — another app has it.`);
  }

  // ── engine + window ───────────────────────────────────────────────────
  ipcMain.handle("engine:info", () => ({
    url: engine.url,
    ws: engine.wsUrl,
    token: engine.apiToken,
    running: engine.running,
    hotkey: HOTKEY,
    logPath: engine.logPath,
  }));
  ipcMain.handle("engine:listen", () => triggerListen());
  // Minimise and hide are genuinely different for an assistant: minimising
  // parks it in the taskbar, hiding tucks it into the tray. Both keep it
  // listening; only "turn off" stops that.
  ipcMain.handle("window:minimize", () => win?.minimize());
  ipcMain.handle("window:hide", () => win?.hide());
  // The HUD's own "turn off" button. `quitting` is what tells the close
  // handler to stop hiding to the tray and actually let the app go.
  ipcMain.handle("app:quit", () => {
    quitting = true;
    app.quit();
  });
  ipcMain.handle("window:pin", (_e, pinned: boolean) => {
    win?.setAlwaysOnTop(Boolean(pinned));
    return Boolean(pinned);
  });
  ipcMain.handle("app:openLog", () => shell.openPath(engine.logPath));

  // ── sign-in ───────────────────────────────────────────────────────────
  ipcMain.handle("auth:config", () => ({
    configured: firebaseConfigured,
    required: requireAuth,
    twoFactor: twoFactorRequired,
  }));
  ipcMain.handle("auth:status", () => sessionManager.status());
  ipcMain.handle("auth:signIn", async (_e, email: string, password: string) => {
    try {
      return await afterAuthChange(await sessionManager.signIn(email, password));
    } catch (err) {
      authError(err);
    }
  });
  ipcMain.handle("auth:signUp", async (_e, email: string, password: string) => {
    try {
      return await afterAuthChange(await sessionManager.signUp(email, password));
    } catch (err) {
      authError(err);
    }
  });
  ipcMain.handle("auth:submitCode", async (_e, code: string) => {
    try {
      return await afterAuthChange(await sessionManager.submitCode(code));
    } catch (err) {
      authError(err);
    }
  });
  ipcMain.handle("auth:beginTotp", async () => {
    try {
      const enrollment = await sessionManager.beginTotpEnrollment();
      return { ...enrollment, qrSvg: await qrCode(enrollment.uri) };
    } catch (err) {
      authError(err);
    }
  });
  ipcMain.handle("auth:resendVerification", async () => {
    try {
      await sessionManager.resendVerification();
    } catch (err) {
      authError(err);
    }
  });
  ipcMain.handle("auth:resetPassword", async (_e, email: string) => {
    try {
      await sessionManager.resetPassword(email);
    } catch (err) {
      authError(err);
    }
  });
  ipcMain.handle("auth:recheck", async () =>
    afterAuthChange(await sessionManager.recheck())
  );
  ipcMain.handle("auth:signOut", () => doSignOut());

  // ── permissions ───────────────────────────────────────────────────────
  ipcMain.handle("consent:list", () => ({
    capabilities: CAPABILITIES.map((c) => ({
      ...c,
      os: c.osPermission ? osAccess(c.osPermission) : null,
    })),
    granted: consentStore.all(),
    needsAsking: consentStore.needsAsking,
  }));
  ipcMain.handle("consent:save", async (_e, granted: Partial<Grants>) => {
    const saved = consentStore.save(granted);
    await syncEngineState();
    await syncScreen();
    buildTrayMenu();
    return saved;
  });
  ipcMain.handle("consent:openOsSettings", (_e, kind: "microphone" | "camera") =>
    openOsSettings(kind)
  );
  ipcMain.handle("consent:osStatus", (_e, kind: "microphone" | "camera") =>
    osAccess(kind)
  );

  // ── the AI model and the speech recogniser ────────────────────────────
  //
  // Note what does *not* cross this boundary: the key itself. The renderer can
  // set one and can ask whether one exists, but there is no handler that
  // returns it.
  ipcMain.handle("ai:info", async () => {
    const info = engineReady
      ? await engine.llmInfo<Record<string, unknown>>().catch(() => ({}))
      : {};
    return { ...info, stored: keyStore.summary() };
  });

  ipcMain.handle(
    "ai:setLlm",
    async (_e, provider: string, model: string, apiKey?: string) => {
      const choice = keyStore.setLlm(provider, model, apiKey);
      if (engineReady) {
        await engine.pushLlm(choice.provider, choice.model, choice.apiKey);
      }
      return keyStore.summary();
    }
  );

  ipcMain.handle("ai:setSpeech", async (_e, provider: string, apiKey?: string) => {
    const choice = keyStore.setSpeech(provider, apiKey);
    if (engineReady) await engine.pushSpeech(choice.provider, choice.apiKey);
    return keyStore.summary();
  });

  ipcMain.handle(
    "ai:validate",
    async (_e, provider: string, apiKey: string, model?: string) => {
      try {
        await engine.validateKey(provider, apiKey, model ?? "");
        return { ok: true };
      } catch (err) {
        return { ok: false, error: (err as Error).message };
      }
    }
  );

  ipcMain.handle("ai:models", async (_e, provider: string, apiKey?: string) => {
    // An empty key means "use the one already stored", so the model list can be
    // refreshed without the renderer ever holding the secret.
    const key = apiKey || keyStore.llmChoice().apiKey;
    if (!key) return { models: [] };
    try {
      return { models: await engine.listModels(provider, key) };
    } catch (err) {
      return { models: [], error: (err as Error).message };
    }
  });

  ipcMain.handle("ai:forget", async () => {
    keyStore.forget();
    if (engineReady) {
      await engine.pushLlm("anthropic", "", "");
      await engine.pushSpeech("", "");
    }
    return keyStore.summary();
  });

  sessionManager.on("change", (status: AuthStatus) => {
    send("auth:status", status);
    buildTrayMenu();
    void syncEngineState();
    void syncScreen();
  });

  // The first screen is decided before anything is shown, so nobody ever sees
  // the HUD flash past on the way to a sign-in prompt.
  await sessionManager.restore();
  // Not forced: restoring emits a change, whose handler has usually already
  // started loading this very screen. Forcing here loaded the same page twice
  // on every launch. Awaiting it still guarantees a screen is settled before
  // the engine boots.
  await showScreen(requiredScreen());
  buildTrayMenu();

  // Ask Windows about the microphone once, up front: if privacy settings block
  // it, the stream opens and delivers silence forever, which reads to a user
  // as "it just doesn't hear me".
  if (systemPreferences.getMediaAccessStatus("microphone") === "denied") {
    send(
      "engine:error",
      "Windows is blocking microphone access for apps. Open Settings → " +
        "Privacy & security → Microphone to allow it."
    );
  }

  await boot();
});

app.on("before-quit", () => {
  quitting = true;
  engine.stop();
});

app.on("will-quit", () => globalShortcut.unregisterAll());

// The tray keeps the app alive after the window closes; this is the point of
// a background assistant, so don't quit on last-window-closed.
app.on("window-all-closed", () => {
  if (process.platform !== "darwin" && quitting) app.quit();
});
