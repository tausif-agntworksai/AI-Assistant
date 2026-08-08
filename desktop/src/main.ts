/**
 * Electron main process: the HUD window, the tray icon, and the global hotkey.
 *
 * The window is deliberately frameless and always-on-top — an assistant you
 * summon with a hotkey shouldn't make you hunt for it in the taskbar.
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
  shell,
  Tray,
} from "electron";
import { Engine, missingPieces } from "./engine";

const HOTKEY = "Control+Alt+J";
const WIDTH = 400;
const HEIGHT = 560;
const MARGIN = 24;

const engine = new Engine();
let win: BrowserWindow | null = null;
let tray: Tray | null = null;
let quitting = false;

// A single instance owns the microphone and the port; a second would fight it
// for both and fail in confusing ways.
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => showWindow());
}

function iconPath(): string {
  return path.join(app.getAppPath(), "assets", "icon.png");
}

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
      sandbox: false,
    },
  });

  window.loadFile(path.join(app.getAppPath(), "renderer", "index.html"));

  // `--hidden` is what the run-at-login entry passes: start listening in the
  // tray without taking focus at sign-in.
  const startHidden = process.argv.includes("--hidden");
  window.once("ready-to-show", () => {
    if (!startHidden) window.show();
  });

  // Closing hides to the tray; the assistant is meant to keep listening.
  window.on("close", (event) => {
    if (quitting) return;
    event.preventDefault();
    window.hide();
  });

  // Links open in the real browser, never inside the HUD.
  window.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  return window;
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
  tray.setContextMenu(
    Menu.buildFromTemplate([
      { label: "Show Jarvis", click: () => showWindow() },
      { label: `Listen now (${HOTKEY})`, click: () => triggerListen() },
      { type: "separator" },
      {
        label: "Start with Windows",
        type: "checkbox",
        checked: launchesAtLogin(),
        click: (item) => setLaunchAtLogin(item.checked),
      },
      { type: "separator" },
      { label: "Open engine log", click: () => shell.openPath(engine.logPath) },
      { label: "Open data folder", click: () => shell.openPath(app.getPath("logs")) },
      { type: "separator" },
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

async function triggerListen(): Promise<void> {
  try {
    await fetch(`${engine.url}/listen`, { method: "POST" });
    showWindow();
  } catch (err) {
    send("engine:error", `Could not reach the engine: ${(err as Error).message}`);
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
    send("engine:status", { state: "ready", url: engine.url, ws: engine.wsUrl });
  } catch (err) {
    const message = (err as Error).message;
    send("engine:status", { state: "failed", message });
    dialog.showErrorBox(
      "Jarvis engine failed to start",
      `${message}\n\nLast output:\n${engine.tail || "(nothing)"}\n\nLog: ${engine.logPath}`
    );
  }
}

app.whenReady().then(async () => {
  win = createWindow();
  createTray();

  if (!globalShortcut.register(HOTKEY, () => triggerListen())) {
    send("engine:error", `Could not register the ${HOTKEY} hotkey — another app has it.`);
  }

  ipcMain.handle("engine:info", () => ({
    url: engine.url,
    ws: engine.wsUrl,
    running: engine.running,
    hotkey: HOTKEY,
    logPath: engine.logPath,
  }));
  ipcMain.handle("engine:listen", () => triggerListen());
  ipcMain.handle("window:hide", () => win?.hide());
  ipcMain.handle("window:pin", (_e, pinned: boolean) => {
    win?.setAlwaysOnTop(Boolean(pinned));
    return Boolean(pinned);
  });
  ipcMain.handle("app:openLog", () => shell.openPath(engine.logPath));

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
