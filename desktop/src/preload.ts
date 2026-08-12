/**
 * The only bridge between the app's pages and Node.
 *
 * Context isolation stays on, the renderer is sandboxed, and nothing from Node
 * leaks into the page — every page gets this fixed set of calls and nothing
 * else. In particular the pages never see the Firebase refresh token: they ask
 * the main process to sign in and are told which screen to draw.
 *
 * The engine token *is* handed over, because the HUD talks to the engine's
 * localhost API directly. That is a deliberate line: our own page may hold it,
 * a page on some website can't reach this bridge at all.
 */
import { contextBridge, ipcRenderer } from "electron";

export interface EngineInfo {
  url: string;
  ws: string;
  token: string;
  running: boolean;
  hotkey: string;
  logPath: string;
}

export interface AuthStatus {
  state:
    | "loading"
    | "unconfigured"
    | "signed-out"
    | "needs-verification"
    | "needs-approval"
    | "rejected"
    | "needs-2fa"
    | "offline-expired"
    | "ready";
  email: string;
  uid: string;
  isAdmin: boolean;
  online: boolean;
  message?: string;
}

contextBridge.exposeInMainWorld("jarvis", {
  info: (): Promise<EngineInfo> => ipcRenderer.invoke("engine:info"),
  listen: (): Promise<void> => ipcRenderer.invoke("engine:listen"),
  hide: (): Promise<void> => ipcRenderer.invoke("window:hide"),
  quit: (): Promise<void> => ipcRenderer.invoke("app:quit"),
  pin: (pinned: boolean): Promise<boolean> => ipcRenderer.invoke("window:pin", pinned),
  openLog: (): Promise<string> => ipcRenderer.invoke("app:openLog"),

  onStatus: (cb: (payload: unknown) => void) => {
    ipcRenderer.on("engine:status", (_e, payload) => cb(payload));
  },
  onLog: (cb: (line: string) => void) => {
    ipcRenderer.on("engine:log", (_e, line) => cb(line));
  },
  onError: (cb: (message: string) => void) => {
    ipcRenderer.on("engine:error", (_e, message) => cb(message));
  },
});

contextBridge.exposeInMainWorld("auth", {
  config: () => ipcRenderer.invoke("auth:config"),
  status: (): Promise<AuthStatus> => ipcRenderer.invoke("auth:status"),
  signIn: (email: string, password: string): Promise<AuthStatus> =>
    ipcRenderer.invoke("auth:signIn", email, password),
  signUp: (email: string, password: string): Promise<AuthStatus> =>
    ipcRenderer.invoke("auth:signUp", email, password),
  submitCode: (code: string): Promise<AuthStatus> =>
    ipcRenderer.invoke("auth:submitCode", code),
  beginTotp: (): Promise<{ sharedSecretKey: string; uri: string }> =>
    ipcRenderer.invoke("auth:beginTotp"),
  resendVerification: (): Promise<void> => ipcRenderer.invoke("auth:resendVerification"),
  resetPassword: (email: string): Promise<void> =>
    ipcRenderer.invoke("auth:resetPassword", email),
  recheck: (): Promise<AuthStatus> => ipcRenderer.invoke("auth:recheck"),
  signOut: (): Promise<AuthStatus> => ipcRenderer.invoke("auth:signOut"),
  onChange: (cb: (status: AuthStatus) => void) => {
    ipcRenderer.on("auth:status", (_e, status) => cb(status));
  },
});

contextBridge.exposeInMainWorld("consent", {
  list: () => ipcRenderer.invoke("consent:list"),
  save: (granted: Record<string, boolean>) => ipcRenderer.invoke("consent:save", granted),
  openOsSettings: (kind: "microphone" | "camera") =>
    ipcRenderer.invoke("consent:openOsSettings", kind),
  osStatus: (kind: "microphone" | "camera") => ipcRenderer.invoke("consent:osStatus", kind),
});
