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
import { contextBridge, ipcRenderer, type IpcRendererEvent } from "electron";

/**
 * Subscribes to a main-process channel and hands back a disposer.
 *
 * The disposer is the point. These used to return `undefined`, which is fine
 * for a page that lives until it navigates away — but a React effect's cleanup
 * has nothing to call, so every remount adds another listener and the same log
 * line arrives two, three, four times. Returning the unsubscribe function makes
 * `useEffect(() => bridge.onLog(...), [])` correct by construction.
 */
function subscribe<T>(channel: string, cb: (payload: T) => void): () => void {
  const handler = (_event: IpcRendererEvent, payload: T) => cb(payload);
  ipcRenderer.on(channel, handler);
  return () => ipcRenderer.removeListener(channel, handler);
}

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
    | "needs-2fa"
    | "offline-expired"
    | "ready";
  email: string;
  uid: string;
  online: boolean;
  codeReason: "sign-in" | "enroll" | null;
  message?: string;
}

contextBridge.exposeInMainWorld("jarvis", {
  info: (): Promise<EngineInfo> => ipcRenderer.invoke("engine:info"),
  listen: (): Promise<void> => ipcRenderer.invoke("engine:listen"),
  minimize: (): Promise<void> => ipcRenderer.invoke("window:minimize"),
  hide: (): Promise<void> => ipcRenderer.invoke("window:hide"),
  quit: (): Promise<void> => ipcRenderer.invoke("app:quit"),
  pin: (pinned: boolean): Promise<boolean> => ipcRenderer.invoke("window:pin", pinned),
  openLog: (): Promise<string> => ipcRenderer.invoke("app:openLog"),

  onStatus: (cb: (payload: unknown) => void) => subscribe("engine:status", cb),
  onLog: (cb: (line: string) => void) => subscribe<string>("engine:log", cb),
  onError: (cb: (message: string) => void) => subscribe<string>("engine:error", cb),
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
  onChange: (cb: (status: AuthStatus) => void) => subscribe<AuthStatus>("auth:status", cb),
});

export interface KeySummary {
  provider: string;
  model: string;
  hasKey: boolean;
}

export interface ModelOption {
  id: string;
  label: string;
  speed: string;
  supports_tools: boolean;
}

/**
 * The AI-model surface. Deliberately write-only for secrets: a key can be set
 * and can be checked, and there is no call anywhere here that reads one back.
 */
contextBridge.exposeInMainWorld("ai", {
  info: () => ipcRenderer.invoke("ai:info"),
  setLlm: (provider: string, model: string, apiKey?: string) =>
    ipcRenderer.invoke("ai:setLlm", provider, model, apiKey),
  setSpeech: (provider: string, apiKey?: string) =>
    ipcRenderer.invoke("ai:setSpeech", provider, apiKey),
  validate: (
    provider: string,
    apiKey: string,
    model?: string
  ): Promise<{ ok: boolean; error?: string; model?: string }> =>
    ipcRenderer.invoke("ai:validate", provider, apiKey, model),
  models: (
    provider: string,
    apiKey?: string
  ): Promise<{ models: ModelOption[]; error?: string }> =>
    ipcRenderer.invoke("ai:models", provider, apiKey),
  forget: () => ipcRenderer.invoke("ai:forget"),
});

contextBridge.exposeInMainWorld("consent", {
  list: () => ipcRenderer.invoke("consent:list"),
  save: (granted: Record<string, boolean>) => ipcRenderer.invoke("consent:save", granted),
  openOsSettings: (kind: "microphone" | "camera") =>
    ipcRenderer.invoke("consent:openOsSettings", kind),
  osStatus: (kind: "microphone" | "camera") => ipcRenderer.invoke("consent:osStatus", kind),
});
