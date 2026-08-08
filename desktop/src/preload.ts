/**
 * The only bridge between the HUD page and Node.
 *
 * Context isolation stays on and nothing from Node leaks into the page — the
 * renderer gets this fixed set of calls and nothing else.
 */
import { contextBridge, ipcRenderer } from "electron";

export interface EngineInfo {
  url: string;
  ws: string;
  running: boolean;
  hotkey: string;
  logPath: string;
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
