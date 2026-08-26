/**
 * Typed access to what the preload script put on `window`.
 *
 * The preload already exports the interfaces it uses, so these are structural
 * copies rather than a second source of truth — the shapes are checked against
 * the real IPC handlers by the main process's own tsc pass.
 *
 * Note what is missing: any call that returns an API key. The main process can
 * be told to set one and can be asked whether one exists, and that is the whole
 * surface. A bug on this side cannot read a secret it was never given.
 */

export type GateState =
  | "loading"
  | "unconfigured"
  | "signed-out"
  | "needs-verification"
  | "needs-2fa"
  | "offline-expired"
  | "ready";

export interface AuthStatus {
  state: GateState;
  email: string;
  uid: string;
  online: boolean;
  codeReason: "sign-in" | "enroll" | null;
  message?: string;
}

export interface EngineInfo {
  url: string;
  ws: string;
  token: string;
  running: boolean;
  hotkey: string;
  logPath: string;
}

export interface CapabilitySpec {
  id: string;
  title: string;
  titleHi: string;
  what: string;
  whatHi: string;
  examples: string;
  required?: boolean;
  granted: boolean;
  unused?: boolean;
  osPermission?: "microphone" | "camera";
  os: "granted" | "denied" | "restricted" | "not-determined" | "unknown" | null;
}

export interface ModelOption {
  id: string;
  label: string;
  speed: string;
  supports_tools: boolean;
}

export interface ProviderMeta {
  id: string;
  label: string;
  help_url: string;
  default_model: string;
  known_models: ModelOption[];
}

export interface SpeechProviderMeta {
  id: string;
  label: string;
  help_url: string;
  note: string;
}

export interface KeySummary {
  provider: string;
  model: string;
  hasKey: boolean;
}

export interface AiInfo {
  providers?: ProviderMeta[];
  speech_providers?: SpeechProviderMeta[];
  selected?: { provider: string; label: string; model: string; has_key: boolean };
  speech?: { provider: string; label: string; has_key: boolean };
  stored: { llm: KeySummary; speech: KeySummary };
}

type Disposer = () => void;

interface JarvisBridge {
  info(): Promise<EngineInfo>;
  listen(): Promise<void>;
  minimize(): Promise<void>;
  hide(): Promise<void>;
  quit(): Promise<void>;
  pin(pinned: boolean): Promise<boolean>;
  openLog(): Promise<string>;
  onStatus(cb: (payload: unknown) => void): Disposer;
  onLog(cb: (line: string) => void): Disposer;
  onError(cb: (message: string) => void): Disposer;
}

interface AuthBridge {
  config(): Promise<{ configured: boolean; required: boolean; twoFactor: boolean }>;
  status(): Promise<AuthStatus>;
  signIn(email: string, password: string): Promise<AuthStatus>;
  signUp(email: string, password: string): Promise<AuthStatus>;
  submitCode(code: string): Promise<AuthStatus>;
  beginTotp(): Promise<{ sharedSecretKey: string; uri: string; qrSvg: string | null }>;
  resendVerification(): Promise<void>;
  resetPassword(email: string): Promise<void>;
  recheck(): Promise<AuthStatus>;
  signOut(): Promise<AuthStatus>;
  onChange(cb: (status: AuthStatus) => void): Disposer;
}

interface AiBridge {
  info(): Promise<AiInfo>;
  setLlm(provider: string, model: string, apiKey?: string): Promise<AiInfo["stored"]>;
  setSpeech(provider: string, apiKey?: string): Promise<AiInfo["stored"]>;
  validate(
    provider: string,
    apiKey: string,
    model?: string
  ): Promise<{ ok: boolean; error?: string; model?: string }>;
  models(
    provider: string,
    apiKey?: string
  ): Promise<{ models: ModelOption[]; error?: string }>;
  forget(): Promise<AiInfo["stored"]>;
}

interface ConsentBridge {
  list(): Promise<{
    capabilities: CapabilitySpec[];
    granted: Record<string, boolean>;
    needsAsking: boolean;
  }>;
  save(granted: Record<string, boolean>): Promise<Record<string, boolean>>;
  openOsSettings(kind: "microphone" | "camera"): Promise<void>;
  osStatus(kind: "microphone" | "camera"): Promise<string>;
}

declare global {
  interface Window {
    jarvis: JarvisBridge;
    auth: AuthBridge;
    ai: AiBridge;
    consent: ConsentBridge;
  }
}

export const jarvis = window.jarvis;
export const auth = window.auth;
export const ai = window.ai;
export const consent = window.consent;

/**
 * Strips the wrapper Electron puts around an error thrown inside an IPC
 * handler: "Error invoking remote method 'auth:signIn': Error: <the real one>".
 * Showing that to a user would be showing them our channel names.
 */
export function cleanError(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err ?? "");
  return (
    message.replace(/^Error invoking remote method '[^']+':\s*(Error:\s*)?/, "") ||
    "Something went wrong."
  );
}
