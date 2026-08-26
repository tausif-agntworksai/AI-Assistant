/**
 * The user's own API keys: stored here, encrypted, and never given to a page.
 *
 * The sibling AI Calculator keeps its key in `localStorage`, which is a
 * reasonable choice for a browser app and a poor one for a desktop app that
 * already has somewhere better. Three differences follow from using
 * `safeStorage` — DPAPI on Windows, Keychain on macOS — instead:
 *
 *   * **The renderer never sees the key.** It asks whether one exists and gets
 *     `hasKey: true`. A bug in the settings screen cannot read it out, and it
 *     never appears in a DevTools inspector or a page-level crash report.
 *   * **It survives sign-out.** The calculator wipes its key on logout so it
 *     can't outlive a session in a shared browser profile; a file encrypted to
 *     one Windows account has no such problem, and making someone re-paste a
 *     key every morning is a real cost for no gain.
 *   * **If the OS declines to encrypt, nothing is written at all.** A plaintext
 *     API key on disk is worse than retyping one tomorrow.
 *
 * Scoped per Firebase uid, so two accounts on one machine never see each
 * other's key.
 */

import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { app, safeStorage } from "electron";

/** What the engine needs to talk to a model provider. */
export interface LlmChoice {
  provider: string;
  model: string;
  apiKey: string;
}

/** The optional cloud speech recogniser. */
export interface SpeechChoice {
  provider: string;
  apiKey: string;
}

/** Safe to hand to the renderer: what is configured, never the secret. */
export interface KeySummary {
  provider: string;
  model: string;
  hasKey: boolean;
}

interface StoredKeys {
  version: number;
  llm?: LlmChoice;
  speech?: SpeechChoice;
}

const VERSION = 1;
const EMPTY_LLM: LlmChoice = { provider: "anthropic", model: "", apiKey: "" };
const EMPTY_SPEECH: SpeechChoice = { provider: "", apiKey: "" };

export class KeyStore {
  private uid = "";
  private llm: LlmChoice = { ...EMPTY_LLM };
  private speech: SpeechChoice = { ...EMPTY_SPEECH };

  /** One file per account. `local` covers an unconfigured-Firebase build. */
  private file(uid = this.uid): string {
    const safe = (uid || "local").replace(/[^A-Za-z0-9_-]/g, "");
    return path.join(app.getPath("userData"), `keys-${safe}.bin`);
  }

  /** Called on every sign-in, and on sign-out with an empty uid. */
  load(uid: string): void {
    this.uid = uid || "local";
    this.llm = { ...EMPTY_LLM };
    this.speech = { ...EMPTY_SPEECH };

    if (!existsSync(this.file()) || !safeStorage.isEncryptionAvailable()) return;
    try {
      const parsed = JSON.parse(
        safeStorage.decryptString(readFileSync(this.file()))
      ) as StoredKeys;
      if (parsed.llm?.provider) this.llm = { ...EMPTY_LLM, ...parsed.llm };
      if (parsed.speech?.provider) this.speech = { ...EMPTY_SPEECH, ...parsed.speech };
    } catch {
      // Wrong Windows account, a rotated DPAPI key, or a corrupt file — it can
      // never be decrypted again, so don't leave it lying around pretending.
      this.forget();
    }
  }

  /* ── reading ─────────────────────────────────────────────────────────── */

  llmChoice(): LlmChoice {
    return { ...this.llm };
  }

  speechChoice(): SpeechChoice {
    return { ...this.speech };
  }

  /** What the settings screen is allowed to know. */
  summary(): { llm: KeySummary; speech: KeySummary } {
    return {
      llm: {
        provider: this.llm.provider,
        model: this.llm.model,
        hasKey: Boolean(this.llm.apiKey),
      },
      speech: {
        provider: this.speech.provider,
        model: "",
        hasKey: Boolean(this.speech.apiKey),
      },
    };
  }

  /* ── writing ─────────────────────────────────────────────────────────── */

  /**
   * `apiKey === undefined` keeps the key already stored, so changing model
   * doesn't ask for it again. `""` removes it.
   */
  setLlm(provider: string, model: string, apiKey?: string): LlmChoice {
    const switchingProvider = provider !== this.llm.provider;
    this.llm = {
      provider,
      model,
      // A key belongs to one provider. Carrying it across a provider change
      // would send an OpenAI key to Groq, which reads back as "that key was
      // rejected" and sends the user hunting for the wrong problem.
      apiKey: apiKey !== undefined ? apiKey : switchingProvider ? "" : this.llm.apiKey,
    };
    this.persist();
    return this.llmChoice();
  }

  setSpeech(provider: string, apiKey?: string): SpeechChoice {
    const switchingProvider = provider !== this.speech.provider;
    this.speech = {
      provider,
      apiKey: apiKey !== undefined ? apiKey : switchingProvider ? "" : this.speech.apiKey,
    };
    this.persist();
    return this.speechChoice();
  }

  /** Removes every key for this account. Used by "remove key" and on delete. */
  forget(): void {
    this.llm = { ...EMPTY_LLM };
    this.speech = { ...EMPTY_SPEECH };
    try {
      rmSync(this.file(), { force: true });
    } catch {
      /* the in-memory clear is what matters this session */
    }
  }

  private persist(): void {
    if (!safeStorage.isEncryptionAvailable()) {
      console.warn(
        "OS encryption unavailable — API keys will not be remembered after quit."
      );
      return;
    }
    const payload: StoredKeys = { version: VERSION, llm: this.llm, speech: this.speech };
    try {
      mkdirSync(path.dirname(this.file()), { recursive: true });
      writeFileSync(this.file(), safeStorage.encryptString(JSON.stringify(payload)), {
        mode: 0o600,
      });
    } catch (err) {
      console.warn("Could not save API keys:", (err as Error).message);
    }
  }
}

export const keyStore = new KeyStore();
