/**
 * Who is signed in, whether they may use the assistant, and keeping it true.
 *
 * The gate is the same one the AI Calculator enforces, in the same order:
 *
 *     signed in → verified email → admin approval → [authenticator] → ready
 *
 * What is different is what "ready" buys. In the calculator it unlocks a page.
 * Here it unlocks the microphone: the engine boots with its listening loop shut
 * and only opens it once `unlock` reaches it (see `engine/jarvis/security.py`).
 * A sign-in screen you can't get past but that leaves the machine listening
 * would be theatre, so this one holds the microphone closed.
 *
 * Two storage decisions:
 *
 *   **The refresh token is encrypted with Electron `safeStorage`**, which on
 *   Windows is DPAPI keyed to this Windows account. Someone with the file but
 *   not the account cannot read it. If the OS declines to provide encryption,
 *   nothing is written at all — a plaintext refresh token on disk is worse than
 *   signing in again tomorrow.
 *
 *   **Every launch refreshes against Firebase.** That is what makes a token
 *   read from disk trustworthy, and it is where a disabled or revoked account
 *   stops working. Offline, the last verified session is reused for
 *   `OFFLINE_GRACE_MS` — stated in `config.ts` as the trade it is.
 */

import { EventEmitter } from "node:events";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { app, safeStorage } from "electron";
import {
  AuthError,
  completeMfaSignIn,
  finishTotpEnrollment,
  lookup,
  readClaims,
  refresh,
  sendPasswordReset,
  sendVerificationEmail,
  signIn,
  signUp,
  startTotpEnrollment,
  type MfaChallenge,
  type TokenClaims,
  type Tokens,
  type TotpEnrollment,
} from "./firebaseAuth";
import { recordOwnAccessRequest } from "./accessRequests";
import {
  OFFLINE_GRACE_MS,
  firebaseConfigured,
  isAdminEmail,
  requireAuth,
  twoFactorEnabled,
} from "../config";

/** Every screen the sign-in flow can be on. The HUD renders one per value. */
export type GateState =
  | "loading"
  | "unconfigured"
  | "signed-out"
  | "needs-verification"
  | "needs-approval"
  | "rejected"
  | "needs-2fa"
  | "offline-expired"
  | "ready";

export interface AuthStatus {
  state: GateState;
  email: string;
  uid: string;
  isAdmin: boolean;
  /** True when the last check reached Firebase rather than the cached session. */
  online: boolean;
  /**
   * Why a 6-digit code is being asked for. "needs-2fa" covers two different
   * screens — finishing a sign-in versus setting the authenticator up for the
   * first time — and they say very different things to the person reading them.
   */
  codeReason: "sign-in" | "enroll" | null;
  message?: string;
}

interface StoredSession {
  refreshToken: string;
  email: string;
  uid: string;
  /** Unix ms of the last successful online verification. */
  verifiedAt: number;
}

/** Refresh a little before the hour is up, so a command never hits an expiry. */
const REFRESH_MARGIN_MS = 5 * 60 * 1000;
const MIN_REFRESH_MS = 60 * 1000;

export class SessionManager extends EventEmitter {
  private tokens: Tokens | null = null;
  private claims: TokenClaims | null = null;
  private stored: StoredSession | null = null;
  private timer: NodeJS.Timeout | null = null;
  private challenge: MfaChallenge | null = null;
  private enrollment: TotpEnrollment | null = null;
  private hasSecondFactor = false;
  private online = false;

  state: GateState = "loading";
  message = "";

  private get file(): string {
    return path.join(app.getPath("userData"), "session.bin");
  }

  /* ── status ───────────────────────────────────────────────────────────── */

  status(): AuthStatus {
    return {
      state: this.state,
      email: this.claims?.email ?? this.stored?.email ?? "",
      uid: this.claims?.uid ?? this.stored?.uid ?? "",
      isAdmin: Boolean(this.claims?.admin) || isAdminEmail(this.claims?.email),
      online: this.online,
      codeReason:
        this.state !== "needs-2fa" ? null : this.challenge ? "sign-in" : "enroll",
      message: this.message || undefined,
    };
  }

  /** The valid ID token, or null. Refreshes first if it is close to expiring. */
  async idToken(): Promise<string | null> {
    if (!this.tokens) return null;
    if (Date.now() > this.tokens.expiresAt - MIN_REFRESH_MS) {
      await this.refreshNow().catch(() => undefined);
    }
    return this.tokens?.idToken ?? null;
  }

  get ready(): boolean {
    return this.state === "ready";
  }

  /** Seconds the engine's unlock should stand for — one ID token's lifetime. */
  get unlockTtlSec(): number {
    if (!this.tokens) return 0;
    return Math.max(60, Math.floor((this.tokens.expiresAt - Date.now()) / 1000));
  }

  private setState(state: GateState, message = ""): void {
    const changed = state !== this.state || message !== this.message;
    this.state = state;
    this.message = message;
    if (changed) this.emit("change", this.status());
  }

  /* ── startup ──────────────────────────────────────────────────────────── */

  async restore(): Promise<AuthStatus> {
    if (!firebaseConfigured || !requireAuth) {
      this.setState("unconfigured");
      return this.status();
    }

    this.stored = this.read();
    if (!this.stored) {
      this.setState("signed-out");
      return this.status();
    }

    try {
      this.tokens = await refresh(this.stored.refreshToken);
      this.online = true;
      await this.evaluate();
    } catch (err) {
      if (err instanceof AuthError && err.code === "network") {
        // Offline. Reuse the last verified session, within the stated window.
        this.online = false;
        const age = Date.now() - this.stored.verifiedAt;
        if (age <= OFFLINE_GRACE_MS) {
          this.setState(
            "ready",
            "Working offline — Jarvis will re-check your account when you're back online."
          );
        } else {
          this.setState(
            "offline-expired",
            "Your session needs re-checking with Firebase, and Jarvis can't reach it."
          );
        }
      } else {
        // A revoked, disabled or deleted account lands here. Nothing to reuse.
        this.forget();
        this.setState("signed-out", err instanceof Error ? err.message : undefined);
      }
    }

    this.scheduleRefresh();
    return this.status();
  }

  /* ── the gate ─────────────────────────────────────────────────────────── */

  /**
   * Works out which screen the user belongs on, from the freshest facts
   * available. Called after every token change.
   */
  private async evaluate(): Promise<void> {
    const token = this.tokens;
    if (!token?.idToken) {
      this.setState("signed-out");
      return;
    }

    this.claims = readClaims(token.idToken);
    const email = this.claims?.email ?? "";

    // The token's `email_verified` is minted at sign-in, so someone who has
    // just clicked the link still carries a stale `false`. Ask directly.
    let verified = Boolean(this.claims?.emailVerified);
    try {
      const account = await lookup(token.idToken);
      verified = account.emailVerified;
      this.hasSecondFactor = account.mfaFactors.length > 0;
      this.online = true;
    } catch {
      /* offline — the claim is the best we have */
    }

    if (!verified) {
      this.setState("needs-verification");
      return;
    }

    // Remember the session from here on, not only once the gate opens. Someone
    // waiting on approval shouldn't have to sign in again to press "check
    // again" tomorrow — and a stored session still passes through every check
    // below on the next launch.
    this.persist(token, email);

    const admin = Boolean(this.claims?.admin) || isAdminEmail(email);
    if (!admin) {
      const approved = await this.checkApproval(token.idToken, email);
      if (approved === "rejected") {
        this.setState("rejected");
        return;
      }
      if (approved !== "approved") {
        this.setState("needs-approval");
        return;
      }
    }

    if (twoFactorEnabled && !this.hasSecondFactor) {
      this.setState("needs-2fa");
      return;
    }

    this.setState("ready");
  }

  /**
   * Approval status, claim first and Firestore second.
   *
   * The `approved` custom claim is authoritative — only a credentialed backend
   * can set it — but it only lands on a *newly minted* token, so someone
   * approved five minutes ago still carries the old one. Firestore has the
   * live answer, and a fresh refresh then picks the claim up properly.
   */
  private async checkApproval(
    idToken: string,
    email: string
  ): Promise<"approved" | "pending" | "rejected" | "unknown"> {
    if (this.claims?.approved) return "approved";

    const filed = await recordOwnAccessRequest(this.claims?.uid ?? "", email, idToken);
    if (filed === "approved") {
      // Firestore says yes but our token predates the decision. Mint a new one
      // so the claim is there for everything downstream.
      try {
        this.tokens = await refresh(this.tokens!.refreshToken);
        this.claims = readClaims(this.tokens.idToken);
      } catch {
        /* the Firestore answer still stands for this session */
      }
      return "approved";
    }
    return filed ?? "unknown";
  }

  /* ── sign-in flow ─────────────────────────────────────────────────────── */

  async signUp(email: string, password: string): Promise<AuthStatus> {
    this.tokens = await signUp(email.trim(), password);
    this.online = true;
    await this.evaluate();
    this.scheduleRefresh();
    return this.status();
  }

  async signIn(email: string, password: string): Promise<AuthStatus> {
    const result = await signIn(email.trim(), password);
    this.online = true;
    if (result.status === "mfa-required") {
      this.challenge = result.challenge;
      this.setState("needs-2fa", "Enter the code from your authenticator app.");
      return this.status();
    }
    this.challenge = null;
    this.tokens = result.tokens;
    await this.evaluate();
    this.scheduleRefresh();
    return this.status();
  }

  /** The 6-digit code — either finishing a sign-in or finishing enrolment. */
  async submitCode(code: string): Promise<AuthStatus> {
    const trimmed = code.replace(/\s+/g, "");
    if (this.challenge) {
      this.tokens = await completeMfaSignIn(this.challenge, trimmed);
      this.challenge = null;
      this.hasSecondFactor = true;
    } else if (this.enrollment && this.tokens) {
      this.tokens = await finishTotpEnrollment(
        this.tokens.idToken,
        this.enrollment.sessionInfo,
        trimmed
      );
      this.enrollment = null;
      this.hasSecondFactor = true;
    } else {
      throw new AuthError("no-challenge", "There's no code to enter right now.");
    }
    await this.evaluate();
    this.scheduleRefresh();
    return this.status();
  }

  /** Starts TOTP enrolment and returns what the screen needs to draw. */
  async beginTotpEnrollment(): Promise<TotpEnrollment> {
    if (!this.tokens) throw new AuthError("signed-out", "Sign in first.");
    this.enrollment = await startTotpEnrollment(
      this.tokens.idToken,
      this.claims?.email ?? ""
    );
    return this.enrollment;
  }

  async resendVerification(): Promise<void> {
    if (!this.tokens) throw new AuthError("signed-out", "Sign in first.");
    await sendVerificationEmail(this.tokens.idToken);
  }

  async resetPassword(email: string): Promise<void> {
    await sendPasswordReset(email.trim());
  }

  /** Re-checks everything: used by the "I've verified" and "check again" buttons. */
  async recheck(): Promise<AuthStatus> {
    if (this.stored && !this.tokens) return this.restore();
    if (!this.tokens) return this.status();
    try {
      this.tokens = await refresh(this.tokens.refreshToken);
      this.online = true;
    } catch (err) {
      if (!(err instanceof AuthError && err.code === "network")) {
        this.forget();
        this.setState("signed-out");
        return this.status();
      }
      this.online = false;
    }
    await this.evaluate();
    this.scheduleRefresh();
    return this.status();
  }

  async signOut(): Promise<AuthStatus> {
    this.forget();
    this.tokens = null;
    this.claims = null;
    this.challenge = null;
    this.enrollment = null;
    this.hasSecondFactor = false;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    this.setState(firebaseConfigured && requireAuth ? "signed-out" : "unconfigured");
    return this.status();
  }

  /* ── keeping it fresh ─────────────────────────────────────────────────── */

  private scheduleRefresh(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    if (!this.tokens) return;

    const delay = Math.max(
      MIN_REFRESH_MS,
      this.tokens.expiresAt - Date.now() - REFRESH_MARGIN_MS
    );
    this.timer = setTimeout(() => void this.refreshNow(), delay);
    // Don't hold the process open for a token refresh.
    this.timer.unref?.();
  }

  private async refreshNow(): Promise<void> {
    if (!this.tokens) return;
    try {
      this.tokens = await refresh(this.tokens.refreshToken);
      this.online = true;
      await this.evaluate();
    } catch (err) {
      if (err instanceof AuthError && err.code === "network") {
        this.online = false;
      } else {
        // Revoked, disabled, or the password changed elsewhere. This is the
        // point where the assistant must stop listening.
        this.forget();
        this.tokens = null;
        this.setState("signed-out", "Your session ended. Please sign in again.");
      }
    }
    this.scheduleRefresh();
  }

  /* ── storage ──────────────────────────────────────────────────────────── */

  private persist(tokens: Tokens, email: string): void {
    const payload: StoredSession = {
      refreshToken: tokens.refreshToken,
      email,
      uid: tokens.uid || (this.claims?.uid ?? ""),
      verifiedAt: Date.now(),
    };
    this.stored = payload;

    if (!safeStorage.isEncryptionAvailable()) {
      // Deliberate: no OS keychain means no stored credential. Signing in again
      // is a smaller cost than a refresh token sitting in a readable file.
      console.warn(
        "OS encryption unavailable — this session won't be remembered after quit."
      );
      return;
    }
    try {
      mkdirSync(path.dirname(this.file), { recursive: true });
      writeFileSync(this.file, safeStorage.encryptString(JSON.stringify(payload)), {
        mode: 0o600,
      });
    } catch (err) {
      console.warn("Could not save the session:", (err as Error).message);
    }
  }

  private read(): StoredSession | null {
    if (!existsSync(this.file) || !safeStorage.isEncryptionAvailable()) return null;
    try {
      const parsed = JSON.parse(
        safeStorage.decryptString(readFileSync(this.file))
      ) as StoredSession;
      return parsed?.refreshToken ? parsed : null;
    } catch {
      // Wrong Windows account, a corrupt file, or a rotated DPAPI key. Either
      // way it can never be decrypted again, so don't leave it lying around.
      this.forget();
      return null;
    }
  }

  private forget(): void {
    this.stored = null;
    try {
      rmSync(this.file, { force: true });
    } catch {
      /* nothing to do — the session simply won't be reusable */
    }
  }
}

export const sessionManager = new SessionManager();
