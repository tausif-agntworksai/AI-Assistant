/**
 * Firebase Authentication over its REST API, from the Electron main process.
 *
 * Two decisions worth stating, because both are deliberate departures from how
 * the AI Calculator does the same job.
 *
 * **REST rather than the Firebase JS SDK.** This project has no bundler — the
 * HUD is a plain HTML file — and adding one to carry an auth library into a
 * renderer would be a lot of machinery for a login box. The REST surface is a
 * dozen JSON calls, needs nothing but `fetch`, and keeps the supply chain of a
 * process that can shut down the machine as short as it can be.
 *
 * **In the main process rather than the renderer.** The refresh token is the
 * long-lived credential — it outlives the ID token by design — and the page
 * never needs it. Keeping it here means the renderer holds no credential at
 * all: it asks main to sign in and is told yes or no.
 *
 * Nothing in this file decides whether someone may *use* the assistant. It
 * only establishes who they are; `session.ts` owns the gate.
 */

import { firebase } from "../config";

const IDENTITY = "https://identitytoolkit.googleapis.com/v1/accounts";
const IDENTITY_V2 = "https://identitytoolkit.googleapis.com/v2/accounts";
const SECURE_TOKEN = "https://securetoken.googleapis.com/v1/token";

/** Firebase is generous with timeouts; a sign-in box should not be. */
const TIMEOUT_MS = 20_000;

export interface Tokens {
  idToken: string;
  refreshToken: string;
  /** Unix ms at which `idToken` stops being accepted. */
  expiresAt: number;
  uid: string;
}

/** A sign-in that stopped to ask for the authenticator code. */
export interface MfaChallenge {
  pendingCredential: string;
  /** Enrolled factors; TOTP is the only kind this app enrolls. */
  factors: { id: string; displayName?: string }[];
}

export type SignInResult =
  | { status: "signed-in"; tokens: Tokens }
  | { status: "mfa-required"; challenge: MfaChallenge };

/** The claims this app reads out of an ID token. */
export interface TokenClaims {
  uid: string;
  email: string;
  emailVerified: boolean;
  /** Set by an admin through the AI Calculator's dashboard, on this project. */
  approved: boolean;
  admin: boolean;
  expiresAt: number;
}

export class AuthError extends Error {
  constructor(
    readonly code: string,
    message: string
  ) {
    super(message);
    this.name = "AuthError";
  }
}

/* ── transport ──────────────────────────────────────────────────────────── */

async function post<T>(url: string, body: unknown): Promise<T> {
  if (!firebase.apiKey) {
    throw new AuthError("not-configured", "Sign-in isn’t configured yet.");
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  let response: Response;
  try {
    response = await fetch(`${url}?key=${encodeURIComponent(firebase.apiKey)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    // A refused connection and an abort both mean the same thing to the user.
    throw new AuthError(
      "network",
      (err as Error).name === "AbortError"
        ? "Firebase didn’t answer in time."
        : "Couldn’t reach Firebase — check your connection."
    );
  } finally {
    clearTimeout(timer);
  }

  const payload = (await response.json().catch(() => ({}))) as Record<string, any>;
  if (!response.ok) {
    const code = String(payload?.error?.message ?? `HTTP_${response.status}`);
    throw new AuthError(code, friendlyMessage(code));
  }
  return payload as T;
}

/**
 * Firebase's REST errors are SHOUTING_SNAKE codes, sometimes with a colon and
 * a debug suffix. Every message a person can act on is written out here; the
 * rest fall through to something honest rather than a raw code.
 */
export function friendlyMessage(rawCode: string): string {
  const code = rawCode.split(":")[0]!.trim();
  switch (code) {
    case "EMAIL_EXISTS":
      return "An account already exists with this email. Try signing in.";
    case "EMAIL_NOT_FOUND":
    case "INVALID_PASSWORD":
    case "INVALID_LOGIN_CREDENTIALS":
      return "Incorrect email or password.";
    case "INVALID_EMAIL":
      return "That doesn’t look like a valid email address.";
    case "WEAK_PASSWORD":
      return "Password is too weak — use at least 6 characters.";
    case "USER_DISABLED":
      return "This account has been disabled by the administrator.";
    case "TOO_MANY_ATTEMPTS_TRY_LATER":
      return "Too many attempts. Wait a moment and try again.";
    case "INVALID_MFA_PENDING_CREDENTIAL":
    case "SESSION_EXPIRED":
      return "That took too long — sign in again.";
    case "INVALID_CODE":
    case "INVALID_TOTP_VERIFICATION_CODE":
      return "That code isn’t right. Check your authenticator app and try again.";
    case "TOKEN_EXPIRED":
    case "USER_NOT_FOUND":
    case "INVALID_REFRESH_TOKEN":
      return "Your session expired. Please sign in again.";
    case "SECOND_FACTOR_EXISTS":
      return "Two-step verification is already set up on this account.";
    case "OPERATION_NOT_ALLOWED":
      return "Email sign-in isn’t enabled on this Firebase project.";
    default:
      return `Sign-in failed (${code}).`;
  }
}

/* ── tokens ─────────────────────────────────────────────────────────────── */

function toTokens(payload: Record<string, any>): Tokens {
  const expiresIn = Number(payload.expiresIn ?? payload.expires_in ?? 3600);
  return {
    idToken: String(payload.idToken ?? payload.id_token ?? ""),
    refreshToken: String(payload.refreshToken ?? payload.refresh_token ?? ""),
    expiresAt: Date.now() + expiresIn * 1000,
    uid: String(payload.localId ?? payload.user_id ?? ""),
  };
}

/**
 * Reads the claims out of an ID token without verifying its signature.
 *
 * That sounds alarming and isn't, because of where the token comes from: every
 * token this app reads was fetched by this process over TLS directly from
 * Google, seconds earlier. Verifying our own response against Google's public
 * keys would be checking whether Google agrees with itself. What matters is
 * that a token from *disk* is never trusted on its own — `session.ts` refreshes
 * against Firebase on every launch, which is what re-establishes provenance.
 */
export function readClaims(idToken: string): TokenClaims | null {
  const parts = idToken.split(".");
  if (parts.length !== 3) return null;
  try {
    const json = Buffer.from(parts[1]!.replace(/-/g, "+").replace(/_/g, "/"), "base64")
      .toString("utf8");
    const claims = JSON.parse(json) as Record<string, any>;
    return {
      uid: String(claims.user_id ?? claims.sub ?? ""),
      email: String(claims.email ?? ""),
      emailVerified: claims.email_verified === true,
      approved: claims.approved === true,
      admin: claims.admin === true,
      expiresAt: Number(claims.exp ?? 0) * 1000,
    };
  } catch {
    return null;
  }
}

/* ── the calls ──────────────────────────────────────────────────────────── */

export async function signUp(email: string, password: string): Promise<Tokens> {
  const tokens = toTokens(
    await post(`${IDENTITY}:signUp`, { email, password, returnSecureToken: true })
  );
  await sendVerificationEmail(tokens.idToken);
  return tokens;
}

export async function signIn(email: string, password: string): Promise<SignInResult> {
  const payload = await post<Record<string, any>>(`${IDENTITY}:signInWithPassword`, {
    email,
    password,
    returnSecureToken: true,
  });

  // With a second factor enrolled, Firebase answers with a pending credential
  // and no tokens at all. That isn't an error — it's the middle of a sign-in.
  if (payload.mfaPendingCredential) {
    return {
      status: "mfa-required",
      challenge: {
        pendingCredential: String(payload.mfaPendingCredential),
        factors: (payload.mfaInfo ?? []).map((f: Record<string, any>) => ({
          id: String(f.mfaEnrollmentId),
          displayName: f.displayName ? String(f.displayName) : undefined,
        })),
      },
    };
  }
  return { status: "signed-in", tokens: toTokens(payload) };
}

/** Completes a sign-in with the 6-digit code from the authenticator app. */
export async function completeMfaSignIn(
  challenge: MfaChallenge,
  code: string
): Promise<Tokens> {
  const factor = challenge.factors[0];
  if (!factor) throw new AuthError("no-factor", "No authenticator is enrolled.");
  return toTokens(
    await post(`${IDENTITY_V2}/mfaSignIn:finalize`, {
      mfaPendingCredential: challenge.pendingCredential,
      mfaEnrollmentId: factor.id,
      totpVerificationInfo: { verificationCode: code },
    })
  );
}

export async function refresh(refreshToken: string): Promise<Tokens> {
  const payload = await post<Record<string, any>>(SECURE_TOKEN, {
    grant_type: "refresh_token",
    refresh_token: refreshToken,
  });
  return toTokens(payload);
}

export async function sendVerificationEmail(idToken: string): Promise<void> {
  await post(`${IDENTITY}:sendOobCode`, { requestType: "VERIFY_EMAIL", idToken });
}

export async function sendPasswordReset(email: string): Promise<void> {
  await post(`${IDENTITY}:sendOobCode`, { requestType: "PASSWORD_RESET", email });
}

export interface AccountInfo {
  emailVerified: boolean;
  email: string;
  /** Enrolled second factors, if any. */
  mfaFactors: { id: string; displayName?: string }[];
}

export async function lookup(idToken: string): Promise<AccountInfo> {
  const payload = await post<Record<string, any>>(`${IDENTITY}:lookup`, { idToken });
  const user = (payload.users ?? [])[0] ?? {};
  return {
    emailVerified: user.emailVerified === true,
    email: String(user.email ?? ""),
    mfaFactors: (user.mfaInfo ?? []).map((f: Record<string, any>) => ({
      id: String(f.mfaEnrollmentId),
      displayName: f.displayName ? String(f.displayName) : undefined,
    })),
  };
}

export interface TotpEnrollment {
  sessionInfo: string;
  sharedSecretKey: string;
  /** `otpauth://` URL for the QR code. */
  uri: string;
}

export async function startTotpEnrollment(
  idToken: string,
  email: string
): Promise<TotpEnrollment> {
  const payload = await post<Record<string, any>>(
    `${IDENTITY_V2}/mfaEnrollment:start`,
    { idToken, totpEnrollmentInfo: {} }
  );
  const info = payload.totpSessionInfo ?? {};
  const secret = String(info.sharedSecretKey ?? "");
  const label = encodeURIComponent(email || "account");
  return {
    sessionInfo: String(info.sessionInfo ?? ""),
    sharedSecretKey: secret,
    uri:
      `otpauth://totp/Jarvis:${label}?secret=${secret}&issuer=Jarvis` +
      `&algorithm=${info.hashingAlgorithm ?? "SHA1"}` +
      `&digits=${info.verificationCodeLength ?? 6}` +
      `&period=${info.periodSec ?? 30}`,
  };
}

export async function finishTotpEnrollment(
  idToken: string,
  sessionInfo: string,
  code: string
): Promise<Tokens> {
  return toTokens(
    await post(`${IDENTITY_V2}/mfaEnrollment:finalize`, {
      idToken,
      displayName: "Authenticator app",
      totpVerificationInfo: { sessionInfo, verificationCode: code },
    })
  );
}
