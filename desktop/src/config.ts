/**
 * Build-time configuration: the Firebase project, and who counts as an admin.
 *
 * Read from `desktop/.env` when running from the repo, and from the
 * `app.env` that `scripts/make-app-env.mjs` writes into the package when
 * installed. Everything here is public-by-design — a Firebase web config is
 * meant to be shipped, and the protection comes from Firebase's own rules plus
 * the checks in `auth/session.ts`, not from the values being secret.
 */
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { app } from "electron";

export interface FirebaseConfig {
  apiKey: string;
  authDomain: string;
  projectId: string;
  appId: string;
}

/** Minimal dotenv reader — enough for the KEY=VALUE files in this repo. */
function readEnvFile(file: string): Record<string, string> {
  if (!existsSync(file)) return {};
  const out: Record<string, string> = {};
  for (const line of readFileSync(file, "utf8").split(/\r?\n/)) {
    const match = /^\s*(?:export\s+)?([A-Z0-9_]+)\s*=\s*(.*)$/i.exec(line);
    if (!match) continue;
    let value = match[2]!.trim();
    if (/^(["']).*\1$/.test(value)) value = value.slice(1, -1);
    else value = value.replace(/\s+#.*$/, "").trim();
    out[match[1]!] = value;
  }
  return out;
}

function load(): Record<string, string> {
  const fromFile = app.isPackaged
    ? readEnvFile(path.join(process.resourcesPath, "app.env"))
    : readEnvFile(path.resolve(app.getAppPath(), ".env"));
  // A real environment variable wins, so a launcher or a test can override
  // anything here without editing a file.
  return { ...fromFile, ...(process.env as Record<string, string>) };
}

const env = load();

function value(key: string, fallback = ""): string {
  return (env[key] ?? "").trim() || fallback;
}

export const firebase: FirebaseConfig = {
  apiKey: value("JARVIS_FIREBASE_API_KEY"),
  authDomain: value("JARVIS_FIREBASE_AUTH_DOMAIN"),
  projectId: value("JARVIS_FIREBASE_PROJECT_ID"),
  appId: value("JARVIS_FIREBASE_APP_ID"),
};

/**
 * True once there is enough configuration to sign anyone in. Until then the
 * app runs unlocked and says so, rather than presenting a sign-in screen that
 * cannot possibly succeed — the same "setup mode" the AI Calculator uses.
 */
export const firebaseConfigured = Boolean(firebase.apiKey && firebase.projectId);

/**
 * Enforce the whole gate: verified email, admin approval, and (when enabled)
 * an authenticator app. Turning this off leaves the sign-in screen in place
 * but stops it blocking the assistant — useful while setting Firebase up.
 */
export const requireAuth =
  value("JARVIS_REQUIRE_AUTH", "true").toLowerCase() !== "false" && firebaseConfigured;

/**
 * Two-step verification is always required — it is not a setting.
 *
 * It used to be one, because the previous design also had an administrator
 * approving every new account, and TOTP needs the Firebase **Blaze** plan
 * (billing account attached; no per-use charge at this volume). With approval
 * gone, the authenticator is the only thing standing between a leaked password
 * and someone else's machine, so it stops being optional. `JARVIS_REQUIRE_AUTH`
 * is the escape hatch, and it is for local development, not for shipping.
 */
export const twoFactorRequired = requireAuth;

/**
 * How long a signed-in session may survive without reaching Firebase.
 *
 * A voice assistant that stops working on a train is a broken voice assistant,
 * so an offline launch reuses the last verified session. It is a real trade:
 * an account revoked while this machine is offline keeps working until the
 * window closes. Seven days is short enough to bound that and long enough to
 * cover any ordinary stretch without a connection.
 */
export const OFFLINE_GRACE_MS = 7 * 24 * 60 * 60 * 1000;
