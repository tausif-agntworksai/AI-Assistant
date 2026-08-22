/**
 * Sign in, verify the email, enrol an authenticator. Six states, one switch.
 *
 * The gate is self-serve: nobody approves anybody. What it protects is the
 * microphone — the engine boots with its listening loop shut and only opens it
 * once the main process says a verified, enrolled person is present.
 *
 * All of the actual auth work happens in the main process. This screen never
 * sees a token or a refresh token, and the only secret it handles is the TOTP
 * key it has to display for enrolment.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { auth, cleanError, type AuthStatus } from "../bridge";
import { Bilingual, Button, Field, Note, Spinner } from "../components/ui";
import { Orb } from "../components/Orb";
import { WindowControls } from "../components/WindowControls";
import "./auth.css";

export function Auth() {
  const [status, setStatus] = useState<AuthStatus | null>(null);

  useEffect(() => {
    void auth.status().then(setStatus);
    // The disposer matters: without it every remount would add another
    // ipcRenderer listener and the same status would arrive twice.
    return auth.onChange(setStatus);
  }, []);

  if (!status) {
    return (
      <Shell title="Loading…">
        <Spinner label="Checking your account" />
      </Shell>
    );
  }

  switch (status.state) {
    case "unconfigured":
      return <Unconfigured />;
    case "signed-out":
      return <SignIn onDone={setStatus} />;
    case "needs-verification":
      return <Verify status={status} onDone={setStatus} />;
    case "needs-2fa":
      return status.codeReason === "enroll" ? (
        <Enrol onDone={setStatus} />
      ) : (
        <Challenge onDone={setStatus} />
      );
    case "offline-expired":
      return <OfflineExpired onDone={setStatus} />;
    case "ready":
      // The main process swaps the page the moment the gate opens; this is
      // only what shows for the instant in between.
      return (
        <Shell title="Signed in" subtitle="Starting Jarvis…" subtitleHi="Jarvis शुरू हो रहा है…">
          <Spinner />
        </Shell>
      );
    default:
      return (
        <Shell title="Loading…">
          <Spinner />
        </Shell>
      );
  }
}

/* ── shell ────────────────────────────────────────────────────────────── */

function Shell({
  title,
  subtitle,
  subtitleHi,
  children,
}: {
  title: string;
  subtitle?: string;
  subtitleHi?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="gate">
      <header className="gate-drag">
        <span className="brand">JARVIS</span>
        <span className="grow" />
        <WindowControls />
      </header>
      <main className="gate-body">
        <div className="gate-card">
          <Orb state="idle" size={54} />
          <h1>{title}</h1>
          {subtitle && (
            <p className="gate-sub">
              <Bilingual en={subtitle} hi={subtitleHi} />
            </p>
          )}
          {children}
        </div>
      </main>
    </div>
  );
}

/** Wraps an async action: one in flight at a time, and errors land on screen. */
function useAction() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const run = useCallback(async (fn: () => Promise<void>) => {
    setError(null);
    setNote(null);
    setBusy(true);
    try {
      await fn();
    } catch (err) {
      if (alive.current) setError(cleanError(err));
    } finally {
      if (alive.current) setBusy(false);
    }
  }, []);

  return { busy, error, note, setNote, run };
}

function Feedback({ error, note }: { error: string | null; note: string | null }) {
  if (error) return <Note kind="danger">{error}</Note>;
  if (note) return <Note kind="info">{note}</Note>;
  return null;
}

/* ── screens ──────────────────────────────────────────────────────────── */

function SignIn({ onDone }: { onDone: (s: AuthStatus) => void }) {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const { busy, error, note, setNote, run } = useAction();

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    void run(async () => {
      if (mode === "signup" && password !== confirm) {
        throw new Error("Those passwords don't match.");
      }
      onDone(await (mode === "signup" ? auth.signUp : auth.signIn)(email, password));
    });
  };

  return (
    <Shell
      title={mode === "login" ? "Welcome back" : "Create your account"}
      subtitle={
        mode === "login"
          ? "Jarvis stays deaf until you sign in."
          : "You will verify your email, then set up an authenticator app."
      }
      subtitleHi={
        mode === "login"
          ? "साइन इन करने तक Jarvis सुनना शुरू नहीं करेगा।"
          : "पहले ईमेल वेरिफ़ाई होगा, फिर ऑथेंटिकेटर ऐप सेट करना होगा।"
      }
    >
      <div className="tabs" role="tablist">
        {(["login", "signup"] as const).map((option) => (
          <button
            key={option}
            role="tab"
            aria-selected={mode === option}
            className={mode === option ? "active" : ""}
            onClick={() => setMode(option)}
          >
            {option === "login" ? "Sign in" : "Sign up"}
          </button>
        ))}
      </div>

      <form onSubmit={submit} className="gate-form">
        <Field label="Email">
          <input
            type="email"
            autoComplete="email"
            required
            autoFocus
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
        </Field>
        <Field label="Password" hint={mode === "signup" ? "At least 6 characters." : undefined}>
          <input
            type="password"
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            required
            minLength={6}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </Field>
        {mode === "signup" && (
          <Field label="Confirm password">
            <input
              type="password"
              autoComplete="new-password"
              required
              value={confirm}
              onChange={(event) => setConfirm(event.target.value)}
            />
          </Field>
        )}
        <Feedback error={error} note={note} />
        <Button type="submit" full disabled={busy}>
          {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
        </Button>
      </form>

      <div className="gate-foot">
        <Button
          variant="ghost"
          disabled={busy}
          onClick={() =>
            void run(async () => {
              if (!email.trim()) throw new Error("Enter your email address first.");
              await auth.resetPassword(email);
              setNote("Reset link sent — check your inbox.");
            })
          }
        >
          Forgot your password?
        </Button>
      </div>
    </Shell>
  );
}

function Verify({ status, onDone }: { status: AuthStatus; onDone: (s: AuthStatus) => void }) {
  const { busy, error, note, setNote, run } = useAction();

  return (
    <Shell
      title="Verify your email"
      subtitle={`We sent a link to ${status.email || "your email"}. Open it, then come back.`}
      subtitleHi="ईमेल में भेजे गए लिंक को खोलिए, फिर यहाँ वापस आइए।"
    >
      <div className="gate-form">
        <Feedback error={error} note={note} />
        <Button
          full
          disabled={busy}
          onClick={() =>
            void run(async () => {
              const next = await auth.recheck();
              if (next.state === "needs-verification") {
                setNote("Still not verified. Open the link, then try again.");
              } else {
                onDone(next);
              }
            })
          }
        >
          {busy ? "Checking…" : "I have verified — continue"}
        </Button>
        <Button
          variant="secondary"
          full
          disabled={busy}
          onClick={() =>
            void run(async () => {
              await auth.resendVerification();
              setNote("Verification email sent.");
            })
          }
        >
          Resend the email
        </Button>
      </div>
      <SignOutFoot onDone={onDone} />
    </Shell>
  );
}

function Challenge({ onDone }: { onDone: (s: AuthStatus) => void }) {
  const [code, setCode] = useState("");
  const { busy, error, note, run } = useAction();

  return (
    <Shell
      title="Two-step verification"
      subtitle="Enter the 6-digit code from your authenticator app."
      subtitleHi="अपने ऑथेंटिकेटर ऐप का 6 अंकों का कोड डालिए।"
    >
      <form
        className="gate-form"
        onSubmit={(event) => {
          event.preventDefault();
          void run(async () => onDone(await auth.submitCode(code)));
        }}
      >
        <Field label="6-digit code">
          <input
            className="code"
            inputMode="numeric"
            autoComplete="one-time-code"
            placeholder="······"
            maxLength={8}
            autoFocus
            value={code}
            onChange={(event) => setCode(event.target.value)}
          />
        </Field>
        <Feedback error={error} note={note} />
        <Button type="submit" full disabled={busy || code.trim().length < 6}>
          {busy ? "Verifying…" : "Verify and sign in"}
        </Button>
      </form>
      <SignOutFoot onDone={onDone} label="Cancel" />
    </Shell>
  );
}

function Enrol({ onDone }: { onDone: (s: AuthStatus) => void }) {
  const [secret, setSecret] = useState<{
    sharedSecretKey: string;
    qrSvg: string | null;
  } | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [code, setCode] = useState("");
  const { busy, error, note, run } = useAction();

  const load = useCallback(() => {
    setLoadError(null);
    setSecret(null);
    auth
      .beginTotp()
      .then(setSecret)
      .catch((err) => setLoadError(cleanError(err)));
  }, []);

  useEffect(load, [load]);

  return (
    <Shell
      title="Set up two-step verification"
      subtitle="Scan this with an authenticator app, then enter the code it shows."
      subtitleHi="ऑथेंटिकेटर ऐप से स्कैन कीजिए, फिर उसका कोड डालिए।"
    >
      {loadError && (
        <div className="gate-form">
          <Note kind="danger">{loadError}</Note>
          <Button variant="secondary" full onClick={load}>
            Try again
          </Button>
        </div>
      )}

      {!secret && !loadError && <Spinner label="Preparing your secure key…" />}

      {secret && (
        <>
          {secret.qrSvg && (
            // Trusted content: generated in our own main process from our own
            // string, and this page's CSP forbids anything external. Rendered
            // locally rather than through a QR web service because the thing
            // being encoded *is* the shared secret.
            <div className="qr" dangerouslySetInnerHTML={{ __html: secret.qrSvg }} />
          )}
          <details className="setup-key" open={!secret.qrSvg}>
            <summary>
              {secret.qrSvg ? "Cannot scan it? Enter the key by hand" : "Setup key"}
            </summary>
            <code className="mono">{secret.sharedSecretKey}</code>
          </details>

          <form
            className="gate-form"
            onSubmit={(event) => {
              event.preventDefault();
              void run(async () => onDone(await auth.submitCode(code)));
            }}
          >
            <Field label="6-digit code">
              <input
                className="code"
                inputMode="numeric"
                autoComplete="one-time-code"
                placeholder="······"
                maxLength={8}
                autoFocus
                value={code}
                onChange={(event) => setCode(event.target.value)}
              />
            </Field>
            <Feedback error={error} note={note} />
            <Button type="submit" full disabled={busy || code.trim().length < 6}>
              {busy ? "Verifying…" : "Turn on two-step verification"}
            </Button>
          </form>
        </>
      )}
      <SignOutFoot onDone={onDone} />
    </Shell>
  );
}

function OfflineExpired({ onDone }: { onDone: (s: AuthStatus) => void }) {
  const { busy, error, note, run } = useAction();
  return (
    <Shell
      title="Cannot reach Firebase"
      subtitle="Jarvis re-checks your account every week and has not been able to. Connect to the internet and try again."
      subtitleHi="इंटरनेट से जुड़िए और दोबारा कोशिश कीजिए।"
    >
      <div className="gate-form">
        <Feedback error={error} note={note} />
        <Button
          full
          disabled={busy}
          onClick={() => void run(async () => onDone(await auth.recheck()))}
        >
          {busy ? "Checking…" : "Try again"}
        </Button>
      </div>
      <SignOutFoot onDone={onDone} />
    </Shell>
  );
}

function Unconfigured() {
  return (
    <Shell
      title="Sign-in is not configured"
      subtitle="Jarvis will run without an account. To turn the security layer on, add your Firebase keys to desktop/.env."
      subtitleHi="बिना अकाउंट के चलेगा। सुरक्षा चालू करने के लिए Firebase keys डालिए।"
    >
      <div className="gate-form">
        <Note kind="warn">Running unlocked — anyone at this machine can use it.</Note>
        <Button full onClick={() => window.location.replace("index.html")}>
          Continue to Jarvis
        </Button>
      </div>
    </Shell>
  );
}

function SignOutFoot({
  onDone,
  label = "Sign out",
}: {
  onDone: (s: AuthStatus) => void;
  label?: string;
}) {
  return (
    <div className="gate-foot">
      <Button variant="ghost" onClick={() => void auth.signOut().then(onDone)}>
        {label}
      </Button>
    </div>
  );
}
