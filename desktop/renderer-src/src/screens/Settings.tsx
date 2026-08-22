/**
 * Settings: the AI model, the speech recogniser, and the theme.
 *
 * The shape of the key flow is taken from the sibling AI Calculator — pick a
 * provider, paste a key, validate it live before saving — with three
 * deliberate differences:
 *
 *   * **The model is chosen too, not pinned by whoever built the app.** On a
 *     voice assistant this is a latency decision: Groq answers in a couple of
 *     hundred milliseconds and Opus takes several seconds, and for "chrome
 *     kholo" that gap is the difference between a product and a demo. Hence the
 *     speed badges.
 *   * **The key never comes back.** It goes to the main process, which encrypts
 *     it with the OS keychain. This screen can set one and can ask whether one
 *     exists; there is no call that reads it.
 *   * **A validated key is saved only after the provider confirms it.** Storing
 *     first and discovering later means an assistant that fails on its first
 *     real question with a message about a key you thought you had set.
 */
import { useCallback, useEffect, useState } from "react";
import {
  ai,
  cleanError,
  type AiInfo,
  type ModelOption,
  type ProviderMeta,
  type SpeechProviderMeta,
} from "../bridge";
import { Badge, Button, Field, Note, Spinner } from "../components/ui";
import { useTheme, type ThemeChoice } from "../useTheme";
import "./settings.css";

type Status = "idle" | "checking" | "ok" | "error";

export function Settings({ onClose }: { onClose: () => void }) {
  const [info, setInfo] = useState<AiInfo | null>(null);

  const reload = useCallback(async () => setInfo(await ai.info()), []);
  useEffect(() => {
    void reload();
  }, [reload]);

  return (
    <div className="settings">
      <div className="settings-head">
        <h2>Settings</h2>
        <Button variant="ghost" onClick={onClose}>
          Done
        </Button>
      </div>

      <div className="settings-body">
        {!info ? (
          <Spinner label="Loading…" />
        ) : (
          <>
            <ModelSection info={info} onSaved={reload} />
            <SpeechSection info={info} onSaved={reload} />
            <AppearanceSection />
          </>
        )}
      </div>
    </div>
  );
}

/* ── the language model ───────────────────────────────────────────────── */

function ModelSection({ info, onSaved }: { info: AiInfo; onSaved: () => Promise<void> }) {
  const providers = info.providers ?? [];
  const stored = info.stored.llm;

  const [providerId, setProviderId] = useState(stored.provider || "anthropic");
  const [model, setModel] = useState(stored.model);
  const [key, setKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState<string | null>(null);
  const [models, setModels] = useState<ModelOption[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);

  const provider: ProviderMeta | undefined = providers.find((p) => p.id === providerId);
  const hasStoredKey = stored.hasKey && stored.provider === providerId;

  // The provider's own list once there is a key to ask with; otherwise the
  // handful we know about, so the dropdown is never empty.
  const loadModels = useCallback(
    async (id: string, candidate: string) => {
      if (!candidate && !(stored.hasKey && stored.provider === id)) {
        setModels([]);
        return;
      }
      setLoadingModels(true);
      try {
        const result = await ai.models(id, candidate);
        setModels(result.models ?? []);
      } finally {
        setLoadingModels(false);
      }
    },
    [stored.hasKey, stored.provider]
  );

  useEffect(() => {
    void loadModels(providerId, "");
  }, [providerId, loadModels]);

  const options: ModelOption[] = models.length ? models : (provider?.known_models ?? []);

  const changeProvider = (id: string) => {
    setProviderId(id);
    // A key belongs to one provider, so carrying it across would send an
    // OpenAI key to Groq and report "that key was rejected".
    setKey("");
    setModel("");
    setStatus("idle");
    setMessage(null);
    setModels([]);
  };

  const save = async () => {
    setStatus("checking");
    setMessage(null);
    try {
      if (key.trim()) {
        const check = await ai.validate(providerId, key.trim(), model);
        if (!check.ok) {
          setStatus("error");
          setMessage(check.error ?? "That key could not be verified.");
          return;
        }
      } else if (!hasStoredKey) {
        setStatus("error");
        setMessage("Paste an API key first.");
        return;
      }
      // Saved only after the provider itself confirmed the key works.
      await ai.setLlm(providerId, model, key.trim() ? key.trim() : undefined);
      setKey("");
      setStatus("ok");
      setMessage(`Ready — using ${provider?.label ?? providerId}.`);
      await onSaved();
      void loadModels(providerId, "");
    } catch (err) {
      setStatus("error");
      setMessage(cleanError(err));
    }
  };

  const forget = async () => {
    await ai.forget();
    setKey("");
    setModel("");
    setStatus("idle");
    setMessage("Key removed. Offline commands still work.");
    await onSaved();
  };

  return (
    <section className="panel section">
      <div className="section-head">
        <h3>AI model</h3>
        {stored.hasKey ? <Badge kind="ok">connected</Badge> : <Badge kind="warn">no key</Badge>}
      </div>

      <p className="section-note">
        Opening apps, volume, brightness, timers and battery all work with no key
        at all. A key is what makes conversation, questions and translation work.
      </p>

      <Field label="Provider">
        <select value={providerId} onChange={(event) => changeProvider(event.target.value)}>
          {providers.map((option) => (
            <option key={option.id} value={option.id}>
              {option.label}
            </option>
          ))}
        </select>
      </Field>

      <Field
        label="Model"
        hint={
          loadingModels
            ? "Loading this provider's models…"
            : models.length
              ? undefined
              : "Add a key to load the full list."
        }
      >
        <select value={model} onChange={(event) => setModel(event.target.value)}>
          <option value="">
            {provider ? `Default (${provider.default_model})` : "Default"}
          </option>
          {options.map((option) => (
            <option key={option.id} value={option.id}>
              {option.label}
              {option.speed ? ` — ${option.speed}` : ""}
            </option>
          ))}
        </select>
      </Field>

      <Field
        label={hasStoredKey ? "Replace API key" : "Your API key"}
        hint="Stored encrypted on this machine with your Windows account's own key. Never written to a log, never sent anywhere except the provider you chose."
      >
        <div className="key-row">
          <input
            className="text grow"
            type={showKey ? "text" : "password"}
            autoComplete="off"
            spellCheck={false}
            placeholder={hasStoredKey ? "•••••••••••••••• (saved)" : "Paste it here"}
            value={key}
            onChange={(event) => setKey(event.target.value)}
          />
          <Button variant="secondary" onClick={() => setShowKey((v) => !v)}>
            {showKey ? "Hide" : "Show"}
          </Button>
        </div>
      </Field>

      {provider && (
        <a className="help-link" href={provider.help_url} target="_blank" rel="noreferrer">
          Get a {provider.label} API key →
        </a>
      )}

      {message && (
        <Note kind={status === "error" ? "danger" : status === "ok" ? "ok" : "info"}>
          {message}
        </Note>
      )}

      <div className="section-actions">
        <Button disabled={status === "checking"} onClick={() => void save()}>
          {status === "checking" ? "Checking…" : "Save"}
        </Button>
        {stored.hasKey && (
          <Button variant="ghost" onClick={() => void forget()}>
            Remove key
          </Button>
        )}
      </div>
    </section>
  );
}

/* ── the speech recogniser ────────────────────────────────────────────── */

function SpeechSection({ info, onSaved }: { info: AiInfo; onSaved: () => Promise<void> }) {
  const providers: SpeechProviderMeta[] = info.speech_providers ?? [];
  const stored = info.stored.speech;

  const [providerId, setProviderId] = useState(stored.provider);
  const [key, setKey] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const provider = providers.find((p) => p.id === providerId);
  const hasStoredKey = stored.hasKey && stored.provider === providerId;

  const save = async () => {
    setSaving(true);
    setMessage(null);
    try {
      await ai.setSpeech(providerId, key.trim() ? key.trim() : undefined);
      setKey("");
      setMessage(
        providerId
          ? "Saved. It will be used only when the local model comes back empty."
          : "Turned off. Speech stays entirely on this machine."
      );
      await onSaved();
    } catch (err) {
      setMessage(cleanError(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="panel section">
      <div className="section-head">
        <h3>Speech recognition</h3>
        {stored.hasKey ? <Badge kind="ok">cloud backup</Badge> : <Badge>local only</Badge>}
      </div>

      <p className="section-note">
        Speech is recognised on this machine by default. Hindi mixed with English
        is the hardest case for the local model, and a cloud recogniser is
        noticeably better at it — so you can add one as a fallback.{" "}
        <strong>
          Audio only leaves this machine after both local passes have already
          failed to make sense of it.
        </strong>
      </p>

      <Field label="Fallback recogniser">
        <select
          value={providerId}
          onChange={(event) => {
            setProviderId(event.target.value);
            setKey("");
            setMessage(null);
          }}
        >
          <option value="">Off — never leave this machine</option>
          {providers.map((option) => (
            <option key={option.id} value={option.id}>
              {option.label}
            </option>
          ))}
        </select>
      </Field>

      {provider && (
        <>
          <p className="section-note tiny">{provider.note}</p>
          <Field label={hasStoredKey ? "Replace API key" : "API key"}>
            <input
              className="text"
              type="password"
              autoComplete="off"
              spellCheck={false}
              placeholder={hasStoredKey ? "•••••••••••••••• (saved)" : "Paste it here"}
              value={key}
              onChange={(event) => setKey(event.target.value)}
            />
          </Field>
          <a className="help-link" href={provider.help_url} target="_blank" rel="noreferrer">
            Get a {provider.label} API key →
          </a>
        </>
      )}

      {message && <Note kind="info">{message}</Note>}

      <div className="section-actions">
        <Button disabled={saving} onClick={() => void save()}>
          {saving ? "Saving…" : "Save"}
        </Button>
      </div>
    </section>
  );
}

/* ── appearance ───────────────────────────────────────────────────────── */

function AppearanceSection() {
  const [theme, setTheme] = useTheme();
  const choices: { id: ThemeChoice; label: string }[] = [
    { id: "system", label: "Follow the system" },
    { id: "dark", label: "Dark" },
    { id: "light", label: "Light" },
  ];

  return (
    <section className="panel section">
      <div className="section-head">
        <h3>Appearance</h3>
      </div>
      <div className="theme-picker">
        {choices.map((choice) => (
          <button
            key={choice.id}
            className={theme === choice.id ? "active" : ""}
            onClick={() => setTheme(choice.id)}
          >
            {choice.label}
          </button>
        ))}
      </div>
    </section>
  );
}
