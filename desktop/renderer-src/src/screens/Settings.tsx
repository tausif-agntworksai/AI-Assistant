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
 *
 * Permissions live here too, and that is the point rather than a convenience.
 * They were asked once during setup and then had no way back: changing your mind
 * about the microphone or the clipboard meant deleting a file you would have to
 * be told about. Anything a user can be asked to grant has to be revocable in
 * the same number of clicks, or the original question was not really a question.
 */
import { useCallback, useEffect, useState } from "react";
import {
  ai,
  cleanError,
  consent,
  type AiInfo,
  type CapabilitySpec,
  type ModelOption,
  type ProviderMeta,
  type SpeechProviderMeta,
} from "../bridge";
import { Badge, Button, Field, Note, Spinner, Switch } from "../components/ui";
import { useTheme, type ThemeChoice } from "../useTheme";
import "./settings.css";

type Status = "idle" | "checking" | "ok" | "error";

/** Sentinel for the "type it yourself" option in the model dropdown. */
const CUSTOM = "__custom__";

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
            <PermissionsSection />
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
  // A dropdown built from a list that failed to load is a dead end, so
  // there is always a way to name a model by hand.
  const [custom, setCustom] = useState(false);

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
    setCustom(false);
  };

  const save = async () => {
    setStatus("checking");
    setMessage(null);
    try {
      let settled = model;
      if (key.trim()) {
        const check = await ai.validate(providerId, key.trim(), model);
        if (!check.ok) {
          setStatus("error");
          setMessage(check.error ?? "That key could not be verified.");
          return;
        }
        // Providers retire models without retiring keys, so the engine reports
        // back the model that actually answered. Save that, not what we asked
        // for — otherwise the very next question fails on a dead model id.
        settled = check.model ?? model;
      } else if (!hasStoredKey) {
        setStatus("error");
        setMessage("Paste an API key first.");
        return;
      }

      await ai.setLlm(providerId, settled, key.trim() ? key.trim() : undefined);
      setKey("");
      setModel(settled);
      setStatus("ok");
      const changed = settled && model && settled !== model;
      setMessage(
        changed
          ? `Ready — ${model} isn't available to this key, so it's using ${settled}.`
          : `Ready — using ${provider?.label ?? providerId}${settled ? ` (${settled})` : ""}.`
      );
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
          custom
            ? "Exact model id, as the provider spells it."
            : loadingModels
              ? "Loading this provider's models…"
              : models.length
                ? "Save checks the model as well as the key, and falls back to a working one."
                : "Add a key to load the full list."
        }
      >
        {custom ? (
          <input
            className="text"
            autoFocus
            spellCheck={false}
            placeholder={provider?.default_model ?? "model-id"}
            value={model}
            onChange={(event) => setModel(event.target.value)}
          />
        ) : (
          <select
            value={options.some((o) => o.id === model) || !model ? model : CUSTOM}
            onChange={(event) => {
              if (event.target.value === CUSTOM) {
                setCustom(true);
                return;
              }
              setModel(event.target.value);
            }}
          >
            <option value="">
              {provider ? `Default (${provider.default_model})` : "Default"}
            </option>
            {options.map((option) => (
              <option key={option.id} value={option.id}>
                {option.label}
                {option.speed ? ` — ${option.speed}` : ""}
              </option>
            ))}
            <option value={CUSTOM}>Type a model id…</option>
          </select>
        )}
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

/* ── permissions ──────────────────────────────────────────────────────── */

/**
 * The same fifteen capabilities the setup screen asks about, editable.
 *
 * Three things differ from the setup screen, all because this is a settings
 * pane rather than a wizard:
 *
 *   * **Changes apply as you make them.** A Save button here would let someone
 *     flip four switches, close the panel and believe they had changed
 *     something. Each toggle posts immediately and the engine re-reads its
 *     consent, so a capability switched off is off before the panel closes.
 *   * **The last state is kept for the reversal.** Posting the whole map on
 *     every toggle means a failed save would otherwise leave the UI showing a
 *     state the engine does not have, so a failure rolls the switch back and
 *     says why.
 *   * **Required capabilities are shown, not hidden.** The microphone and the
 *     speaker cannot be turned off — an assistant that cannot hear or answer is
 *     not one — and showing them greyed with the reason is more honest than a
 *     list that quietly omits two entries.
 */
function PermissionsSection() {
  const [specs, setSpecs] = useState<CapabilitySpec[] | null>(null);
  const [granted, setGranted] = useState<Record<string, boolean>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    const data = await consent.list();
    setSpecs(data.capabilities);
    setGranted(data.granted);
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const toggle = async (id: string, next: boolean) => {
    const previous = granted;
    const updated = { ...granted, [id]: next };
    setGranted(updated);
    setBusy(id);
    setError("");
    try {
      await consent.save(updated);
      // Re-read rather than trusting the local copy: the engine is the
      // authority on what it will actually honour, and a capability the
      // installer marked required will come back on.
      await load();
    } catch (err) {
      setGranted(previous);
      setError(cleanError(err));
    } finally {
      setBusy(null);
    }
  };

  if (!specs) {
    return (
      <section className="panel section">
        <div className="section-head">
          <h3>Permissions</h3>
        </div>
        <Spinner label="Reading permissions…" />
      </section>
    );
  }

  const on = specs.filter((spec) => granted[spec.id]).length;

  return (
    <section className="panel section">
      <div className="section-head">
        <h3>Permissions</h3>
        <Badge kind="info">
          {on} of {specs.length} on
        </Badge>
      </div>
      <p className="section-note">
        What Jarvis is allowed to reach. Switching one off removes the ability
        rather than adding a question — the engine refuses the skill outright.
      </p>

      {error && <Note kind="warn">{error}</Note>}

      <div className="perm-list">
        {specs.map((spec) => (
          <PermissionRow
            key={spec.id}
            spec={spec}
            on={Boolean(granted[spec.id])}
            busy={busy === spec.id}
            onChange={(next) => void toggle(spec.id, next)}
          />
        ))}
      </div>
    </section>
  );
}

function PermissionRow({
  spec,
  on,
  busy,
  onChange,
}: {
  spec: CapabilitySpec;
  on: boolean;
  busy: boolean;
  onChange: (next: boolean) => void;
}) {
  // Windows keeps its own switch for the microphone and the camera, and consent
  // here means nothing if that one is off. The failure is silent — the stream
  // opens and delivers nothing forever — so it is called out and linked.
  const osBlocked = spec.osPermission && spec.os === "denied";

  return (
    <div className={`perm${on ? "" : " off"}`}>
      <div className="perm-main">
        <div className="perm-text">
          <div className="perm-title">
            {spec.title}
            {spec.required && <Badge kind="info">always on</Badge>}
            {spec.unused && <Badge kind="info">not used yet</Badge>}
          </div>
          <div className="perm-what">{spec.what}</div>
        </div>
        <Switch
          checked={on}
          disabled={Boolean(spec.required) || busy}
          onChange={onChange}
          label={spec.title}
        />
      </div>
      {osBlocked && (
        <Note kind="warn">
          Windows is blocking this, so granting it here has no effect.{" "}
          <button
            className="linkish"
            onClick={() => void consent.openOsSettings(spec.osPermission!)}
          >
            Open Windows settings
          </button>
        </Note>
      )}
    </div>
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
