/**
 * What Jarvis may use, asked once and enforced everywhere.
 *
 * Fifteen capabilities, each with a plain-language sentence and a switch. Every
 * one maps to a `Capability` in the engine's `permissions.py`, and the engine
 * refuses a skill whose capability is off — *before* the confirmation prompt, so
 * switching something off removes the ability rather than adding a question.
 *
 * Two things make this more than a checklist:
 *
 *   * **Windows has its own switch** for the microphone and camera, and consent
 *     here means nothing if that one is off. The failure is completely silent —
 *     the stream opens and delivers digital silence forever — so it is checked
 *     and linked directly.
 *   * **The microphone gets tested before you leave.** "It never hears me" is
 *     the single most common way a voice assistant is broken, and almost always
 *     because of a muted, blocked or wrong-default device. Catching that at
 *     setup, with a live meter, is worth a whole step.
 */
import { useEffect, useMemo, useState } from "react";
import { consent, type CapabilitySpec } from "../bridge";
import { Bilingual, Badge, Button, Note, Switch } from "../components/ui";
import { MicCheck } from "../components/MicCheck";
import { WindowControls } from "../components/WindowControls";
import "./consent.css";

type Step = "capabilities" | "microphone";

export function Consent() {
  const [specs, setSpecs] = useState<CapabilitySpec[]>([]);
  const [granted, setGranted] = useState<Record<string, boolean>>({});
  const [firstRun, setFirstRun] = useState(true);
  const [step, setStep] = useState<Step>("capabilities");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    void consent.list().then((data) => {
      setSpecs(data.capabilities);
      setGranted(data.granted);
      setFirstRun(data.needsAsking);
    });
  }, []);

  const micBlocked = useMemo(
    () => specs.some((s) => s.osPermission === "microphone" && s.os === "denied"),
    [specs]
  );

  const save = async (advance: boolean) => {
    setSaving(true);
    try {
      await consent.save(granted);
      // On a first run the microphone check comes next. Afterwards the main
      // process decides where to go — so this page never navigates itself.
      if (advance && firstRun) setStep("microphone");
    } finally {
      setSaving(false);
    }
  };

  const onlyEssentials = () => {
    // The smallest set that still leaves a working assistant: hear, answer,
    // open apps. Everything else off.
    const essentials = new Set(["microphone", "speaker", "app_control"]);
    setGranted(
      Object.fromEntries(specs.map((s) => [s.id, Boolean(s.required) || essentials.has(s.id)]))
    );
  };

  if (step === "microphone") {
    return (
      <div className="consent">
        <ConsentHeader
          title="Can Jarvis hear you?"
          titleHi="क्या Jarvis आपकी आवाज़ सुन पा रहा है?"
          sub="Say “hey jarvis” and watch the ring. This catches a muted or wrong microphone now rather than the first time you need it."
          subHi="“hey jarvis” बोलिए और रिंग देखिए।"
        />
        <main className="consent-body">
          <MicCheck blocked={micBlocked} />
        </main>
        <footer className="consent-foot">
          <Button variant="secondary" onClick={() => setStep("capabilities")}>
            Back
          </Button>
          <Button full onClick={() => void consent.save(granted)}>
            Done
          </Button>
        </footer>
      </div>
    );
  }

  return (
    <div className="consent">
      <ConsentHeader
        title={firstRun ? "What may Jarvis use?" : "Permissions"}
        titleHi={firstRun ? "Jarvis क्या-क्या इस्तेमाल कर सकता है?" : "अनुमतियाँ"}
        sub="Everything here can be changed later from the tray icon, and anything switched off is refused by the engine itself."
        subHi="इन्हें आप बाद में ट्रे आइकॉन से बदल सकते हैं।"
      />

      <main className="consent-body">
        {specs.map((spec) => (
          <Capability
            key={spec.id}
            spec={spec}
            granted={Boolean(granted[spec.id])}
            onChange={(next) => setGranted((prev) => ({ ...prev, [spec.id]: next }))}
          />
        ))}
      </main>

      <footer className="consent-foot">
        <Button variant="secondary" onClick={onlyEssentials}>
          Only the essentials
        </Button>
        <Button full disabled={saving} onClick={() => void save(true)}>
          {saving ? "Saving…" : firstRun ? "Continue" : "Save"}
        </Button>
      </footer>
    </div>
  );
}

function ConsentHeader({
  title,
  titleHi,
  sub,
  subHi,
}: {
  title: string;
  titleHi: string;
  sub: string;
  subHi: string;
}) {
  return (
    <header className="consent-head">
      <div className="consent-head-top">
        <span className="brand">JARVIS</span>
        <span className="grow" />
        <WindowControls />
      </div>
      <h1>
        <Bilingual en={title} hi={titleHi} />
      </h1>
      <p className="consent-sub">
        <Bilingual en={sub} hi={subHi} />
      </p>
    </header>
  );
}

function Capability({
  spec,
  granted,
  onChange,
}: {
  spec: CapabilitySpec;
  granted: boolean;
  onChange: (next: boolean) => void;
}) {
  const osProblem =
    spec.osPermission && spec.os && spec.os !== "granted" && spec.os !== "unknown";

  return (
    <div className={`cap${granted ? "" : " off"}`}>
      <div className="cap-top">
        <div className="grow">
          <div className="cap-title">
            {spec.title}
            {spec.required && <Badge kind="ok">required</Badge>}
            {spec.unused && <Badge kind="warn">unused</Badge>}
          </div>
          <div className="cap-title-hi hi" lang="hi">
            {spec.titleHi}
          </div>
        </div>
        <Switch
          checked={granted}
          disabled={spec.required}
          onChange={onChange}
          label={spec.title}
        />
      </div>

      <p className="cap-what">{spec.what}</p>
      <p className="cap-what hi" lang="hi">
        {spec.whatHi}
      </p>
      {spec.examples && spec.examples !== "—" && (
        <p className="cap-eg mono">{spec.examples}</p>
      )}

      {osProblem && (
        <Note
          kind="warn"
          action={
            <Button
              variant="secondary"
              onClick={() => void consent.openOsSettings(spec.osPermission!)}
            >
              Open settings
            </Button>
          }
        >
          {spec.os === "denied"
            ? `Windows is currently blocking ${spec.osPermission} access for apps.`
            : `Windows has not been asked for ${spec.osPermission} access yet.`}
        </Note>
      )}
    </div>
  );
}
