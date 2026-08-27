/**
 * What Jarvis is allowed to touch, asked once and enforced everywhere.
 *
 * An always-listening assistant that can shut the machine down, type into the
 * focused window and open WhatsApp is asking for a lot of trust, and the
 * honest time to ask for it is before it starts — not the first time each
 * capability happens to be used.
 *
 * Three properties this file is built around:
 *
 *   **Every entry is enforced.** Each `id` matches a `Capability` in
 *   `engine/jarvis/permissions.py`, and the engine refuses a skill whose
 *   capability is off. Turning something off here removes the ability rather
 *   than adding a confirmation prompt.
 *
 *   **Every entry is explainable.** If a capability can't be described in one
 *   sentence naming what it unlocks, a user can't meaningfully consent to it.
 *   That constraint is why this list has fifteen entries rather than seventy.
 *
 *   **Everything is revocable.** The same screen reopens from the tray, and a
 *   change takes effect immediately — including closing the microphone.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { app, shell, systemPreferences } from "electron";

/** Kept in lockstep with `Capability` in engine/jarvis/permissions.py. */
export type CapabilityId =
  | "microphone"
  | "speaker"
  | "camera"
  | "screen_capture"
  | "files"
  | "clipboard"
  | "input_synthesis"
  | "app_control"
  | "window_control"
  | "system_power"
  | "system_settings"
  | "device_status"
  | "shell"
  | "network"
  | "messaging"
  | "administrator";

export interface CapabilitySpec {
  id: CapabilityId;
  title: string;
  titleHi: string;
  /** One sentence: what it does, in the user's terms. */
  what: string;
  whatHi: string;
  /** Concrete examples, so "app control" isn't an abstraction. */
  examples: string;
  /** Required capabilities can't be switched off without the app being useless. */
  required?: boolean;
  /** Recommended default when the user has never decided. */
  granted: boolean;
  /** A Windows privacy setting also governs this one. */
  osPermission?: "microphone" | "camera";
  /** Shown when nothing currently uses it. */
  unused?: boolean;
}

/**
 * The manifest. Ordered by how much trust each one asks for, so someone
 * skim-reading meets the microphone before they meet "shut down the machine".
 */
export const CAPABILITIES: CapabilitySpec[] = [
  {
    id: "microphone",
    title: "Microphone",
    titleHi: "माइक्रोफ़ोन",
    what:
      "Listen for “hey jarvis”. Until the wake word fires, audio stays in " +
      "memory on this machine — nothing is transcribed, stored or sent.",
    whatHi:
      "“hey jarvis” सुनने के लिए। wake word से पहले ऑडियो सिर्फ़ इसी " +
      "मशीन की मेमोरी में रहता है — न रिकॉर्ड, न कहीं भेजा जाता है।",
    examples: "Everything spoken. Without this, Jarvis is keyboard-only.",
    required: true,
    granted: true,
    osPermission: "microphone",
  },
  {
    id: "speaker",
    title: "Speakers",
    titleHi: "स्पीकर",
    what: "Answer out loud, in Hindi or English.",
    whatHi: "हिंदी या अंग्रेज़ी में बोलकर जवाब देना।",
    examples: "Spoken replies, timer announcements, confirmation questions.",
    granted: true,
  },
  {
    id: "network",
    title: "Internet",
    titleHi: "इंटरनेट",
    what:
      "Reach Claude for anything the offline rules don't recognise, and fetch " +
      "weather, news and the neural voices.",
    whatHi:
      "जो कमांड ऑफ़लाइन नियम नहीं समझते उनके लिए Claude, और मौसम, " +
      "ख़बरें व नेचुरल आवाज़ें लाने के लिए।",
    examples: "“explain quantum computing”, “aaj ka mausam”, opening websites.",
    granted: true,
  },
  {
    id: "app_control",
    title: "Opening and closing apps",
    titleHi: "ऐप्स खोलना और बंद करना",
    what: "Launch and close installed applications, and switch between them.",
    whatHi: "इंस्टॉल ऐप्स खोलना, बंद करना और उनके बीच स्विच करना।",
    examples: "“chrome kholo”, “close spotify”, “switch to vs code”.",
    granted: true,
  },
  {
    id: "window_control",
    title: "Window management",
    titleHi: "विंडो संभालना",
    what: "Minimise, maximise, snap and close windows, and switch desktops.",
    whatHi: "विंडो छोटी-बड़ी करना, snap करना, बंद करना, डेस्कटॉप बदलना।",
    examples: "“minimise everything”, “show the desktop”, “snap left”.",
    granted: true,
  },
  {
    id: "device_status",
    title: "Device readings",
    titleHi: "डिवाइस की जानकारी",
    what: "Read the battery, CPU, memory, disk and network state.",
    whatHi: "बैटरी, CPU, मेमोरी, डिस्क और नेटवर्क की स्थिति पढ़ना।",
    examples: "“kitna battery bacha hai”, “system status”.",
    granted: true,
  },
  {
    id: "system_settings",
    title: "Windows settings",
    titleHi: "विंडोज़ सेटिंग्स",
    what: "Change brightness and night light, and open Settings pages.",
    whatHi: "ब्राइटनेस और नाइट लाइट बदलना, सेटिंग्स पेज खोलना।",
    examples: "“brightness 40”, “wi-fi settings kholo”.",
    granted: true,
  },
  {
    id: "shell",
    title: "Running system commands",
    titleHi: "सिस्टम कमांड चलाना",
    what:
      "Run PowerShell and command-line tools in the background. Windows " +
      "exposes brightness, wi-fi, window snapping and the Store-app list " +
      "no other way, so several features above stop working without this.",
    whatHi:
      "बैकग्राउंड में PowerShell और कमांड-लाइन टूल चलाना। ब्राइटनेस, " +
      "वाई-फ़ाई और Store ऐप्स की सूची विंडोज़ सिर्फ़ इसी तरह देता है।",
    examples: "No window is ever shown. Every command is in the Activity log.",
    granted: true,
  },
  {
    id: "files",
    title: "Your files",
    titleHi: "आपकी फ़ाइलें",
    what: "Open your personal folders and search them by name.",
    whatHi: "आपके फ़ोल्डर खोलना और नाम से फ़ाइल ढूँढना।",
    examples: "“open downloads”, “find file called invoice”.",
    granted: true,
  },
  {
    id: "screen_capture",
    title: "Screenshots",
    titleHi: "स्क्रीनशॉट",
    what: "Capture the screen and save it to your Pictures folder.",
    whatHi: "स्क्रीन कैप्चर करके Pictures फ़ोल्डर में सेव करना।",
    examples: "“take a screenshot”, “screenshot lo”.",
    granted: true,
  },
  {
    id: "clipboard",
    title: "Clipboard",
    titleHi: "क्लिपबोर्ड",
    what: "Read what you've copied and write to the clipboard.",
    whatHi: "आपने जो कॉपी किया है वो पढ़ना और क्लिपबोर्ड में लिखना।",
    examples: "“summarise the clipboard”, “copy this”.",
    granted: false,
  },
  {
    id: "input_synthesis",
    title: "Typing for you",
    titleHi: "आपकी जगह टाइप करना",
    what:
      "Type text into whatever window is focused. Anything typed goes " +
      "wherever the cursor is, so this one stays off unless you dictate.",
    whatHi:
      "जिस विंडो पर फ़ोकस है उसमें टाइप करना। कर्सर जहाँ है वहीं टेक्स्ट " +
      "जाएगा, इसलिए डिक्टेशन के बिना इसे बंद रखें।",
    examples: "“type dear sir, thank you for…”.",
    granted: false,
  },
  {
    id: "system_power",
    title: "Power controls",
    titleHi: "पावर कंट्रोल",
    what:
      "Lock, sleep, sign out, restart and shut down. Every one of these asks " +
      "out loud first, and shutdown waits 15 seconds so you can cancel it.",
    whatHi:
      "लॉक, स्लीप, साइन आउट, रीस्टार्ट और शटडाउन। हर बार पहले पूछा जाता " +
      "है, और शटडाउन 15 सेकंड रुकता है ताकि आप रोक सकें।",
    examples: "“laptop sula do”, “lock the screen”, “shutdown”.",
    granted: true,
  },
  {
    id: "administrator",
    title: "Administrator actions",
    titleHi: "एडमिनिस्ट्रेटर एक्शन",
    what:
      "Switch Wi-Fi and Bluetooth on or off. Windows refuses these to an " +
      "ordinary program, so Jarvis has to ask for Administrator rights — and " +
      "Windows still shows its own prompt every single time, naming what is " +
      "about to run. Jarvis never holds those rights while it is listening.",
    whatHi:
      "Wi-Fi और Bluetooth on/off करना। Windows ये काम आम प्रोग्राम को नहीं " +
      "करने देता, इसलिए Jarvis को एडमिन अनुमति माँगनी पड़ती है — और हर बार " +
      "Windows खुद भी पूछेगा। सुनते समय Jarvis के पास ये अधिकार कभी नहीं रहते।",
    examples: "“turn off wifi”, “bluetooth chalu karo”.",
    granted: false,
  },
  {
    id: "messaging",
    title: "Messaging",
    titleHi: "मैसेजिंग",
    what:
      "Open WhatsApp or your email client with a message prepared. Jarvis " +
      "never presses send — you do.",
    whatHi:
      "WhatsApp या ईमेल में मैसेज तैयार करके खोलना। भेजने का बटन " +
      "Jarvis कभी नहीं दबाता — वो आप दबाते हैं।",
    examples: "“whatsapp amit ko bolo main der se aaunga”.",
    granted: false,
  },
  {
    id: "camera",
    title: "Camera",
    titleHi: "कैमरा",
    what:
      "Nothing uses this today. It is listed so that if a camera feature is " +
      "ever added, it arrives switched off and asks you first.",
    whatHi:
      "अभी इसका कोई इस्तेमाल नहीं है। भविष्य में कैमरा फ़ीचर आया तो वो " +
      "बंद हालत में आएगा और पहले आपसे पूछेगा।",
    examples: "—",
    granted: false,
    osPermission: "camera",
    unused: true,
  },
];

export type Grants = Record<CapabilityId, boolean>;

interface StoredConsent {
  version: number;
  decidedAt: number;
  granted: Partial<Grants>;
}

/** Bumped when the manifest gains a capability that needs asking about. */
const VERSION = 1;

export class ConsentStore {
  private granted: Grants = defaults();
  private decidedVersion = 0;

  private get file(): string {
    return path.join(app.getPath("userData"), "consent.json");
  }

  load(): void {
    try {
      const parsed = JSON.parse(readFileSync(this.file, "utf8")) as StoredConsent;
      this.decidedVersion = Number(parsed.version) || 0;
      this.granted = { ...defaults(), ...(parsed.granted ?? {}) } as Grants;
      // A required capability can't be off — a stale file from an older
      // manifest must not leave the assistant unable to hear.
      for (const spec of CAPABILITIES) {
        if (spec.required) this.granted[spec.id] = true;
      }
    } catch {
      this.decidedVersion = 0;
      this.granted = defaults();
    }
  }

  /** True when the user has never been through the permission screen. */
  get needsAsking(): boolean {
    return this.decidedVersion < VERSION;
  }

  all(): Grants {
    return { ...this.granted };
  }

  allows(id: CapabilityId): boolean {
    return this.granted[id] === true;
  }

  save(granted: Partial<Grants>): Grants {
    this.granted = { ...this.granted, ...granted } as Grants;
    for (const spec of CAPABILITIES) {
      if (spec.required) this.granted[spec.id] = true;
    }
    this.decidedVersion = VERSION;
    try {
      mkdirSync(path.dirname(this.file), { recursive: true });
      const payload: StoredConsent = {
        version: VERSION,
        decidedAt: Date.now(),
        granted: this.granted,
      };
      writeFileSync(this.file, JSON.stringify(payload, null, 2));
    } catch (err) {
      console.warn("Could not save permissions:", (err as Error).message);
    }
    return this.all();
  }

  /** Wipes the record, so the next launch asks again. */
  reset(): void {
    this.decidedVersion = 0;
    this.granted = defaults();
    try {
      if (existsSync(this.file)) writeFileSync(this.file, "{}");
    } catch {
      /* the in-memory reset is enough to re-ask this session */
    }
  }
}

function defaults(): Grants {
  return Object.fromEntries(CAPABILITIES.map((c) => [c.id, c.granted])) as Grants;
}

/* ── the Windows privacy settings ───────────────────────────────────────── */

export type OsAccess = "granted" | "denied" | "restricted" | "not-determined" | "unknown";

/**
 * What Windows itself thinks. Consent here means nothing if Settings →
 * Privacy → Microphone is switched off, and that failure is otherwise
 * completely silent: the stream opens and delivers digital silence forever,
 * which reads to a user as "it just doesn't hear me".
 */
export function osAccess(kind: "microphone" | "camera"): OsAccess {
  try {
    return systemPreferences.getMediaAccessStatus(kind) as OsAccess;
  } catch {
    return "unknown";
  }
}

/** Opens the Windows privacy page for a capability the OS is blocking. */
export async function openOsSettings(kind: "microphone" | "camera"): Promise<void> {
  await shell.openExternal(`ms-settings:privacy-${kind}`);
}

export const consentStore = new ConsentStore();
