# Jarvis — a bilingual voice assistant for Windows

Say **“hey jarvis”**, then talk to your laptop in Hindi, English, or a mix of
the two. It opens apps, controls volume and brightness, manages windows, runs
timers, answers questions, and puts the machine to sleep — and it answers out
loud in whichever language you used.

```
  you    “chrome kholo”                  →  Chrome opens, "Chrome khol raha hoon."
  you    “kitna battery bacha hai”       →  "Battery 65 percent hai aur charge ho rahi hai."
  you    “volume pachaas kar do”         →  volume to 50%
  you    “laptop sula do”                →  "Laptop ko sula doon?"  →  “haan”  →  sleeps
  you    “paanch minute ka timer laga do”→  announces out loud when it fires
```

---

## How it fits together

Two processes. The **engine** (Python) owns the microphone, the models and
every Windows action. The **desktop app** (Electron) is the face — it spawns
and supervises the engine and renders its state. They talk over a localhost
WebSocket, so the UI holds no logic of its own and can reconnect freely.

```
                 ┌──────────────── engine (Python) ────────────────┐
  microphone ──▶ │ wake word → VAD → Whisper → rules ─┬─▶ skill    │ ──▶ Windows
                 │   (all local)                      └─▶ Claude   │
                 │                                                 │ ──▶ speech out
                 └────────────────────┬────────────────────────────┘
                          ws://127.0.0.1:8756
                 ┌────────────────────┴────────────────────────────┐
                 │ desktop app (Electron): orb, transcript, tray   │
                 └─────────────────────────────────────────────────┘
```

**Nothing is recorded or sent anywhere until the wake word fires.** While
idle, audio lives in a ring buffer and is scored by a local ONNX model. No
transcription, no storage, no network.

### The listening loop

```
IDLE ──"hey jarvis" | Ctrl+Alt+J | click the orb──▶ LISTENING
LISTENING ──~700 ms of silence──▶ THINKING ──▶ ACTING ──▶ SPEAKING ──▶ IDLE
                                     │
                     rules match? ───┴─── no ──▶ Claude picks a tool
```

Common commands never leave the machine: `“chrome kholo”` is matched by local
rules in microseconds, with no API call and no cost. Only phrasings the rules
don't recognise go to Claude.

### Understanding two languages at once

Whisper auto-detects the language per utterance and often returns Hindi in
Devanagari. `nlu/normalize.py` transliterates it to Roman and collapses both
languages onto one spelling, so a single rule covers both word orders —
English puts the verb first (“open chrome”), Hindi puts it last (“chrome
kholo”).

```
  "क्रोम खोलो"  ──transliterate──▶  "chrome kholo"  ──┐
  "open chrome"                                      ├──▶  open_app(app="chrome")
  "chrome chalu karo"                                ──┘
```

---

## Setup

Needs Windows 10/11, Python 3.11+ (3.14 is what this was built and tested on)
and Node 20+.

```powershell
cd engine
powershell -ExecutionPolicy Bypass -File setup.ps1     # venv + dependencies
```

Add your Anthropic API key to `engine\.env` — everything works without it
except conversation, questions, translation and the fallback that understands
unusual phrasings:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Then check the environment and run it:

```powershell
.\run-engine.bat --doctor        # verifies models, mic, dependencies, keys
.\run-engine.bat                 # start listening
```

For the full app with the HUD:

```powershell
cd ..\desktop
npm install
npm start
```

For a one-double-click launcher, add a Desktop shortcut:

```powershell
npm run shortcut                        # points at the packaged .exe if you've built one
npm run shortcut -- -Source repo        # always run from this folder instead
```

`-Source repo` launches Electron against the repo, so `engine\config.yaml` and
`engine\.env` are the ones it reads — handy while you're still editing them.
`-Source packaged` requires `npm run dist` and keeps its settings in
`%LOCALAPPDATA%\Jarvis` instead. The shortcut needs the repo to stay put; move
the folder and you'll need to re-run this.

First launch downloads ~150 MB of models (Whisper `base`, the wake word, and
the VAD) into `%LOCALAPPDATA%\Jarvis\models`. After that it runs offline
except for Claude and the neural voices.

---

## Trying it without a microphone

Voice, the HUD and `--text` all funnel through the same `handle_text()`, so a
typed command exercises exactly the path a spoken one takes.

```powershell
.\run-engine.bat --text "chrome kholo"
.\run-engine.bat --text "computer band kar do" --dry-run   # resolve, don't run
.\run-engine.bat --list-skills                             # everything it can do
.\run-engine.bat --test-tts "नमस्ते, मैं जार्विस हूँ"        # check the Hindi voice
.\run-engine.bat --test-mic 5                              # record 5s and transcribe
.\run-engine.bat --list-devices                            # pick the right mic
```

Run the tests with `python -m pytest` from `engine\`.

---

## Permissions

An always-listening program that can shut down your machine needs one place
where every action is classified — not a judgement call scattered across
seventy skill functions. Each skill declares a risk tier, and
`registry.execute()` is the single chokepoint that enforces it and writes the
audit log (visible in the HUD's Activity tab).

| Tier | Behaviour | Examples |
|---|---|---|
| `SAFE` | runs immediately | open an app, volume, media, any query |
| `CONFIRM` | asks out loud, needs a yes | sleep, lock, close an app or window, typing |
| `CRITICAL` | always asks | shutdown, restart, sign out, empty the bin, send a message |

Confirmation is deliberately strict: anything that isn't a clear yes counts as
a no, and with nothing able to ask, gated actions are refused rather than
allowed. Shutdown and restart also run on a 15-second delay — say **“cancel
shutdown”** to stop one.

Messaging never sends by itself. `send_whatsapp` opens the chat with the
message typed and waits for you to press send.

---

## What it can do

72 skills across 12 categories. `--list-skills` prints them all with example
phrasings.

**Apps & windows** — open or close any installed or Store app by fuzzy name,
switch to a running one, minimise/maximise/close/snap windows, switch virtual
desktops, show the desktop.

**Web** — open sites, Google searches, YouTube search-and-play
(`“youtube pe lofi chalao”`).

**System** — lock, sleep, restart, shut down, sign out, empty the Recycle Bin,
cancel a pending shutdown.

**Audio** — set/raise/lower volume, mute, per-app volume
(`“set chrome volume to 30”`), play/pause/next/previous in any player.

**Display & device** — brightness, night light, battery with time remaining,
CPU/memory/disk, Wi-Fi, Bluetooth, network status, any Settings page.

**Files** — screenshots, open known folders, search your personal folders,
read/write the clipboard, dictate into the focused window.

**Productivity** — timers and alarms that announce themselves out loud,
reminders and notes that survive a restart.

**Knowledge** — free-form conversation, questions, Hindi↔English translation,
summarise the clipboard, weather, news headlines.

**Messaging** — WhatsApp and email drafts, prepared but never sent for you.

### Adding a skill

One decorator is the single source of truth for the rule matcher, Claude's
tool schema and the permission gate. Write the function; nothing else needs
updating.

```python
@skill(
    name="open_app",
    description="Open or launch an application by name",
    risk=Risk.SAFE,
    category="apps",
    params={"app": "Name of the app, e.g. chrome, whatsapp"},
    examples=["open chrome", "chrome kholo", "whatsapp chalu karo"],
)
def open_app(app: str) -> object:
    ...
    return ok("Opening Chrome.", "Chrome khol raha hoon.")
```

Skills return both languages; the orchestrator picks based on how you were
speaking.

---

## Configuration

Running from source, `config.yaml` and `.env` live in `engine\`. In an
**installed** build they live in `%LOCALAPPDATA%\Jarvis\` — both are created
from their examples on first launch, so that's where to paste your API key.
Notable settings:

| Setting | Default | Why |
|---|---|---|
| `audio.input_device` | `null` | Picks a real mic automatically, skipping virtual inputs (DroidCam, Stereo Mix, VB-Audio) that Windows sometimes makes the default. Pin one with `--list-devices`. |
| `stt.model` | `base` | Benchmarked on this laptop: `base` = 0.63× realtime (~1.5 s per command), `small` = 2.0× (~4.6 s) with no better Hindi accuracy. |
| `stt.cpu_threads` | `4` | More was measurably *slower* — 16 threads ran ~40% worse than 4. |
| `wake_word.threshold` | `0.5` | Raise toward 0.7 if it triggers on its own. |
| `brain.model` | `claude-opus-5` | Thinking stays on: with it disabled this model can emit a tool call as plain text that silently never runs. |
| `brain.effort` | `low` | Keeps spoken replies quick. |
| `tts.voice_hi` | `hi-IN-MadhurNeural` | This machine has no Hindi SAPI voice, so Windows can't speak Hindi offline. |

---

## Packaging

```powershell
cd desktop
npm run dist      # freezes the engine, then builds the NSIS installer
```

The installer lands in `desktop\release` (~212 MB — most of it is the Whisper
runtime and ONNX libraries). PyInstaller uses one-folder mode deliberately:
one-file unpacks the ONNX and CTranslate2 native libraries to a temp directory
on every launch, which is slow and a common source of load failures.

The build refuses to package an engine that can't start — `build-engine.ps1`
runs the frozen executable's own `--doctor` and fails the build if it doesn't
pass, so a broken bundle can't reach an installer.

---

## Known limits

- **Hindi speech recognition is imperfect on the `base` model.** Short Hindi
  commands transcribe well; longer ones can garble. Those fall through to
  Claude, which usually recovers the intent. Set `stt.model: small` to trade
  ~3× latency for some accuracy, or add a cloud key (below) for the best
  Hinglish accuracy.
- **Optional cloud recognition.** Set `CLOUD_STT_PROVIDER` (`openai` or
  `deepgram`) and `CLOUD_STT_API_KEY` in `.env` and the local model gets a
  cloud retry whenever it comes back empty — audio only leaves the machine
  after the local pass has already failed. `stt.backend: cloud` sends
  everything instead.
- **Neural voices need internet.** `edge-tts` is the default because this
  machine has no Hindi SAPI voice. Offline it falls back to Windows SAPI,
  which is English-only — installing the Windows Hindi language pack adds
  `Microsoft Hemant` for offline Hindi.
- **Night light and Bluetooth open Settings** rather than toggling directly.
  Windows exposes no supported API for either; the alternatives are undocumented
  registry blobs that change between builds.
- **Wi-Fi toggling needs administrator rights.** Without them it opens the
  Wi-Fi settings page and says so.
- **Brightness needs a WMI-controllable panel** — the built-in laptop display
  qualifies; most external monitors don't.

---

## Layout

```
engine/
  jarvis/
    orchestrator.py     the listening state machine; where everything meets
    audio/              capture, wake word, VAD, playback, decoding
    stt/                faster-whisper + the guards against its failure modes
    tts/                Edge neural voices, SAPI fallback
    nlu/                normalise → rules → Claude
    skills/             every capability, one module per category
    permissions.py      risk tiers, the confirmation gate, the audit log
    app_index.py        fuzzy index of installed apps
    server.py           localhost HTTP + WebSocket API for the HUD
  tests/                pytest suite
desktop/
  src/main.ts           window, tray, global hotkey
  src/engine.ts         spawns and supervises the Python engine
  renderer/index.html   the HUD
```
