# Jarvis — a bilingual voice assistant for Windows

Say **“hey jarvis”**, then talk to your laptop in Hindi, English, or a mix of
the two. It opens apps, controls volume and brightness, manages windows, runs
timers, answers questions, and puts the machine to sleep — and it answers out
loud in whichever language you used.

```
  you    “chrome kholo”                  →  Chrome opens, "Chrome khol raha hoon."
  you    “kitna battery bacha hai”       →  "Battery 65 percent hai aur charge ho rahi hai."
  you    “volume pachaas kar do”         →  volume to 50%
  you    “laptop sula do”                →  sleeps
  you    “paanch minute ka timer laga do”→  announces out loud when it fires
```

---

## How it fits together

Two processes. The **engine** (Python) owns the microphone, the models and
every Windows action. The **desktop app** (Electron) is the face — it spawns
and supervises the engine, holds the sign-in, and renders its state. They talk
over a localhost WebSocket, so the UI holds no logic of its own and can
reconnect freely.

```
                 ┌──────────────── engine (Python) ────────────────┐
  microphone ──▶ │ wake word → VAD → Whisper → rules ─┬─▶ skill    │ ──▶ Windows
                 │   (all local)                      └─▶ Claude   │
                 │  ▲                                              │ ──▶ speech out
                 └──┼──────────────────┬────────────────────────────┘
        unlock ─────┘   http + ws://127.0.0.1:8756  (bearer token required)
                 ┌────────────────────┴────────────────────────────┐
                 │ desktop app: sign-in, permissions, orb, tray    │
                 └─────────────────────────────────────────────────┘
```

The microphone is the thing sign-in protects. The engine boots with its
listening loop shut and opens it only when the app says a verified, signed-in
person is present — so the login screen is not a page you click past, it is
what decides whether the machine is listening at all.

**Nothing is recorded or sent anywhere until the wake word fires.** While
idle, audio lives in a ring buffer and is scored by a local ONNX model. No
transcription, no storage, no network.

### The listening loop

```
IDLE ──"hey jarvis" | Ctrl+Alt+J | click the orb──▶ ACK ──▶ LISTENING
LISTENING ──trailing silence──▶ THINKING ──▶ ACTING ──▶ SPEAKING ──▶ IDLE
                                     │                        │
                     rules match? ───┴─── no ──▶ Claude       └──▶ FOLLOW-UP
                                                                (no wake word)
```

Common commands never leave the machine: `“chrome kholo”` is matched by local
rules in microseconds, with no API call and no cost. Only phrasings the rules
don't recognise go to Claude.

For six seconds after it finishes speaking, Jarvis keeps listening without the
wake word, so a correction (`“nahi, chrome”`) or a second command lands
straight away.

Nobody says the wake word in that window, so what arrives is as likely to be
the room as it is to be you. Anything that doesn't match a skill offline has to
come back clearly heard before it reaches the model, and is otherwise discarded
without a word — answering a conversation you were not having with it is most
of what "it wakes up on its own" actually is. Three follow-ups back to back end
the chain, so one detection cannot hold the microphone open indefinitely.

**The wake word answers back** with a 140 ms rising chime, so you know you were
heard rather than saying it twice. Latency is the whole constraint: the listen
loop drops every microphone frame while the cue plays, so the cue is dead time
and a spoken `“Yes?”` costs about a second of it on every single turn. Set
`wake_word.acknowledge` to `voice` to be answered in words instead — the spoken
cues are rendered once in the background and cached, and fall back to the chime
until they are ready — or `none` for silence.

The chime is levelled to the same target the spoken cues are normalised to.
It used to carry a hardcoded amplitude instead, leaving it 11.5 dB below the
spoken cue it stands in for — quiet enough on laptop speakers to read as no
acknowledgement at all, so people said the wake word a second time, which is
the exact problem the cue exists to prevent.

**And it has to hear the word twice.** openWakeWord scores each 80 ms frame on
its own, so a cough or a consonant off the television could clear the threshold
once and count as a detection. The wake word actually spoken holds the score up
across consecutive frames, so `wake_word.confirm_frames` of the last
`confirm_window` must clear it — 2 of 3 by default, which costs one frame of
latency and removes the whole class of one-frame flukes.

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

**It answers in the language you used.** Whisper's own language label is close
to a coin flip on a two-word romanised command — it called *“Chrome kholo”*
Polish in testing — so the label is one input, not the answer.
`nlu/normalize.detect_language` decides in a fixed order: Devanagari on screen
settles it; then distinctive romanised Hindi (*kholo*, *kitna*, *karo*); then
a confident Whisper label, including the languages Hindi gets mistaken for;
then distinctive English words; and finally the language the conversation was
already in, so `“aur battery?”` doesn't reset to English. Claude is told the
answer explicitly rather than left to infer it from words that look like both.

---

## Bringing your own model

The installer ships **no API key**. That is the point of it being an installer:
a key baked into something you hand to strangers is both extractable and billed
to whoever built it. So Jarvis is useful the moment it starts, and asks for a
key only when something actually needs one.

```
  "hey jarvis" · "hello" · "thanks" · "who are you"      ─┐
  "chrome kholo" · "volume 40" · "screenshot lo"          ├─ offline rules
  "battery kitni hai" · "5 minute ka timer laga do"      ─┘  no key, no network,
                                                             no cost, instant

  "explain quantum computing" · "translate this"         ──▶ needs a model
  anything phrased in a way no rule recognises                and therefore a key
```

**The router decides, and it shows you what it decided.** Each turn in the HUD
is badged `offline`, `local` or `AI`, and the footer keeps a running count for
the session — so "how much of this is the AI?" is a number you can read rather
than a claim you have to take. Of 77 skills, 72 are reachable with no key at
all; the model is consulted only when the rules do not recognise a phrasing,
and for the four genuinely conversational skills.

The first time an utterance needs the model and no key is set, Jarvis says so
out loud and opens settings — rather than going quiet and leaving you to guess
whether it heard you.

Six providers, and you pick the **model** as well as the provider:

| Provider | Why you might pick it |
|---|---|
| Anthropic (Claude) | What the prompts and tool-use were built against |
| Groq | Answers in a couple of hundred milliseconds — the difference between a conversation and a progress bar |
| OpenAI, Gemini, DeepSeek, Mistral | Whatever you already have a key for |

Model choice matters more here than in a chat app: a spoken reply that takes
six seconds feels broken even when it is correct, so every model in the list
carries a *fastest / balanced / most capable* badge and you can trade depth for
latency deliberately.

**Where the key lives.** In the desktop app, encrypted with your own Windows
account's key (DPAPI, via Electron `safeStorage`) — not in `localStorage`, and
never in a page. The renderer can set a key and ask whether one exists; there
is no call that reads one back. It is lent to the engine in memory for as long
as it runs, written to no file and stripped from every log line. A key is
verified against the provider *before* it is saved, so a typo fails at the
settings screen rather than on your first question.

### Speech recognition, optionally

Speech stays on this machine by default. Hindi mixed with English is the
hardest case for the local model, and a cloud recogniser is noticeably better
at it — Deepgram's `nova-3` is built for exactly that — so you can add one as a
fallback in the same panel. The behaviour is narrow and the panel says so:
**audio only leaves the machine after both local passes have already failed to
make sense of it.** A command the rules recognise never reaches a network, and
neither does one the local model transcribes cleanly.

---

## Being understood the first time

Whisper's response to a quiet or clipped recording is not "I'm not sure" — it
is a confident transcription of the wrong words. Four things attack that, in
the order the sound meets them.

**The recording is conditioned before recognition** (`audio/enhance.py`). Desk
rumble below 80 Hz is removed, the speech is brought up from a laptop
microphone's typical −38 dBFS to the −20 dBFS Whisper expects, and a quarter
second of silence is padded onto each end because the model regularly drops
the first phoneme of a clip that starts on speech. Gain is capped so silence
is never amplified into hiss the model can hallucinate onto.

**The endpoint adapts.** A fixed silence window is wrong in both directions:
cut early and *“chrome… kholo”* arrives as *“chrome”*, which routes nowhere;
wait long and every command feels slow. The window starts at 650 ms and
stretches to 1.3 s only while barely any speech has happened yet — which is
what a mid-thought pause looks like and what a finished command doesn't.

**Recognition has two tiers.** Every utterance is decoded by `base` first.
Only the ones that come back with low decoder confidence — or whose words
matched no skill at all — are re-decoded from the *same recording* by `small`.
Nobody speaks twice; the machine listens twice.

```
  fast pass ──confident? and it meant something?──▶ done      (~90% of commands)
                    │
                    └── no ──▶ accurate pass ──▶ best of the two
```

**And it says so when it fails.** A transcript that survives none of the above
gets *“Sorry — say that again?”* and an open microphone, not silence and a
fresh wake word. After two of those it stops asking, because by then the
problem is the room. The HUD also says outright when the microphone is muted,
blocked by Windows privacy settings, or simply too quiet — failures that are
otherwise completely silent.

One bug fixed here is worth naming: `“haan”`, `“ji”` and `“ok”` are all on the
list of things Whisper invents out of silence, so the filter that removes that
noise was deleting every spoken *yes* — and the confirmation gate reads
anything that isn't a clear yes as a refusal. Answers to a confirmation are
now transcribed in a mode that keeps them.

---

## Setup

Needs Windows 10/11, Python 3.11+ (3.14 is what this was built and tested on)
and Node 20+.

```powershell
cd engine
powershell -ExecutionPolicy Bypass -File setup.ps1     # venv + dependencies
```

Nothing else is required to start. Around fifty commands — opening apps,
volume, brightness, timers, battery, screenshots — are matched by offline rules
and need no API key at all. Conversation, questions and translation do, and
you add that **in the app** rather than in a file (see *Bringing your own
model* below). For a command-line-only setup you can still put one in
`engine\.env`:

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
copy .env.example .env      # then paste your Firebase web config into it
npm install
npm start
```

First launch asks which capabilities Jarvis may use, then (if Firebase is
configured) for a sign-in. Nothing listens until both are answered.

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
.\run-engine.bat --tune-wake-word                          # measure your own voice
```

### One spoken name, one program

The alias table maps what people say to what is installed — "calc" to
Calculator, "cmd" to Command Prompt. It used to be handed to *every* indexed
app whose name contained any of those words, and both halves of that were
wrong.

Matching was by substring, so `ps` was found inside **ma**`ps`, **ste**`ps`
and `wmpshare` — and Google Maps answered to "open powershell", scoring a
confident 100 with no ambiguity raised, which is the worst way to be wrong.
Matching is now on whole words.

Attaching the group to everything that matched was the other half. "cmd" is a
word in *Git CMD* as surely as in *Command Prompt*, so both claimed it and the
shorter name won. A spoken name now belongs to one program: the entry with the
best claim takes it, preferring an exact match, then the order the aliases
were written in, since that order is the author saying which meaning comes
first. A name the index merely *inferred* — the last word of a display name,
which is what lets "chrome" find Google Chrome — is scored slightly below one
somebody declared, so a coincidence loses to an intention.

Two entries in the table were also describing different programs as synonyms:
Windows Settings is not Control Panel, and Command Prompt is not Terminal.
Asking for the second got you the first.

### The browser means your browser

`“open the browser”` used to be a hardcoded synonym for Chrome, so on a machine
whose default is Edge it opened the wrong program — a lookup that was never
done, wearing the costume of a resolution bug. The default is now read from
the user's own association and attached to whichever entry names it best, so
the reply says *"Opening Microsoft Edge"* rather than *"Opening msedge"*.

### Where the time goes

Every turn is logged stage by stage, so "it feels slow" has a number attached
to it and an optimisation can be checked rather than argued about:

```
turn 2.48s  wake 0.14  record 0.90  stt 1.21  route 0.00 act 0.22  tts 0.31
```

The two microphone stages are the ones that were invisible before, and they
are worth reading first. `wake` is dead time by construction — the listen loop
drops every frame while the acknowledgement plays, so anything said over it is
lost, which is why the cue is a 140 ms chime rather than a spoken sentence.
`record` includes the trailing silence the segmenter waits through before it
decides you have stopped talking (`vad.silence_ms`), which is often the
largest single item in the budget and is a setting rather than a limit.

`route` and `act` are deliberately separate. They used to be one number, which
made the router look expensive when it costs a fraction of a millisecond and
the work was all in the skill.

Each turn also reports how it was endpointed — whether the utterance settled,
how long the trailing-silence window was, and how often the speaker went quiet
mid-sentence for longer than a quarter of a second:

```
turn 2.48s  ...  endpoint_ms=640 settled=True pauses=1 longest_pause_ms=384
```

That last number is there to settle a specific question. Most of `record` is
silence waited through to be sure you have stopped, and decoding could start
before that wait ends — but only if mid-sentence pauses are rare, because a
speculative decode that the speaker then talks over has to be finished and
thrown away. `“chrome… kholo”` is exactly that case. Rather than guess at how
common it is, the counter accumulates during ordinary use.

The same line appears in the HUD footer, and `tests/test_latency_budget.py`
fails the build if routing or entity resolution regresses by an order of
magnitude.

### If the wake word needs repeating

`--tune-wake-word` asks you to say "hey jarvis" five times, records the peak
detection score for each, then listens to your room saying nothing and records
the highest score *that* produced. It recommends a threshold between the two, or
tells you that no such threshold exists — which means the microphone or the room
is the problem, not the setting. Add `--apply` to write the result to
`config.yaml`.

Two things worth knowing before you run it, both measured (see
[docs/HEARING.md](docs/HEARING.md)):

- **Say it as one connected phrase.** A pause between "hey" and "jarvis" costs
  up to 0.86 of detection score. Slowing down and separating the words after a
  miss — the natural reaction — makes the next attempt score *lower*.
- **The pretrained model is weaker on accented English.** Native en-US and
  en-GB test clips never scored below 0.979; Indian-accented clips of the same
  phrase ranged from 0.079 to 0.999. The shipped threshold of 0.30 is set for
  the latter, and there is no larger wake-word model to move to — only four
  pretrained ones exist and only one of them says "jarvis".

Run the tests with `python -m pytest` from `engine\`.

---

## Without internet

Most of what Jarvis does never touches the network, and that half keeps working
with the router unplugged: opening and closing apps, volume, brightness,
windows, sleep and shutdown, timers and reminders, notes, the clipboard, file
search, screenshots, the camera, and answering calls. Around fifty commands.

Four things do need a connection — the model, the weather, the news, and web
search — and offline they say so:

> *"The weather needs the internet, and there's no connection right now.
> Everything on your computer still works."*

That replaced **"That didn't work."**, which was indistinguishable from a bug and
invited you to repeat a command that could not succeed. The check is only run for
skills that declared they need one, so a local command never waits on it, and an
inconclusive check counts as online — a wrong "you're offline" would block
something that works, which is worse than a slow failure.

Speech keeps working too: replies fall back from Edge's neural voices to the
Windows offline voice, and the wake-word acknowledgement falls back to a chime
if its spoken cues were never cached.

Two things are deliberately *not* gated. `open_website` and `open_url` still open
— they can point at `localhost`, a router or an intranet host, all of which
work offline — but the reply adds *"there's no internet right now, so it may
not load"* so a blank tab does not look like a failed command.

---

## Signing in

If `desktop\.env` names a Firebase project, Jarvis will not open the
microphone until someone is signed in. Accounts are **self-serve** — there is
no approval queue and no administrator:

```
sign up → verify your email → set up an authenticator → microphone opens
```

The authenticator is not optional. Once nobody is vetting new accounts, a
password on its own is one leak away from someone else's machine and someone
else's API key, so enrolling a TOTP app (Google Authenticator, Authy,
1Password) is part of signing up rather than a setting. That needs Firebase
Authentication with Identity Platform, which needs the **Blaze** plan — no
per-use charge at this volume, but a billing account has to be attached.

Copy `desktop\.env.example` to `desktop\.env` and fill in the web app config
from Firebase console → Project settings → Your apps. Leave it blank and
Jarvis runs unlocked and says so on screen, rather than showing a sign-in
prompt that couldn't succeed. `JARVIS_REQUIRE_AUTH=false` keeps the screens but
stops them gating the assistant — for local development, not for shipping.

Where the security actually is:

- **The refresh token never reaches a page.** It lives in the Electron main
  process, encrypted at rest with `safeStorage` — DPAPI, keyed to your Windows
  account. If the OS can't provide encryption, nothing is written at all;
  signing in again tomorrow beats a plaintext credential on disk.
- **Every launch re-checks with Firebase.** That is what makes a session read
  from disk trustworthy, and where a disabled or revoked account stops
  working — within one refresh cycle, not at the next restart. Offline, the
  last verified session is reused for seven days and then asks.
- **The engine's API needs a bearer token**, minted per launch and handed over
  through the environment. Loopback is not a boundary on a desktop: any page
  in any browser can `fetch("http://127.0.0.1:8756/command")`, and that API
  can shut the machine down. A web page can't read the token, and a request
  carrying any real browser `Origin` is refused outright even if it somehow
  could. Renderers are sandboxed, `contextIsolation` is on, every Chromium
  permission request is denied, and API keys are stripped from the log the
  moment they appear.
- **The audit log records the account.** On a shared machine, the history says
  who asked for what.

---

## Permissions

An always-listening program that can shut down your machine, type into the
focused window and open WhatsApp is asking for a lot of trust, and the honest
time to ask for it is before it starts. On first launch — and any time after,
from the tray — Jarvis shows every capability it has, what each one unlocks,
and a switch.

| | |
|---|---|
| Microphone, Speakers | hear you, answer out loud |
| Internet | Claude, weather, news, the neural voices |
| Apps, Windows | open/close/switch apps, manage windows |
| Device readings, Windows settings | battery and CPU, brightness and night light |
| Running system commands | PowerShell in the background — Windows exposes brightness, wi-fi and the Store-app list no other way |
| Your files, Screenshots | open and search your folders, capture the screen |
| Clipboard, Typing for you | *off by default* |
| Power controls | lock, sleep, sign out, restart, shut down |
| Messaging | *off by default* — never sends, only prepares |
| Camera | *off, and unused* — listed so a future camera feature arrives switched off |

These are enforced by the engine, not drawn by the UI: each id maps to a
`Capability` in `permissions.py`, and `registry.execute()` refuses a skill
whose capability is off — *before* the confirmation prompt, so turning one off
removes the ability rather than adding a question. Command-line access is
enforced at `winutil.run()`, the single place the assistant shells out.

Running the engine straight from a terminal leaves no consent file, and that
is treated as "not asked", which allows everything. Typing `python -m jarvis`
is its own consent; silently refusing to work for someone who launched it by
hand would be a puzzle, not a safeguard.

Windows has its own switch for the microphone and camera, and consent here
means nothing if that one is off — the stream opens and delivers digital
silence forever. The permission screen checks it and links straight to the
Windows privacy page when it's blocking.

### Risk tiers

Separately from *may it*, every action is classified by *how bad if it
misheard*. Each skill declares a tier, and `registry.execute()` is the single
chokepoint that enforces it and writes the audit log (visible in the HUD's
Activity tab).

| Tier | Behaviour | Examples |
|---|---|---|
| `SAFE` | runs immediately | open an app, volume, media, sleep, lock, any query |
| `CONFIRM` | asks out loud, needs a yes | close an app or window, typing, deleting a file |
| `CRITICAL` | always asks | shutdown, restart, sign out, empty the bin, send a message |

Sleep and lock sit in `SAFE` on purpose. Both are completely reversible — a
keypress or a password puts the machine back exactly where it was — while
confirming them costs a spoken prompt, a listening window and a second
recognition pass, about four seconds, on two of the commands people give most
often. The tier is a preference rather than a law, so
`permissions.risk_overrides` can put either back behind a prompt, or take one
away from `close_app`:

```yaml
permissions:
  risk_overrides:
    sleep_pc: confirm      # ask me again before sleeping
    close_app: safe        # stop asking before closing a window
```

Every routed command also reports the decision behind it — the intent, the
confidence, the risk tier it will be gated at, and which path it came down —
as one record rather than three things to join by hand:

```json
{"intent": "sleep_pc", "confidence": 0.95, "risk_level": 0,
 "execution_path": "fast", "requires_reasoning": false}
```

Confirmation is deliberately strict: anything that isn't a clear yes counts as
a no, and with nothing able to ask, gated actions are refused rather than
allowed. Shutdown and restart also run on a 15-second delay — say **“cancel
shutdown”** to stop one.

### Not having to repeat yourself

Every turn used to be an island. The model's history deliberately dropped any
turn that ran a skill — the reasoning being that tool calls need their results
echoed back to stay valid — so Jarvis could open Chrome and, one sentence
later, have no idea what "it" meant. Rule-routed commands, which are most of
them, left no trace at all.

What was missing was never the data. Skills already report what they resolved:
`open_app` hands back the application it matched, `send_message` the contact.
That was being discarded one line after it arrived. Now it lands in a
short-lived memory, and a pronoun can find it:

```
  you    “open whatsapp”              →  opens WhatsApp
  you    “message sana”               →  resolves Sana Ahmed
  you    “tell her I'll be late”      →  messages Sana Ahmed
  you    “close it”                   →  closes WhatsApp
```

It expires after five minutes and is never written to disk — a referent is a
half-finished sentence, not a setting, and an answer that is stale in a way
you cannot see is worse than no answer.

**Only people are substituted into what you said.** Pronouns for things are a
trap: `“turn it up”` is a volume command that works today, and rewriting every
"it" to the last application would break it. The one exception is `“close it”`,
where a closing verb makes the object unambiguous. Hindi keeps its case
particle — "usko" is "us" + "ko", and replacing the whole word with a name
leaves a sentence that parses as nothing.

### Contacts, and the names you actually say

The matching was never the weak part — the book was. Contacts could only arrive
one at a time, by voice, so on a fresh install every "text Sana" failed against
an empty list. Import one instead:

```
python -m jarvis --import-windows-contacts       # ~/Contacts, no export, no sign-in
python -m jarvis --import-contacts contacts.csv  # Google Contacts, Outlook, WhatsApp
python -m jarvis --import-contacts contacts.vcf  # iPhone, Android
python -m jarvis --list-contacts
```

Imports replace rather than accumulate, because an address book is a snapshot
of the truth and a contact deleted on the phone should not survive here.
Numbers you saved by voice live in a different file and win every collision —
someone who spelled a number out loud meant that number.

You never have to give the whole name, and you never have to know how the
contact was *spelled*. Phone books are not written to be spoken — the same
person is `Sana Ahmed` in one app, `sana ❤️` in another, `Sana(Office)` at
work and `सना` on a Hindi handset, and nobody pronounces a heart. Saying
"sana" finds all of them: a name is reduced to the words a person would
actually say before it is matched, reusing the same transliteration the command
language already uses.

Saying *more* narrows rather than widens. A subset scores a perfect match in
both directions under plain token matching, so "sana ahmed" used to tie with a
contact saved as plain `sana` and ask which you meant — punishing you for being
precise. A word you said that a stored name does not contain now counts against
it.

**"Message Rohit"** finds Rohit Sharma.
**"Email Sana"** finds the Sana who has an email address, because someone
reachable on WhatsApp and not by email is not a candidate for Gmail. And when
two people genuinely match, it asks once and then remembers:

```
you     “message Sana”        →  “Did you mean Sana Ahmed or Sana Khan?”
you     “message Sana Ahmed”  →  sends
you     “message Sana”        →  Sana Ahmed, no question
```

### Sending mail, not just composing it

"email Sana saying I'll send the report tomorrow" opens a pre-filled compose
window and stops, which is honest but leaves the last click to you every time.
Authorise Gmail once and the message actually goes:

```
python -m jarvis --authorise-gmail     # one browser consent, send-only access
python -m jarvis --forget-gmail        # back to the compose window
```

It needs an OAuth client of your own, because a desktop client secret shipped
inside a binary is a secret in name only — create one at
[Google Cloud credentials](https://console.cloud.google.com/apis/credentials)
as an **OAuth client ID → Desktop app**, enable the Gmail API, and save the
downloaded JSON as `gmail_client_secret.json` in the data directory.
`--doctor` prints the exact path and says which step is outstanding.

Three things are deliberate. The scope is **`gmail.send` and nothing else** —
it cannot read a single message, and an assistant that asks for mailbox access
in order to send mail is asking for the wrong thing. Consent is a thing you do
once, from the command line, never mid-sentence: being redirected to a browser
because you said "email Sana" would be startling. And when it is not
authorised, has expired, or the send fails, it falls back to the compose window
that always worked — an unauthorised install loses nothing it had.

The subject comes from the first clause of what you dictated, because nobody
says "subject colon" out loud and asking turns a one-sentence errand into an
interview.

### Which app a message goes to

WhatsApp, SMS, Telegram, Signal, Slack, Gmail and Google Chat sit behind one
`App` record that says how each addresses people — WhatsApp by phone number,
Gmail by email address — so "who" and "which platform" stay one lookup instead
of a resolver per service. Naming the platform anywhere in the sentence works:

```
“text Sana on WhatsApp saying I'll be there in 10 minutes”
“email Sana saying I'll send the report tomorrow”
“send a message to Sana on Google Chat saying I'm joining in 5 minutes”
```

Google Chat has no public URL that opens a conversation with a named person
from outside it, so it opens Chat with the message staged and you pick the
conversation — which is the most that can be done honestly.

Messaging is read back before anything opens: **"say hi to sana"** is confirmed
as *"Send “Hi” to Sana?"*, with the contact name it resolved to, so a
wrong match is caught before a stranger gets the message. After you agree, the
chat opens with the message typed and Jarvis presses send — but only once it
has confirmed the messaging app actually holds focus, so a slow window leaves
the draft sitting there rather than firing a stray keystroke into whatever was
in front. Set `messaging.auto_send: false` to always stop at the draft.

Naming an app (**"message rahul on telegram"**) overrides the default; otherwise
it uses `messaging.default_app`, which is WhatsApp. `compose_email` never
auto-sends — mail clients differ too much for a blind keystroke to mean
send.

If two saved contacts are both close to the name you said, it stops and asks
which one rather than picking the higher score. That is the one mistake here
that reaches a stranger and cannot be undone.

### Camera

**"take a photo"** / **"photo le lo"** does not open the Camera app. It opens the
device, discards eight frames while auto-exposure settles — otherwise the
first photo of a session is a dark rectangle — grabs one, releases the camera
and saves it to `Pictures\Jarvis`. The device is closed on every path out,
including the failures: a webcam left with its light on after one photo is not
something to ship.

**"open the camera"** is the separate, explicit request that hands you the
Windows Camera app.

### Calls

**"answer the call"**, **"pick up"**, **"call uthao"** — and **"hang up"**,
**"call kaat do"** — work for WhatsApp Desktop, Teams, Zoom and Google
Chat/Meet calls ringing on this computer.

Be aware of what this is. None of those apps has an API or a documented hotkey
for "answer", so the only available route is the one a person uses: find the
window that is ringing, bring it to the front, press the key that accepts. Two
consequences are designed for rather than hoped away:

- **Nothing is sent unless the ringing window was found and confirmed to have
  come forward.** A stray `Enter` or `Escape` into whatever happened to be
  focused could do anything. Not finding it says *"I can't see a call
  ringing"*, which is true and useful.
- **A browser call needs its tab to be the one you are looking at.** A
  background tab receives no keystrokes, and nothing outside the browser can
  change that, so the reply says so.

**Calls to your phone number are not here and cannot be.** A Windows machine has
no cellular radio and no access to the phone's call stack — that needs the
mobile client, which is not built. The reply says this rather than failing
silently, so it is heard once instead of discovered repeatedly.

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
| `stt.model` | `base` | The fast pass. Benchmarked here at 0.63× realtime (~1.5 s per command). |
| `stt.accurate_model` | `small` | The second opinion, ~2.0× realtime. Only reached when the fast pass is unsure or its words matched no skill, so most commands never pay for it. Set to `""` to turn escalation off. |
| `stt.min_confidence` | `0.62` | Decoder confidence below which the accurate pass runs. A clean short command sits around 0.75–0.9. |
| `stt.cpu_threads` | `4` | More was measurably *slower* — 16 threads ran ~40% worse than 4. |
| `vad.patience_silence_ms` | `1300` | The longer endpoint used while barely any speech has happened, so a pause between “chrome” and “kholo” doesn't end the turn. |
| `assistant.followup_sec` | `6` | How long the microphone stays open after a reply, so a correction needs no wake word. `0` disables it. |
| `wake_word.threshold` | `0.3` | Lower is more sensitive. Measured: 0.5 missed 22% of utterances and 0.3 misses 9%, while the only thing that falsely scores above 0.3 is the bare word "jarvis". Run `--tune-wake-word` to measure your own voice. |
| `security.require_session` | `false` | Refuse to open the microphone until someone signs in. The desktop app turns this on for its own launches; leaving it false keeps `python -m jarvis` usable from a terminal. |
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

- **Hindi speech recognition is imperfect.** Short Hindi commands transcribe
  well; longer ones can garble. The `small` re-decode and Claude between them
  usually recover the intent, but the first pass is still a `base` model.
  Setting `stt.model: small` makes every utterance accurate and slow; adding a
  cloud key (below) gives the best Hinglish accuracy of all.
- **The accurate tier is a ~480 MB download**, fetched in the background on
  first launch behind an assistant that already works. `stt.escalate: false`
  skips it entirely.
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
      enhance.py        conditioning that makes a quiet clip recognisable
    stt/                faster-whisper + the guards against its failure modes
      tiered.py         fast pass, then the accurate one when it wobbles
    tts/                Edge neural voices, SAPI fallback
    nlu/                normalise → detect language → rules → Claude
    skills/             every capability, one module per category
    permissions.py      capability consent, risk tiers, the audit log
    security.py         the API token, the session lock, redaction
    app_index.py        fuzzy index of installed apps
    server.py           localhost HTTP + WebSocket API for the HUD
  tests/                pytest suite
desktop/
  src/main.ts           windows, tray, hotkey, and which screen you're on
  src/engine.ts         spawns and supervises the Python engine
  src/config.ts         the Firebase project and what the gate enforces
  src/consent.ts        the capability manifest and its store
  src/llmKeys.ts        the user's API keys, encrypted with the OS keychain
  src/auth/             Firebase over REST, and the session that gates the mic
  renderer-src/         the React app: three entry points, one design system
    src/theme.css       one token vocabulary, light and dark
    src/screens/        Hud, Auth, Consent, Settings
    src/useEngine.ts    the WebSocket, the reconnect loop, the event reducer
  build/permissions.txt shown by the installer, before anything is installed
```
