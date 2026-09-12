# Hearing: which models, and why

Everything here is measured on this machine, not taken from a model card. The
question behind it was *"which voice model will be the best? it takes two to
four attempts to understand the wake word."*

The short answer is that those are two separate components with two separate
answers, and neither one is fixed by a bigger model:

| | keep | because |
|---|---|---|
| **Wake word** | `hey_jarvis` at threshold **0.30** | there is no bigger model to switch to; 0.50 was throwing away 22% of utterances |
| **Speech-to-text** | `base`, escalating to `small` | `large-v3-turbo` is the only more accurate one and it runs 13× slower than real time |

---

## 1. The wake word

`openWakeWord`'s `hey_jarvis`, run on 80 ms frames. There are only four
pretrained models (`hey_jarvis`, `alexa`, `hey_mycroft`, `hey_rhasspy`) and only
one of them says "jarvis", so **the model is not a choice** — the threshold is.

### Why it took two to four attempts

20 synthesised "hey jarvis" clips (5 voices × 4 phrasings), each streamed frame
by frame through the detector in 4 gain/noise conditions, recording the peak
score per utterance — the number a threshold is actually compared against.

Peak score, by voice (worst of 16 measurements each):

| voice | worst | median | verdict |
|---|---|---|---|
| en-US Aria | 0.996 | 0.999 | never at risk |
| en-GB Ryan | 0.979 | 0.994 | never at risk |
| hi-IN Madhur | 0.274 | 0.966 | at risk |
| en-IN Prabhat | 0.136 | 0.995 | at risk |
| en-IN Neerja | **0.079** | 0.416 | usually at risk |

**Every miss came from an accented voice.** The native en-US and en-GB clips
never dropped below 0.979 in any condition; en-IN Neerja fell to 0.079. This is
bias in the pretrained model, and it is why the assistant seemed to work fine in
demos and not in use.

Against that, the things that are *not* the wake word:

| utterance | worst peak |
|---|---|
| "jarvis" (the bare name) | **0.304** |
| "take a screenshot" | 0.038 |
| "hey google" | 0.025 |
| "hey there" | 0.018 |
| "what is the time" | 0.008 |
| "chrome kholo", "volume kam karo", "battery kitni bachi hai", "set a timer for five minutes", "stop the music please" | ≤ 0.001 |

The gap is enormous. Ordinary speech — including the assistant's own commands,
which is what this microphone hears all day — sits at essentially zero. The only
thing anywhere near the wake word is the name **"jarvis"** on its own.

So the whole trade-off is one table:

| threshold | heard | missed | false fires |
|---|---|---|---|
| 0.50 | 62/80 | **18 (22%)** | 0/40 |
| 0.40 | 67/80 | 13 | 0/40 |
| **0.30** | **73/80** | **7 (9%)** | **1/40** |
| 0.20 | 78/80 | 2 | 3/40 |
| 0.10 | 79/80 | 1 | 3/40 |

**0.30 is the shipped default.** It cuts misses by 61%, and the single false fire
it admits is the word "jarvis" said alone — which waking the assistant is hardly
the wrong response to.

Erring low is deliberate and asymmetric: a miss costs a whole repeated sentence,
a false wake costs a 140 ms cue and a discarded second of silence.

`tests/test_wake_tune.py::test_the_shipped_threshold_matches_what_was_measured`
pins the number so it cannot drift back without someone reading this first.

### Say it as two words, not two halves

The clearest single effect in the data, and a counter-intuitive one:

| phrasing | en-IN Prabhat | en-IN Neerja | hi-IN Madhur |
|---|---|---|---|
| "hey jarvis?" | 0.997 | 0.985 | 0.999 |
| "hey jarvis" | 0.997 | 0.301 | 0.995 |
| "hey, **pause** jarvis" | **0.136** | 0.295 | 0.492 |

A pause between the two words costs up to 0.86 of score. The model was trained
on the phrase said as one connected unit, so **saying it more slowly and more
separately after being missed makes the next attempt score lower** — which is the
exact instinct a missed wake word produces. `--tune-wake-word` says this out
loud before it starts measuring.

### Tuning it to your own voice

Nobody can pick this number for you from here — it depends on your accent, your
microphone gain, your distance and your room. So:

```
python -m jarvis --tune-wake-word          # measure and recommend
python -m jarvis --tune-wake-word --apply  # and write it to config.yaml
```

It records the peak score for each time you say it, then listens to your room
saying nothing and records the highest score *that* produced. The
recommendation sits at 75% of your weakest utterance, and if your room scores
higher than your voice it says that no threshold separates them rather than
inventing one — that case is a microphone problem, not a settings problem.

### What was ruled out

- **Level compensation / auto-gain.** Measured first, because it was the
  obvious hypothesis: a clip amplitude-scaled from 0.02 to 0.95 scored 0.999 at
  every step. The detector normalises internally. Scores *do* move with gain on
  marginal clips, but not in a consistent direction — en-IN Neerja's "hey
  jarvis" scored 0.301 clean and 0.878 quiet — so there is nothing to steer.
- **A bigger wake-word model.** There isn't one. Fixing the accent bias properly
  means training a custom model on recordings of the actual user, which is a
  different project.

---

## 2. Speech-to-text

`faster-whisper` (CTranslate2), int8 on CPU, in two tiers: `base` for every
utterance, escalating to `small` when confidence is low.

**The slow pass runs at most once per recording.** Two things ask for it, and
on a bad utterance both do: `transcribe` escalates when the fast model comes
back unsure, and the orchestrator escalates again when the words matched no
skill. Since the audio and the settings are identical and greedy decoding is
deterministic, the second request was four seconds spent re-deriving a
transcript we already had — on exactly the utterances that were already the
slowest. It is now memoised for the recording it belongs to, keyed on the
audio's contents rather than the buffer's identity (numpy hands out freed
memory again, so identity would answer a later utterance with an earlier
speaker's words).

### Scored on task success, not word error rate

WER is the wrong metric here. "chrome kolo" is a WER failure and a complete
success, because normalisation and the rule matcher absorb it. So each of 12
Hinglish command clips is scored on whether it reached the **right skill**.

| model | condition | routed ok | time/clip | × realtime |
|---|---|---|---|---|
| tiny | clean | 9/12 | 0.64 s | 0.27× |
| tiny | quiet+noise | 10/12 | 0.72 s | 0.30× |
| **base** | **clean** | **11/12** | **1.37 s** | **0.57×** |
| **base** | **quiet+noise** | **11/12** | **1.45 s** | **0.61×** |
| small | clean | 11/12 | 4.17 s | 1.74× |
| small | quiet+noise | 10/12 | 4.17 s | 1.74× |
| large-v3-turbo | clean | 12/12 | 32.07 s | 13.39× |

Two conclusions:

- **`small` is 3× slower than `base` and no more accurate.** It stays only as
  the low-confidence escalation tier, which is what it is good for — a second
  opinion on the utterances `base` was unsure about, not a default.
- **`large-v3-turbo` is the only model that gets everything right, and it is
  unusable.** 32 seconds to transcribe a 2.4-second command. Being right 40
  seconds late is being wrong. (Its degraded pass was not run: the latency
  already settles it, and the download is 1.6 GB.)

### The one clip every model got wrong

All four sizes mangled `लैपटॉप सुला दो` ("put the laptop to sleep"), each in a
different way and none of them into English words:

```
tiny            "Laptop s'olado."
base            "L'attompsulado."
small           "La tente soulado."
large-v3-turbo  "Laptop sous l'adot."
```

`large-v3-turbo` was the only one that then routed correctly — and only because
its version happened to keep "Laptop" as a recognisable token.

This is the case that makes the point. Fixing it with a bigger model costs
1.6 GB and 30 seconds per command. Fixing it in `nlu/normalize.py` costs five
lines mapping the mangled spellings to `sulao`, and works on `base`:

```python
"solado": "sulao", "sulado": "sulao", "soulado": "sulao",
"sulade": "sulao", "sulaado": "sulao",
```

Every transcript above is now a regression test in
`tests/test_mangled_transcripts.py`, verbatim, apostrophes and French guesses
included.

---

## 3. Matching, and the risk of trying too hard

Recovering more commands offline means accepting weaker fuzzy matches, and that
has a failure mode worse than not understanding.

Measured fuzzy scores for transcripts that were being discarded:

| transcript | best match | score |
|---|---|---|
| "laptop solado" | `sleep_pc` | 74 |
| "brightness calm" | **`brightness_up`** | 76 |
| "brightness come caro" | **`brightness_up`** | 72 |
| "paj minakt ka tamar laga do" | `set_timer` | 77 |

The threshold was 78, so all of them fell through — a whole class of "it did not
understand me" that was a threshold, not a model. It is now **70**.

But look at rows two and three: the closest match to a garbled *"turn the
brightness down"* is **`brightness_up`**. Lowering the threshold alone would have
made the assistant turn the brightness up when asked to turn it down, which is
worse than failing. Two guards make the lower threshold safe:

1. **A fuzzy match can never invert a direction.** For skills that come in
   opposing pairs, the utterance must contain a word that distinguishes the two —
   derived from the skills' own `examples`, so a new phrasing improves this for
   free rather than needing a hand-kept synonym list.
2. **A fuzzy match can never supply a magnitude.** `set_brightness(level: int = 60)`
   has a *default*, so nothing had been stopping a 76-scoring "brightness calm"
   from silently setting the screen to 60%. A number is a decision the utterance
   has to actually contain.

Both are tested with the real mangled transcripts that motivated them.

---

## Reproducing any of this

The benchmark scripts are not shipped — they synthesise their audio with
edge-tts, which means they need a network connection and produce optimistic
absolute numbers. What they are good for is the *ordering* between models and
the latency, which is measured on the CPU that has to run it.

For your own microphone, the shipped diagnostics are the real measurement:

```
python -m jarvis --list-devices        # is the right microphone selected?
python -m jarvis --test-mic            # level, language, confidence, transcript
python -m jarvis --tune-wake-word      # peak scores and a recommended threshold
```
