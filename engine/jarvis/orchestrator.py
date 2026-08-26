"""The listening state machine — where every part of the assistant meets.

    IDLE ──wake word / hotkey / HUD──▶ LISTENING ──trailing silence──▶ THINKING
    THINKING ──rules or Claude──▶ ACTING ──▶ SPEAKING ──▶ IDLE
                                              │
                                              └──▶ FOLLOW-UP (no wake word)

Voice, `--text`, and the HUD all funnel into `handle_text()`, so testing a
command from the CLI exercises exactly the path a spoken command takes. Only
the way the words arrive differs.

Two things here exist specifically so nobody has to repeat themselves:

  **The follow-up window.** For a few seconds after Jarvis finishes speaking,
  it keeps listening without the wake word. A correction ("nahi, chrome") or a
  second command lands immediately instead of needing "hey jarvis" again.

  **Re-decode before giving up.** A transcript that matches no skill is
  evidence of a mishearing, not of a missing feature. The same audio — still
  in memory — goes back through the larger speech model before the assistant
  admits defeat, and only then does it ask you to say it again.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import winutil
from .announce import set_handler as set_announce_handler
from .audio.capture import AudioCapture, FrameAccumulator, RingBuffer, rms_level
from .audio.enhance import is_too_quiet
from .audio.player import AudioPlayer
from .audio.vad import SegmentResult, SpeechSegmenter, create_vad
from .audio.wakeword import create_wakeword
from .bus import Event, State, bus
from .config import settings
from .nlu.confirm import is_affirmative
from .nlu.rules import route
from .permissions import Capability, Risk, consent, gate
from .security import session
from .skills import load_all, registry
from .skills.registry import SkillContext

log = logging.getLogger(__name__)

# How often to push a microphone level to the HUD (in captured blocks).
_LEVEL_EVERY = 8

# How many consecutive misheard utterances to ask about before going quiet.
# Past this the problem is the microphone or the room, and repeating "sorry?"
# at someone is worse than silence.
_MAX_REASKS = 2


@dataclass
class Turn:
    """What one utterance amounted to.

    `understood` is the interesting field: it separates "I did something"
    from "those were words but they meant nothing to me", which is the signal
    used to re-decode the audio rather than blame the user.
    """

    reply: str = ""
    understood: bool = False
    skill: str = ""


class Orchestrator:
    def __init__(self) -> None:
        self.cfg = settings
        self.capture: AudioCapture | None = None
        self.player = AudioPlayer(settings.audio.output_device)
        self.speaker = None
        self.transcriber = None
        self.wake = None
        self.vad = None
        self.segmenter: SpeechSegmenter | None = None

        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._trigger = threading.Event()  # hotkey / HUD asked us to listen
        self._speak_lock = threading.RLock()
        self._busy = threading.Lock()  # one utterance handled at a time
        self._audio_lock = threading.RLock()
        self._awaiting_confirmation = False
        self._want_audio = False
        self._followup_until = 0.0
        #: Last acknowledgement spoken per language, so the next one differs.
        self._last_ack: dict[str, str] = {}
        self._misses = 0
        self._recent_peak = 0.0
        self._quiet_warned = False
        self.last_language = "en"

    # -- lifecycle ----------------------------------------------------------

    def start(self, with_audio: bool = True) -> None:
        bus.set_state(State.STARTING)
        load_all()

        from .app_index import app_index
        from .skills.productivity import restore_timers
        from .tts import create_speaker

        app_index.build()
        restore_timers()

        self.speaker = create_speaker(self.cfg.tts, player=self.player)
        set_announce_handler(self._on_announcement)
        gate.set_confirmer(self._confirm_by_voice if with_audio else self._confirm_headless)

        self._want_audio = with_audio
        if not with_audio:
            bus.set_state(State.IDLE)
            log.info("Engine ready (text mode — no microphone)")
            return

        from .stt import create_transcriber

        self.transcriber = create_transcriber(self.cfg.stt)
        self.wake = (
            create_wakeword(
                self.cfg.wake_word.model, self.cfg.wake_word.threshold,
                self.cfg.wake_word.cooldown_sec,
            )
            if self.cfg.wake_word.enabled
            else None
        )
        if not self.cfg.wake_word.enabled:
            log.info("Wake word disabled by config — use the hotkey or the HUD")
        self.vad = create_vad(self.cfg.vad.backend)
        self.segmenter = self._make_segmenter()

        # Load the Whisper weights now, so the first real command isn't
        # waiting on a 500 MB download and a model build.
        threading.Thread(target=self._warmup, name="warmup", daemon=True).start()

        if self._may_listen():
            self._start_audio()
        else:
            log.info("Microphone held closed — %s", self._blocked_reason())

        bus.set_state(State.IDLE)
        trigger = "say 'hey jarvis'" if self._wake_available else "use the hotkey"
        log.info("Engine ready — %s to talk", trigger)

    def _make_segmenter(self, no_speech_timeout_sec: float | None = None) -> SpeechSegmenter:
        vad_cfg = self.cfg.vad
        return SpeechSegmenter(
            self.vad,
            sample_rate=self.cfg.audio.sample_rate,
            threshold=vad_cfg.threshold,
            silence_ms=vad_cfg.silence_ms,
            min_speech_ms=vad_cfg.min_speech_ms,
            max_utterance_sec=vad_cfg.max_utterance_sec,
            patience_silence_ms=vad_cfg.patience_silence_ms,
            no_speech_timeout_sec=(
                vad_cfg.no_speech_timeout_sec
                if no_speech_timeout_sec is None
                else no_speech_timeout_sec
            ),
        )

    @property
    def _wake_available(self) -> bool:
        return bool(getattr(self.wake, "available", False))

    def _warmup(self) -> None:
        try:
            self.transcriber.warmup()
        except Exception as exc:  # noqa: BLE001
            log.warning("Speech model warmup failed: %s", exc)
        self._warm_acknowledgement()

    def _warm_acknowledgement(self) -> None:
        """Synthesize the wake-word replies now, while nobody is waiting.

        The acknowledgement plays *before* the microphone opens, so paying for
        a cold network round-trip on the first "hey jarvis" would mean a
        multi-second wait at exactly the wrong moment. Doing it at startup
        means the first one is served from cache like every one after it.
        """
        if self.cfg.wake_word.acknowledge != "voice" or self.speaker is None:
            return
        synthesize = getattr(self.speaker, "synthesize", None)
        if not callable(synthesize):
            return
        # Every phrase, not just one: which gets picked is random, so any that
        # isn't cached would be the slow one at the worst moment.
        warmed = 0
        for language in ("en", "hi"):
            for phrase in self.cfg.wake_word.ack_texts(language):
                try:
                    synthesize(phrase, language)
                    warmed += 1
                except Exception as exc:  # noqa: BLE001 - a cold cache is survivable
                    log.debug("Could not pre-render %r: %s", phrase, exc)
        if warmed:
            log.info("Wake-word replies ready (%d phrases)", warmed)

    def stop(self) -> None:
        self._stop_audio()
        if self.speaker:
            self.speaker.close()
        self.player.close()
        set_announce_handler(None)
        gate.set_confirmer(None)
        log.info("Engine stopped")

    def trigger_listen(self) -> None:
        """Start listening without the wake word (hotkey or HUD button)."""
        self._trigger.set()

    # -- the microphone gate ------------------------------------------------
    #
    # Two independent reasons the microphone may stay shut, and neither is a
    # setting buried in the UI: nobody is signed in, or the microphone
    # permission was never granted. An assistant that keeps listening through
    # either of those would make its own sign-in screen decorative.

    def _may_listen(self) -> bool:
        if not self._want_audio:
            return False
        if not consent.allows(Capability.MICROPHONE):
            return False
        if self.cfg.security.session_required and not session.active:
            return False
        return True

    def _blocked_reason(self) -> str:
        if not consent.allows(Capability.MICROPHONE):
            return "microphone permission not granted"
        if self.cfg.security.session_required and not session.active:
            return "waiting for sign-in"
        return "audio disabled"

    @property
    def listening_enabled(self) -> bool:
        return self.capture is not None and self.capture.running

    def resume(self) -> None:
        """Called after a successful sign-in, or after permissions change."""
        if self._may_listen():
            self._start_audio()
        else:
            log.info("Not resuming the microphone — %s", self._blocked_reason())

    def pause(self) -> None:
        """Called on sign-out. Releases the microphone, not just the routing."""
        self._stop_audio()
        bus.set_state(State.IDLE)

    def apply_consent(self) -> None:
        """Re-evaluate the microphone after the permission screen was used."""
        if self._may_listen():
            self._start_audio()
        else:
            self._stop_audio()

    def _start_audio(self) -> None:
        with self._audio_lock:
            if self.capture is not None and self.capture.running:
                return
            self.capture = AudioCapture(
                device=self.cfg.audio.input_device,
                sample_rate=self.cfg.audio.sample_rate,
                block_size=self.cfg.audio.block_size,
            )
            try:
                self.capture.start()
            except Exception as exc:  # noqa: BLE001
                self.capture = None
                log.error("Could not open the microphone: %s", exc)
                bus.publish(Event.ERROR, message=f"Microphone unavailable: {exc}")
                return

            self._running.set()
            self._thread = threading.Thread(target=self._loop, name="listen", daemon=True)
            self._thread.start()
            log.info("Microphone open — listening")

    def _stop_audio(self) -> None:
        with self._audio_lock:
            self._running.clear()
            if self.capture:
                self.capture.stop()
                self.capture = None
            thread, self._thread = self._thread, None
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def microphone_health(self) -> dict[str, Any]:
        """What the HUD shows when someone asks "why can't it hear me?"."""
        capture = self.capture
        return {
            "open": bool(capture and capture.running),
            "reason": None if self.listening_enabled else self._blocked_reason(),
            "recent_peak": round(self._recent_peak, 4),
            "too_quiet": self._quiet_warned,
            "overflows": getattr(capture, "overflow_count", 0),
            "dropped": getattr(capture, "dropped_blocks", 0),
            "wake_word": self._wake_available,
        }

    # -- the audio loop -----------------------------------------------------

    def _loop(self) -> None:
        capture = self.capture
        if capture is None:
            return
        ww_frames = FrameAccumulator(getattr(self.wake, "frame_size", 1280))
        vad_frames = FrameAccumulator(self.vad.frame_size)
        preroll = RingBuffer(
            int(self.cfg.audio.sample_rate * self.cfg.vad.preroll_ms / 1000)
        )
        listening = False
        block_count = 0

        for block in capture.blocks():
            if not self._running.is_set():
                break

            block_count += 1
            if block_count % _LEVEL_EVERY == 0:
                level = rms_level(block)
                self._recent_peak = max(self._recent_peak * 0.995, level)
                bus.publish(Event.LEVEL, level=level)

            # Barge-in: the wake word cuts off a reply that's still playing.
            if self.player.is_playing:
                if listening:
                    # Playback started after we'd already begun listening —
                    # the user got in during synthesis. They take priority, so
                    # stop talking and fall through to keep capturing them.
                    log.info("Reply cut short — already listening to the user")
                    self.speaker.stop()
                else:
                    if self.cfg.tts.barge_in and self._wake_available:
                        for frame in ww_frames.push(block):
                            if self.wake.triggered(frame):
                                log.info("Barge-in — stopping playback")
                                self.speaker.stop()
                                self._begin_listening(preroll, vad_frames)
                                listening = True
                                break
                    continue

            if not listening:
                preroll.push(block)

                if self._trigger.is_set():
                    self._trigger.clear()
                    self._begin_listening(preroll, vad_frames)
                    listening = True
                    continue

                # Follow-up: for a few seconds after speaking we accept a
                # reply without the wake word, so "no, the other one" works
                # the way it would with a person.
                if time.monotonic() < self._followup_until:
                    self._followup_until = 0.0
                    self._begin_listening(preroll, vad_frames, follow_up=True)
                    listening = True
                    continue

                if self._wake_available:
                    for frame in ww_frames.push(block):
                        if self.wake.triggered(frame):
                            if self._acknowledge():
                                # We just spoke. Everything buffered is our own
                                # voice, and seeding the segmenter with it would
                                # have the assistant transcribe itself.
                                self.capture.drain()
                                preroll.clear()
                                ww_frames.reset()
                            self._begin_listening(preroll, vad_frames)
                            listening = True
                            break
                continue

            # LISTENING — accumulate until the segmenter says the utterance ended.
            for frame in vad_frames.push(block):
                result = self.segmenter.push(frame)
                if result in (SegmentResult.WAITING, SegmentResult.SPEAKING):
                    continue

                listening = False
                audio = self.segmenter.audio
                preroll.clear()
                ww_frames.reset()
                vad_frames.reset()

                if result in (SegmentResult.COMPLETE, SegmentResult.TIMEOUT):
                    self._process_utterance(audio, truncated=result is SegmentResult.TIMEOUT)
                else:
                    log.debug("Utterance discarded (%s)", result.value)
                    bus.set_state(State.IDLE)

                if self.capture:
                    self.capture.drain()  # don't transcribe our own reply
                break

        log.debug("Listen loop finished")

    def _begin_listening(
        self,
        preroll: RingBuffer,
        vad_frames: FrameAccumulator,
        follow_up: bool = False,
    ) -> None:
        vad_frames.reset()
        # A follow-up gets a short patience: if nobody speaks we should be back
        # to idle quickly rather than sitting with the microphone open.
        self.segmenter = self._make_segmenter(
            no_speech_timeout_sec=self.cfg.assistant.followup_sec if follow_up else None
        )
        self.segmenter.reset(preroll=preroll.read())
        preroll.clear()
        bus.set_state(State.LISTENING, follow_up=follow_up)

    def _acknowledge(self) -> bool:
        """Answer the wake word, so the user knows they were heard.

        Returns True if it made a sound — the caller then drops the pre-roll,
        because that buffer now holds our own voice rather than the user's.

        Deliberately synchronous. Speaking on a worker thread would overlap
        the user's first words with our own, and with no echo cancellation the
        microphone would transcribe both. A short phrase costs ~400 ms, and
        after the first one it is served from the TTS cache, so the wait is
        local and consistent rather than a network round-trip every time.
        """
        bus.publish(Event.LOG, message="wake word detected")

        mode = self.cfg.wake_word.acknowledge
        if mode == "none" or self.speaker is None:
            return False

        try:
            if mode == "chime":
                return self._play_chime()

            language = self.last_language
            phrase = self._pick_acknowledgement(language)
            if not phrase:
                return False
            bus.set_state(State.SPEAKING)
            self.speaker.speak(phrase, language)
            return True
        except Exception as exc:  # noqa: BLE001 - never miss a command over a chirp
            log.debug("Acknowledgement failed: %s", exc)
            return False

    def _pick_acknowledgement(self, language: str) -> str:
        """One of the configured replies, never the same one twice running.

        Repeating verbatim is what makes an assistant sound like a recording;
        an immediate repeat is the only case anyone actually notices, so that
        is all this guards against.
        """
        choices = self.cfg.wake_word.ack_texts(language)
        if not choices:
            return ""
        if len(choices) > 1:
            previous = self._last_ack.get(language[:2])
            fresh = [c for c in choices if c != previous]
            if fresh:
                choices = fresh
        phrase = random.choice(choices)
        self._last_ack[language[:2]] = phrase
        return phrase

    def _play_chime(self) -> bool:
        """A 140 ms rising blip. Generated, so there is no asset to ship."""
        rate = 24000
        duration = 0.14
        t = np.linspace(0.0, duration, int(rate * duration), endpoint=False)
        # Two soft partials sliding upward read as friendly rather than alarm-like.
        tone = 0.5 * np.sin(2 * np.pi * (620 + 300 * t / duration) * t)
        tone += 0.2 * np.sin(2 * np.pi * (1240 + 600 * t / duration) * t)
        # Raised-cosine envelope: an abrupt edge would click.
        envelope = np.sin(np.pi * np.linspace(0.0, 1.0, t.size)) ** 1.5
        self.player.play((tone * envelope * 0.28).astype(np.float32), rate)
        return True

    def _open_followup(self) -> None:
        """Arm the follow-up so the listen loop picks it up on its next block.

        This is a latch, not the window itself: the loop consumes it within a
        block or two and then hands the actual waiting to a segmenter built
        with `followup_sec` of patience. Keeping the latch short-lived means a
        stale arm can't reopen the microphone minutes later.
        """
        if float(self.cfg.assistant.followup_sec) > 0 and self.listening_enabled:
            self._followup_until = time.monotonic() + 1.0

    # -- processing ---------------------------------------------------------

    def _stt_hint(self) -> str:
        """Vocabulary to bias the decoder toward — the apps on this machine.

        Whisper mangles proper nouns it has no reason to expect. Telling it
        that "Obsidian" and "Rufus" are words that exist here turns a whole
        class of "it never opens the right app" into a solved problem.
        """
        try:
            from .app_index import app_index

            names = [e.name for e in app_index.entries[:14]]
            return ", ".join(names)
        except Exception:  # noqa: BLE001
            return ""

    def _process_utterance(self, audio: np.ndarray, truncated: bool = False) -> None:
        bus.set_state(State.THINKING)

        if is_too_quiet(audio, self.cfg.audio.sample_rate):
            self._warn_quiet_microphone()

        hint = self._stt_hint()
        try:
            transcript = self.transcriber.transcribe(
                audio, self.cfg.audio.sample_rate, hint=hint
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Transcription failed")
            bus.publish(Event.ERROR, message=f"transcription failed: {exc}")
            bus.set_state(State.IDLE)
            return

        if transcript.is_empty:
            self._not_understood(heard_something=transcript.filtered)
            return

        bus.publish(Event.TRANSCRIPT, text=transcript.text, language=transcript.language,
                    confidence=transcript.language_probability,
                    accuracy=round(transcript.confidence, 3), truncated=truncated)

        language = self._resolve_language(transcript)
        turn = self._handle(transcript.text, language, source="voice", resolve=False)

        # Words that route to nothing are usually the wrong words. The audio is
        # still here, so ask the bigger model before asking the user.
        if not turn.understood:
            turn = self._retry_with_better_hearing(audio, transcript, hint, turn)

        if turn.understood:
            self._misses = 0

        if turn.reply:
            # Speak on a worker thread so the listen loop keeps consuming audio
            # while the reply plays. Blocking here would park the only thread
            # that can notice the wake word, which is what barge-in needs.
            self.speak_async(turn.reply, self.last_language)
        elif not turn.understood:
            self._not_understood(heard_something=True)
        else:
            bus.set_state(State.IDLE)
            self._open_followup()

    def _retry_with_better_hearing(
        self,
        audio: np.ndarray,
        first: Any,
        hint: str,
        turn: Turn,
    ) -> Turn:
        """Second opinion from the accurate model, on the same recording."""
        escalate = getattr(self.transcriber, "escalate", None)
        if not callable(escalate):
            return turn

        from .nlu.normalize import normalize

        try:
            better = escalate(audio, self.cfg.audio.sample_rate,
                              previous=first, hint=hint)
        except Exception as exc:  # noqa: BLE001
            log.warning("Re-decode failed: %s", exc)
            return turn

        if better.is_empty or normalize(better.text) == normalize(first.text):
            return turn

        log.info("Re-decoded %r as %r", first.text, better.text)
        bus.publish(Event.TRANSCRIPT, text=better.text, language=better.language,
                    confidence=better.language_probability,
                    accuracy=round(better.confidence, 3), corrected=True)
        language = self._resolve_language(better)
        return self._handle(better.text, language, source="voice", resolve=False)

    def _warn_quiet_microphone(self) -> None:
        if self._quiet_warned:
            return
        self._quiet_warned = True
        log.warning("Microphone input is very quiet — recognition will suffer")
        bus.publish(
            Event.ERROR,
            message="Your microphone is very quiet. Raise its level in Windows "
                    "sound settings, or pick a different one with --list-devices.",
        )

    def _not_understood(self, heard_something: bool) -> None:
        """Say "again?" and keep listening, rather than silently giving up.

        The old behaviour dropped straight back to idle, which is why the same
        command had to be said three times: each attempt needed a fresh wake
        word, and nothing ever told the user it had failed.
        """
        if not heard_something:
            log.debug("Nothing intelligible in the utterance")
            bus.set_state(State.IDLE)
            return

        self._misses += 1
        hi = self.last_language.startswith("hi")

        if self._misses > _MAX_REASKS:
            # Say so once and stop. Past this the microphone or the room is the
            # problem, and a third "sorry?" only makes that more annoying.
            log.info("Giving up after %d unclear utterances", self._misses)
            self._misses = 0
            self.speak_async(
                "मैं ये समझ नहीं पाया।" if hi else "I didn't catch that one.",
                self.last_language,
            )
            return

        prompt = "फिर से बोलिए?" if hi else "Sorry — say that again?"
        bus.publish(Event.LOG, message="asked for a repeat", misses=self._misses)
        self.speak_async(prompt, self.last_language)

    def handle_text(self, text: str, language: str = "en", source: str = "text",
                    dry_run: bool = False) -> str:
        """Turn an utterance into an action and/or a spoken reply.

        The one place voice, CLI and HUD input converge — which is what makes
        `--text` a faithful rehearsal of the spoken path.
        """
        return self._handle(text, language, source, dry_run).reply

    def _handle(self, text: str, language: str = "en", source: str = "text",
                dry_run: bool = False, resolve: bool = True) -> Turn:
        """`resolve=False` means the caller already decided the language.

        The voice path has Whisper's own label and its confidence to work
        with; re-running detection here would throw both away and reach a
        different answer from the same words.
        """
        text = (text or "").strip()
        if not text:
            return Turn()

        if resolve:
            language = self._resolve_language(None, language, text)
        self.last_language = language
        ctx = SkillContext(language=language, transcript=text, source=source,
                           dry_run=dry_run, account=session.account)

        with self._busy:
            intent = route(text, threshold=float(self.cfg.brain.rules_threshold))

            if intent is not None:
                log.info("Routed locally: %s", intent)
                bus.publish(Event.ACTION, skill=intent.skill, args=intent.args,
                            via=intent.matched_by)
                bus.set_state(State.ACTING)
                result = registry.execute(intent.skill, intent.args, ctx)
                return Turn(reply=result.text(language), understood=True,
                            skill=intent.skill)

            turn = self._ask_brain(text, language, ctx)

        # Voice keeps its silence here on purpose: the caller still has the
        # recording and will re-decode it before saying anything. Typed input
        # has nothing left to try, so it gets an answer rather than nothing.
        if not turn.understood and not turn.reply and source != "voice":
            turn.reply = ("मैं ये समझ नहीं पाया।" if language.startswith("hi")
                          else "I didn't catch that one.")
        return turn

    def _ask_brain(self, text: str, language: str, ctx: SkillContext) -> Turn:
        from .nlu.llm import brain

        if not brain.available:
            log.info("No local rule matched and no API key is set: %r", text)
            return Turn(reply="", understood=False)

        if not consent.allows(Capability.NETWORK):
            log.info("Brain skipped — internet permission not granted")
            return Turn(
                reply=("इंटरनेट की इजाज़त नहीं है, इसलिए ये समझ नहीं पाया।"
                       if language.startswith("hi")
                       else "I need internet permission to work that one out."),
                understood=True,
            )

        bus.set_state(State.THINKING)
        result = brain.interpret(text, language, self._live_context())

        if result.error:
            bus.publish(Event.ERROR, message=result.error)
            return Turn(
                reply=("अभी दिमाग़ काम नहीं कर रहा।" if language.startswith("hi")
                       else "I couldn't reach my brain just now."),
                understood=True,
            )

        replies: list[str] = []
        for action in result.actions:
            bus.publish(Event.ACTION, skill=action.skill, args=action.args, via="llm")
            bus.set_state(State.ACTING)
            outcome = registry.execute(action.skill, action.args, ctx)
            spoken = outcome.text(language)
            if spoken:
                replies.append(spoken)

        # Claude's own words only get spoken when it didn't take an action —
        # otherwise the skill's confirmation is the more accurate reply.
        if not replies and result.speech:
            replies.append(result.speech)

        understood = bool(result.actions or result.speech.strip())
        return Turn(reply=" ".join(replies), understood=understood,
                    skill=result.actions[0].skill if result.actions else "")

    def _resolve_language(self, transcript=None, detected: str = "", text: str = "") -> str:
        """Which language to reply in.

        Answering a Hindi question in English is the single most jarring thing
        a bilingual assistant can do, so this is deliberate rather than a
        one-line guess. `detect_language` owns the policy; the config can pin
        one language if you'd rather it never switch.
        """
        from .nlu.normalize import detect_language

        preference = self.cfg.assistant.default_reply_language
        if preference in ("hi", "en"):
            return preference

        if transcript is not None:
            return detect_language(
                transcript.text,
                whisper_language=transcript.language,
                whisper_probability=transcript.language_probability,
                previous=self.last_language,
            )
        return detect_language(text, whisper_language=detected, previous=self.last_language)

    @staticmethod
    def _live_context() -> dict[str, Any]:
        """Volatile facts for the LLM. Kept out of the cached system prompt."""
        context: dict[str, Any] = {"time": time.strftime("%I:%M %p").lstrip("0")}
        window = winutil.foreground_window()
        if window.get("title"):
            context["foreground"] = window["title"][:80]
        try:
            import psutil

            battery = psutil.sensors_battery()
            if battery:
                context["battery"] = round(battery.percent)
        except Exception:  # noqa: BLE001
            pass
        return context

    # -- speaking -----------------------------------------------------------

    def speak(self, text: str, language: str = "en") -> bool:
        """Speak and block until done. Used where ordering matters."""
        if not text.strip() or self.speaker is None:
            return True
        if not consent.allows(Capability.SPEAKER):
            bus.publish(Event.REPLY, text=text, language=language, spoken=False)
            return True
        with self._speak_lock:
            bus.set_state(State.SPEAKING)
            bus.publish(Event.REPLY, text=text, language=language)
            ok = self.speaker.speak(text, language)
            if self.capture:
                # Drop whatever the microphone picked up of our own voice.
                self.capture.drain()
            return ok

    def speak_async(self, text: str, language: str = "en") -> None:
        """Speak without blocking the caller, returning to IDLE when finished."""
        if not text.strip() or self.speaker is None:
            bus.set_state(State.IDLE)
            self._open_followup()
            return

        def run() -> None:
            try:
                self.speak(text, language)
            finally:
                # Barge-in already moved us to LISTENING; don't stomp on it.
                if bus.state is State.SPEAKING:
                    bus.set_state(State.IDLE)
                self._open_followup()

        threading.Thread(target=run, name="speak", daemon=True).start()

    def _on_announcement(self, en: str, hi: str, kind: str) -> None:
        """A timer fired or a reminder came due — say it unprompted."""
        language = self.last_language
        bus.publish(Event.LOG, message=f"announcement ({kind})", kind=kind)
        self.speak(hi if language.startswith("hi") and hi else en, language)
        bus.set_state(State.IDLE)

    # -- confirmation -------------------------------------------------------

    def _confirm_by_voice(self, prompt_en: str, prompt_hi: str, risk: Risk) -> bool:
        """Ask out loud and listen for a yes.

        Runs on the listen thread, inside the skill call, so it borrows the
        microphone directly instead of going back through the state machine.
        """
        language = self.last_language
        prompt = prompt_hi if language.startswith("hi") and prompt_hi else prompt_en

        bus.set_state(State.CONFIRMING, prompt=prompt, risk=risk.value)
        bus.publish(Event.CONFIRM_REQUEST, prompt=prompt, risk=risk.value)
        self._awaiting_confirmation = True
        try:
            self.speak(prompt, language)
            answer = self._listen_briefly(self.cfg.permissions.confirm_timeout_sec)
        finally:
            self._awaiting_confirmation = False

        approved = bool(answer) and is_affirmative(answer)
        bus.publish(Event.CONFIRM_RESULT, approved=approved, heard=answer or "")
        log.info("Confirmation heard %r -> %s", answer, "yes" if approved else "no")
        return approved

    def _listen_briefly(self, timeout_sec: float) -> str:
        """Record one short answer. Returns '' on silence.

        Transcribed in `short_answer` mode, which matters more than it sounds:
        "haan", "ji" and "ok" are on the list of things Whisper hallucinates
        onto silence, so the ordinary filter threw away every spoken yes and
        the gate read it as a refusal.
        """
        if self.capture is None or self.transcriber is None:
            return ""

        self.capture.drain()
        segmenter = SpeechSegmenter(
            self.vad,
            sample_rate=self.cfg.audio.sample_rate,
            threshold=self.cfg.vad.threshold,
            silence_ms=500,       # answers are one word; end them quickly
            patience_silence_ms=700,
            min_speech_ms=150,
            max_utterance_sec=4,
            no_speech_timeout_sec=timeout_sec,
        )
        segmenter.reset()
        frames = FrameAccumulator(self.vad.frame_size)
        deadline = time.monotonic() + timeout_sec + 5

        for block in self.capture.blocks(timeout=0.3):
            if time.monotonic() > deadline:
                break
            for frame in frames.push(block):
                result = segmenter.push(frame)
                if result in (SegmentResult.WAITING, SegmentResult.SPEAKING):
                    continue
                if result in (SegmentResult.COMPLETE, SegmentResult.TIMEOUT):
                    try:
                        return self.transcriber.transcribe(
                            segmenter.audio, self.cfg.audio.sample_rate,
                            short_answer=True,
                        ).text
                    except Exception as exc:  # noqa: BLE001
                        log.error("Confirmation transcription failed: %s", exc)
                        return ""
                return ""  # silence or a blip — treat as no
        return ""

    @staticmethod
    def _confirm_headless(prompt_en: str, prompt_hi: str, risk: Risk) -> bool:
        """Text-mode confirmation, for `--text` and scripted runs."""
        try:
            answer = input(f"\n  {prompt_en} [y/N] ").strip()
        except (EOFError, KeyboardInterrupt):
            return False
        return is_affirmative(answer)


orchestrator = Orchestrator()
