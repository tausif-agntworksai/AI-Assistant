"""The listening state machine — where every part of the assistant meets.

    IDLE ──wake word / hotkey / HUD──▶ LISTENING ──trailing silence──▶ THINKING
    THINKING ──rules or Claude──▶ ACTING ──▶ SPEAKING ──▶ IDLE

Voice, `--text`, and the HUD all funnel into `handle_text()`, so testing a
command from the CLI exercises exactly the path a spoken command takes. Only
the way the words arrive differs.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from . import winutil
from .announce import set_handler as set_announce_handler
from .audio.capture import AudioCapture, FrameAccumulator, RingBuffer, rms_level
from .audio.player import AudioPlayer
from .audio.vad import SegmentResult, SpeechSegmenter, create_vad
from .audio.wakeword import create_wakeword
from .bus import Event, State, bus
from .config import settings
from .nlu.confirm import is_affirmative
from .nlu.rules import route
from .permissions import Risk, gate
from .skills import load_all, registry
from .skills.registry import SkillContext

log = logging.getLogger(__name__)

# How often to push a microphone level to the HUD (in captured blocks).
_LEVEL_EVERY = 8


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
        self._awaiting_confirmation = False
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

        if not with_audio:
            bus.set_state(State.IDLE)
            log.info("Engine ready (text mode — no microphone)")
            return

        from .stt import create_transcriber

        self.transcriber = create_transcriber(self.cfg.stt)
        self.wake = create_wakeword(
            self.cfg.wake_word.model, self.cfg.wake_word.threshold,
            self.cfg.wake_word.cooldown_sec,
        )
        self.vad = create_vad(self.cfg.vad.backend)
        self.segmenter = SpeechSegmenter(
            self.vad,
            sample_rate=self.cfg.audio.sample_rate,
            threshold=self.cfg.vad.threshold,
            silence_ms=self.cfg.vad.silence_ms,
            min_speech_ms=self.cfg.vad.min_speech_ms,
            max_utterance_sec=self.cfg.vad.max_utterance_sec,
        )

        # Load the Whisper weights now, so the first real command isn't
        # waiting on a 500 MB download and a model build.
        threading.Thread(target=self._warmup, name="warmup", daemon=True).start()

        self.capture = AudioCapture(
            device=self.cfg.audio.input_device,
            sample_rate=self.cfg.audio.sample_rate,
            block_size=self.cfg.audio.block_size,
        )
        self.capture.start()

        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="listen", daemon=True)
        self._thread.start()

        bus.set_state(State.IDLE)
        trigger = "say 'hey jarvis'" if getattr(self.wake, "available", False) else "use the hotkey"
        log.info("Engine ready — %s to talk", trigger)

    def _warmup(self) -> None:
        try:
            self.transcriber.warmup()
        except Exception as exc:  # noqa: BLE001
            log.warning("Speech model warmup failed: %s", exc)

    def stop(self) -> None:
        self._running.clear()
        if self.capture:
            self.capture.stop()
        if self.speaker:
            self.speaker.close()
        self.player.close()
        set_announce_handler(None)
        gate.set_confirmer(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        log.info("Engine stopped")

    def trigger_listen(self) -> None:
        """Start listening without the wake word (hotkey or HUD button)."""
        self._trigger.set()

    # -- the audio loop -----------------------------------------------------

    def _loop(self) -> None:
        assert self.capture is not None
        ww_frames = FrameAccumulator(getattr(self.wake, "frame_size", 1280))
        vad_frames = FrameAccumulator(self.vad.frame_size)
        preroll = RingBuffer(
            int(self.cfg.audio.sample_rate * self.cfg.vad.preroll_ms / 1000)
        )
        listening = False
        block_count = 0

        for block in self.capture.blocks():
            if not self._running.is_set():
                break

            block_count += 1
            if block_count % _LEVEL_EVERY == 0:
                bus.publish(Event.LEVEL, level=rms_level(block))

            # Barge-in: the wake word cuts off a reply that's still playing.
            if self.player.is_playing:
                if listening:
                    # Playback started after we'd already begun listening —
                    # the user got in during synthesis. They take priority, so
                    # stop talking and fall through to keep capturing them.
                    log.info("Reply cut short — already listening to the user")
                    self.speaker.stop()
                else:
                    if self.cfg.tts.barge_in and getattr(self.wake, "available", False):
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

                if getattr(self.wake, "available", False):
                    for frame in ww_frames.push(block):
                        if self.wake.triggered(frame):
                            self._acknowledge()
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

                self.capture.drain()  # don't transcribe our own reply
                break

        log.debug("Listen loop finished")

    def _begin_listening(self, preroll: RingBuffer, vad_frames: FrameAccumulator) -> None:
        vad_frames.reset()
        self.segmenter.reset(preroll=preroll.read())
        preroll.clear()
        bus.set_state(State.LISTENING)

    def _acknowledge(self) -> None:
        """A short cue so the user knows the wake word landed."""
        bus.publish(Event.LOG, message="wake word detected")

    # -- processing ---------------------------------------------------------

    def _process_utterance(self, audio: np.ndarray, truncated: bool = False) -> None:
        bus.set_state(State.THINKING)
        try:
            transcript = self.transcriber.transcribe(audio, self.cfg.audio.sample_rate)
        except Exception as exc:  # noqa: BLE001
            log.exception("Transcription failed")
            bus.publish(Event.ERROR, message=f"transcription failed: {exc}")
            bus.set_state(State.IDLE)
            return

        if transcript.is_empty:
            log.debug("Nothing intelligible in the utterance")
            bus.set_state(State.IDLE)
            return

        bus.publish(Event.TRANSCRIPT, text=transcript.text, language=transcript.language,
                    confidence=transcript.language_probability, truncated=truncated)

        reply = self.handle_text(transcript.text, transcript.language, source="voice")
        if reply:
            # Speak on a worker thread so the listen loop keeps consuming audio
            # while the reply plays. Blocking here would park the only thread
            # that can notice the wake word, which is what barge-in needs.
            self.speak_async(reply, self.last_language)
        else:
            bus.set_state(State.IDLE)

    def handle_text(self, text: str, language: str = "en", source: str = "text",
                    dry_run: bool = False) -> str:
        """Turn an utterance into an action and/or a spoken reply.

        The one place voice, CLI and HUD input converge — which is what makes
        `--text` a faithful rehearsal of the spoken path.
        """
        text = (text or "").strip()
        if not text:
            return ""

        language = self._resolve_language(language, text)
        self.last_language = language
        ctx = SkillContext(language=language, transcript=text, source=source,
                           dry_run=dry_run)

        with self._busy:
            intent = route(text, threshold=float(self.cfg.brain.rules_threshold))

            if intent is not None:
                log.info("Routed locally: %s", intent)
                bus.publish(Event.ACTION, skill=intent.skill, args=intent.args,
                            via=intent.matched_by)
                bus.set_state(State.ACTING)
                result = registry.execute(intent.skill, intent.args, ctx)
                return result.text(language)

            return self._ask_brain(text, language, ctx)

    def _ask_brain(self, text: str, language: str, ctx: SkillContext) -> str:
        from .nlu.llm import brain

        if not brain.available:
            log.info("No local rule matched and no API key is set: %r", text)
            return (
                "मैं ये समझ नहीं पाया।" if language.startswith("hi")
                else "I didn't catch that one."
            )

        bus.set_state(State.THINKING)
        result = brain.interpret(text, language, self._live_context())

        if result.error:
            bus.publish(Event.ERROR, message=result.error)
            return ("अभी दिमाग़ काम नहीं कर रहा।" if language.startswith("hi")
                    else "I couldn't reach my brain just now.")

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

        return " ".join(replies)

    def _resolve_language(self, detected: str, text: str = "") -> str:
        """Which language to reply in.

        Whisper's own detection is the primary signal, but typed input (the
        CLI and HUD) arrives with no detection at all — so Devanagari in the
        text is treated as Hindi regardless of what was passed in.
        """
        from .nlu.normalize import looks_hindi

        preference = self.cfg.assistant.default_reply_language
        if preference in ("hi", "en"):
            return preference
        if (detected or "").lower().startswith("hi"):
            return "hi"
        return "hi" if looks_hindi(text) else "en"

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
            return

        def run() -> None:
            try:
                self.speak(text, language)
            finally:
                # Barge-in already moved us to LISTENING; don't stomp on it.
                if bus.state is State.SPEAKING:
                    bus.set_state(State.IDLE)

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
        """Record one short answer. Returns '' on silence."""
        if self.capture is None or self.transcriber is None:
            return ""

        self.capture.drain()
        segmenter = SpeechSegmenter(
            self.vad,
            sample_rate=self.cfg.audio.sample_rate,
            threshold=self.cfg.vad.threshold,
            silence_ms=500,       # answers are one word; end them quickly
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
                            segmenter.audio, self.cfg.audio.sample_rate
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
