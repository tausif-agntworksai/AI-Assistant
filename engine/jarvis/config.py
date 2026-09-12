"""Configuration: typed defaults, overridden by config.yaml, secrets from .env.

Import `settings` (a module-level singleton) anywhere; call `reload_settings()`
if config.yaml changes at runtime.
"""

from __future__ import annotations

import os
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from . import paths


class AudioConfig(BaseModel):
    input_device: int | str | None = None
    output_device: int | str | None = None
    sample_rate: int = 16000
    block_size: int = 512


class WakeWordConfig(BaseModel):
    enabled: bool = True
    model: str = "hey_jarvis"
    # Measured, not chosen. Streaming 20 synthesised "hey jarvis" clips through
    # openWakeWord across four gain/noise conditions, the peak score per
    # utterance was:
    #
    #   threshold   heard    missed   false fires
    #      0.50     62/80      18        0/40      <- the old default
    #      0.40     67/80      13        0/40
    #      0.30     73/80       7        1/40      <- here
    #      0.20     78/80       2        3/40
    #
    # 0.50 was missing 22% of them, which is exactly the "I have to say it two
    # or three times" complaint. Every one of those misses came from the
    # Indian-accented voices — the native en-US and en-GB clips never scored
    # below 0.979 in any condition, so this is the pretrained model's bias, not
    # a microphone problem.
    #
    # The only thing that scores above 0.30 without being the wake word is the
    # bare word "jarvis" (0.30 at worst); every other utterance tested — "hey
    # google", "hey there", and the assistant's own commands — stayed under
    # 0.04. So the cost of coming down this far is that saying "jarvis" alone
    # may wake it, which is hardly wrong.
    #
    # Asymmetry is the reason for erring low: a miss costs a whole repeated
    # sentence, a false wake costs a 140 ms cue and a discarded second.
    # `--tune-wake-word` measures this on your own voice and microphone.
    threshold: float = 0.3
    cooldown_sec: float = 2.0
    # What you hear when the wake word fires. Without any acknowledgement there
    # is no way to know you were heard, so people say it twice.
    #   voice — a short spoken cue ("Yes?" / "जी?"), matched to the language of
    #           the last turn. Rendered in the background and cached; falls back
    #           to the chime until it is ready or if the network is unavailable.
    #   chime — a 140 ms rising two-note blip. Always available, never overlaps
    #           what you say next, and language-neutral.
    #   none  — silence, as it was before.
    #
    # `chime` is the default because the cue is dead time: the listen loop
    # drops every microphone frame while it plays, so whatever you say over it
    # is lost. A spoken "I'm listening." is about a second of that on every
    # single turn; the chime is 140 ms. Set this to `voice` if you would
    # rather be answered in words and don't mind waiting through them.
    acknowledge: Literal["voice", "chime", "none"] = "chime"
    #: How many of the last `confirm_window` frames must clear the threshold
    #: before this counts as a detection. openWakeWord scores every 80 ms
    #: frame independently, so a single spike — a cough, a consonant off the
    #: television — used to be enough on its own. Real speech holds the score
    #: up across several frames, so asking for 2 of 3 costs 80 ms of latency
    #: and removes the whole class of one-frame flukes. Set to 1 for the old
    #: single-frame behaviour.
    confirm_frames: int = 2
    confirm_window: int = 3


class VadConfig(BaseModel):
    backend: Literal["silero", "energy"] = "silero"
    # Measured, not chosen. Streaming 20 synthesised "hey jarvis" clips through
    # openWakeWord across four gain/noise conditions, the peak score per
    # utterance was:
    #
    #   threshold   heard    missed   false fires
    #      0.50     62/80      18        0/40      <- the old default
    #      0.40     67/80      13        0/40
    #      0.30     73/80       7        1/40      <- here
    #      0.20     78/80       2        3/40
    #
    # 0.50 was missing 22% of them, which is exactly the "I have to say it two
    # or three times" complaint. Every one of those misses came from the
    # Indian-accented voices — the native en-US and en-GB clips never scored
    # below 0.979 in any condition, so this is the pretrained model's bias, not
    # a microphone problem.
    #
    # The only thing that scores above 0.30 without being the wake word is the
    # bare word "jarvis" (0.30 at worst); every other utterance tested — "hey
    # google", "hey there", and the assistant's own commands — stayed under
    # 0.04. So the cost of coming down this far is that saying "jarvis" alone
    # may wake it, which is hardly wrong.
    #
    # Asymmetry is the reason for erring low: a miss costs a whole repeated
    # sentence, a false wake costs a 140 ms cue and a discarded second.
    # `--tune-wake-word` measures this on your own voice and microphone.
    threshold: float = 0.3
    silence_ms: int = 650
    # Used instead of `silence_ms` while barely any speech has happened yet,
    # which is what a false start or a mid-thought pause looks like. Cutting
    # "chrome… kholo" in half is the most common way a command gets lost.
    patience_silence_ms: int = 1300
    min_speech_ms: int = 200
    max_utterance_sec: int = 15
    # Audio kept from *before* speech started. Generous, because the wake word
    # and the command usually arrive in one breath.
    preroll_ms: int = 600
    # How long to wait for someone to start talking before giving up on a turn.
    no_speech_timeout_sec: float = 6.0


class SttConfig(BaseModel):
    backend: Literal["local", "cloud"] = "local"
    # The fast tier: what every utterance is decoded with first. Measured on
    # this machine at ~0.63x realtime (~1.5s for a spoken command).
    model: str = "base"
    # The accurate tier: only reached when the fast pass comes back unsure, or
    # when the words it produced matched no skill. ~2.0x realtime, so paying
    # for it on every utterance would be the wrong trade — paying for it on
    # the 10% that would otherwise need repeating is the right one.
    accurate_model: str = "small"
    escalate: bool = True
    # Decoder confidence (exp of the mean per-token logprob) below which the
    # accurate tier is consulted. A clean short command sits around 0.75-0.9.
    min_confidence: float = 0.62
    compute_type: str = "int8"
    language: str | None = None
    beam_size: int = 3
    best_of: int = 3
    # More is not better: 16 threads measured ~40% slower than 4 on this
    # 8-core CPU, from oversubscription.
    cpu_threads: int = 4

    @property
    def cloud_provider(self) -> str:
        return os.environ.get("CLOUD_STT_PROVIDER", "").strip().lower()

    @property
    def cloud_api_key(self) -> str:
        return os.environ.get("CLOUD_STT_API_KEY", "").strip()


class TtsConfig(BaseModel):
    backend: Literal["edge", "sapi", "none"] = "edge"
    voice_hi: str = "hi-IN-MadhurNeural"
    voice_en: str = "en-IN-NeerjaNeural"
    rate: str = "+10%"
    volume: str = "+0%"
    barge_in: bool = True

    def voice_for(self, language: str | None) -> str:
        return self.voice_hi if (language or "").lower().startswith("hi") else self.voice_en


class BrainConfig(BaseModel):
    enabled: bool = True
    # Which provider to start with, and which of its models. The user's own
    # choice — pushed down by the desktop app's settings screen — overrides
    # both; these are only the defaults for a fresh install or a bare
    # `python -m jarvis`. An empty model means "whatever that provider's
    # default is", so changing provider doesn't require changing both.
    provider: Literal["anthropic", "openai", "gemini", "groq", "deepseek", "mistral"] = (
        "anthropic"
    )
    model: str = ""
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    max_tokens: int = 2048
    # Fuzzy-match confidence below which the model is consulted. Measured on
    # garbled-but-recoverable transcripts, real commands scored 70-77 and were
    # being thrown away at 78 — a whole class of "it did not understand me" that
    # was a threshold, not a model. Safe to lower only because a fuzzy match can
    # no longer invert a direction (see _OPPOSITES in nlu/rules.py).
    rules_threshold: int = 70
    history_turns: int = 6


class PermissionsConfig(BaseModel):
    confirm_tier_enabled: bool = True
    allow_critical_without_confirm: bool = False
    confirm_timeout_sec: float = 12.0
    #: Re-tier individual skills without editing their declaration:
    #: ``{"close_app": "safe"}``. A skill declares the risk that is right in
    #: general; this is where one machine's owner says what is right for them.
    #: Values are Risk names — safe | confirm | critical.
    risk_overrides: dict[str, str] = Field(default_factory=dict)


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8756

    @property
    def bind_port(self) -> int:
        """The port to actually listen on.

        The desktop app scans for a free one and passes it in `JARVIS_PORT`,
        because 8756 may already be taken by something else — and nothing here
        read it. The app would then poll the port it chose while the engine sat
        on the one from config.yaml, and give up fifteen minutes later with
        "The engine did not respond in time." An occupied port is not a rare
        situation on a machine you are handing software to.
        """
        raw = os.environ.get("JARVIS_PORT", "").strip()
        if raw.isdigit() and 1 <= int(raw) <= 65535:
            return int(raw)
        return self.port


class SecurityConfig(BaseModel):
    """Whether the engine refuses to listen until someone has signed in.

    Off by default so `python -m jarvis` from a terminal still works — running
    the program by hand is its own authorisation. The desktop app sets
    `JARVIS_REQUIRE_SESSION=1` when it spawns the engine, which turns it on for
    every launch that goes through the installed application.
    """

    require_session: bool = False

    @property
    def session_required(self) -> bool:
        raw = os.environ.get("JARVIS_REQUIRE_SESSION", "").strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
        return self.require_session


class AssistantConfig(BaseModel):
    name: str = "Jarvis"
    default_reply_language: Literal["hi", "en", "auto"] = "auto"
    # Seconds to keep listening after a reply, so a correction or a second
    # command needs no wake word. 0 disables it.
    followup_sec: float = 6.0

    # How much the assistant is allowed to spend on a turn, and how much it is
    # allowed to say. A mode changes *how* something is done; it never changes
    # *what can be done*, so no mode adds or removes a single skill.
    #
    #   normal  — the shipped balance: rules first, the model when they miss.
    #   fast    — offline only. The rules answer or nothing does, which makes
    #             every turn instant and some turns unanswerable.
    #   deep    — let the model think harder on the turns that reach it.
    #   silent  — do everything, say nothing aloud; replies go to the window.
    #   offline — assume no network: no model, no neural voice, no web skills.
    mode: Literal["normal", "fast", "deep", "silent", "offline"] = "normal"

    # Suppresses everything the assistant would have said unprompted. Kept
    # separate from `mode` because wanting quiet for an hour is not the same
    # as wanting a different kind of assistant, and folding the two together
    # would mean choosing between a fast Jarvis and an undisturbed one.
    do_not_disturb: bool = False

    @property
    def speaks(self) -> bool:
        """Whether a reply is spoken aloud at all."""
        return self.mode != "silent"

    @property
    def may_use_model(self) -> bool:
        """Whether a turn the rules missed may reach the language model."""
        return self.mode not in ("fast", "offline")

    @property
    def effort(self) -> str | None:
        """Reasoning effort for this mode, or None to leave the config alone."""
        return {"deep": "high", "fast": "low"}.get(self.mode)


class MessagingConfig(BaseModel):
    """How "say hi to sana" turns into a sent message."""

    #: Used whenever the user names no app. WhatsApp because that is what the
    #: phrase means in practice here, but it is a setting rather than a constant
    #: so it can be someone else's default.
    default_app: Literal["whatsapp", "sms", "telegram", "signal", "slack"] = "whatsapp"

    #: Whether to press send, or leave the draft open with the cursor in it.
    #:
    #: On by default, because "send hi to sana" asks for a message to be sent and
    #: stopping one keystroke short is a strange place to stop. It is safe to
    #: default on only because of the two gates around it: the command is
    #: CRITICAL, so it is read back and confirmed before anything opens, and the
    #: keystroke is only sent once the messaging app is confirmed to hold focus.
    #: Turn it off to review every message before it goes.
    auto_send: bool = True

    #: How long to wait for the app's window to take focus before giving up on
    #: pressing send. Generous: WhatsApp Desktop cold-starting is slow, and the
    #: consequence of being impatient is a message left unsent, which the reply
    #: then says.
    focus_timeout_sec: float = 6.0


class Settings(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    brain: BrainConfig = Field(default_factory=BrainConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    assistant: AssistantConfig = Field(default_factory=AssistantConfig)
    messaging: MessagingConfig = Field(default_factory=MessagingConfig)


def bootstrap_config() -> None:
    """Seed config.yaml and .env on first run.

    Matters most for the installed build: without this there is nothing for
    the user to edit, and no obvious place to paste an API key.
    """
    if paths.CONFIG_EXAMPLE.exists() and not paths.CONFIG_FILE.exists():
        _copy(paths.CONFIG_EXAMPLE, paths.CONFIG_FILE)

    env_example = paths.ENGINE_DIR / ".env.example"
    if env_example.exists() and not paths.ENV_FILE.exists():
        _copy(env_example, paths.ENV_FILE)


def _copy(source, destination) -> None:
    try:
        paths.ensure_dirs()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError:
        # A read-only location is survivable; the built-in defaults still apply.
        pass


def load_settings() -> Settings:
    """Read config.yaml over the typed defaults.

    A missing or empty config.yaml is not an error — the defaults are the
    intended out-of-the-box behaviour.
    """
    bootstrap_config()
    if not paths.CONFIG_FILE.exists():
        return Settings()
    try:
        raw = yaml.safe_load(paths.CONFIG_FILE.read_text(encoding="utf-8")) or {}
        return Settings.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        # A typo in config.yaml must not stop the assistant from starting.
        import logging

        logging.getLogger(__name__).error(
            "config.yaml is invalid (%s) — falling back to defaults", exc
        )
        return Settings()


# Order matters: the files have to exist before python-dotenv reads them, or a
# freshly installed build would ignore its own .env until the second launch.
bootstrap_config()
load_dotenv(paths.ENV_FILE, override=False)

settings: Settings = load_settings()


def reload_settings() -> Settings:
    """Re-read config.yaml and .env. Used when settings change at runtime."""
    global settings
    load_dotenv(paths.ENV_FILE, override=True)
    settings = load_settings()
    return settings
