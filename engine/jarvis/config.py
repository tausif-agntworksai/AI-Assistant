"""Configuration: typed defaults, overridden by config.yaml, secrets from .env.

Import `settings` (a module-level singleton) anywhere; call `reload_settings()`
if config.yaml changes at runtime.
"""

from __future__ import annotations

import os
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from . import paths


class AudioConfig(BaseModel):
    input_device: int | str | None = None
    output_device: int | str | None = None
    sample_rate: int = 16000
    block_size: int = 512


class WakeWordConfig(BaseModel):
    enabled: bool = True
    model: str = "hey_jarvis"
    threshold: float = 0.5
    cooldown_sec: float = 2.0
    #: What you hear when the wake word fires. config.yaml has documented this
    #: for a while, but the field was missing here — so pydantic dropped the
    #: setting and the acknowledgement never made a sound.
    acknowledge: Literal["voice", "chime", "none"] = "voice"
    #: What the spoken acknowledgement says — one is picked at random each
    #: time, so it doesn't feel like a recording.
    #:
    #: Every entry is a real word or phrase on purpose. "Mm-hmm" was the first
    #: attempt and came out as an unintelligible mumble: neural voices are
    #: trained on written language, and non-lexical sounds have no spelling
    #: they can pronounce reliably.
    #:
    #: Keep them short — this plays before the microphone opens, so each
    #: syllable is one the user waits through.
    ack_text_en: list[str] = Field(
        default_factory=lambda: ["Yes?", "I'm listening.", "Go ahead.", "I'm here."]
    )
    #: Feminine forms, to match the female Hindi voice.
    ack_text_hi: list[str] = Field(
        default_factory=lambda: ["जी?", "जी बोलिए.", "हाँ जी?", "सुन रही हूँ."]
    )

    @field_validator("ack_text_en", "ack_text_hi", mode="before")
    @classmethod
    def _accept_a_bare_string(cls, value: object) -> object:
        """Tolerate `ack_text_en: "Yes?"` as well as a list.

        The setting used to be a single string, and a config.yaml written
        against that shouldn't stop the assistant from starting.
        """
        if isinstance(value, str):
            return [value]
        return value

    def ack_texts(self, language: str) -> list[str]:
        texts = self.ack_text_hi if (language or "").startswith("hi") else self.ack_text_en
        return [t for t in texts if t and t.strip()]


class VadConfig(BaseModel):
    backend: Literal["silero", "energy"] = "silero"
    threshold: float = 0.5
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
    # One persona in both languages: Swara and Neerja are both warm female
    # voices. The old pairing was a male Hindi voice with a female English one,
    # which made the assistant sound like two different people.
    voice_hi: str = "hi-IN-SwaraNeural"
    voice_en: str = "en-IN-NeerjaExpressiveNeural"
    # Prosody. `+10%` read as brisk and clipped; slightly under normal speed
    # with a touch of lift sounds unhurried and friendly instead.
    rate: str = "-4%"
    pitch: str = "+3Hz"
    volume: str = "+0%"
    barge_in: bool = True

    def voice_for(self, language: str | None) -> str:
        return self.voice_hi if (language or "").lower().startswith("hi") else self.voice_en


class BrainConfig(BaseModel):
    enabled: bool = True
    model: str = "claude-opus-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    max_tokens: int = 2048
    rules_threshold: int = 78
    history_turns: int = 6

    @property
    def api_key(self) -> str:
        return os.environ.get("ANTHROPIC_API_KEY", "").strip()

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.api_key)


class PermissionsConfig(BaseModel):
    confirm_tier_enabled: bool = True
    allow_critical_without_confirm: bool = False
    confirm_timeout_sec: float = 12.0


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8756

    @property
    def resolved_port(self) -> int:
        """The port to actually bind.

        The desktop app picks a free port before spawning us and passes it in
        JARVIS_PORT — it may not be 8756 if something else already holds that.
        Ignoring it means the app health-checks a port nothing is listening on
        and waits out its whole timeout, which looks like the engine never
        starting.
        """
        raw = os.environ.get("JARVIS_PORT", "").strip()
        if raw:
            try:
                chosen = int(raw)
            except ValueError:
                log_config_warning("JARVIS_PORT is not a number: %r" % raw)
            else:
                if 1 <= chosen <= 65535:
                    return chosen
                log_config_warning("JARVIS_PORT out of range: %d" % chosen)
        return self.port


def log_config_warning(message: str) -> None:
    import logging

    logging.getLogger(__name__).warning("%s — using config.yaml instead", message)


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
