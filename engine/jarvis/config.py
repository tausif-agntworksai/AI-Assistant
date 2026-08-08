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
    threshold: float = 0.5
    cooldown_sec: float = 2.0


class VadConfig(BaseModel):
    backend: Literal["silero", "energy"] = "silero"
    threshold: float = 0.5
    silence_ms: int = 700
    min_speech_ms: int = 250
    max_utterance_sec: int = 15
    preroll_ms: int = 300


class SttConfig(BaseModel):
    backend: Literal["local", "cloud"] = "local"
    # Measured on this machine: `base` runs at ~0.63x realtime (~1.5s for a
    # spoken command) while `small` runs at ~2.0x (~4.6s) — and `small` was no
    # more accurate on Hindi commands. Latency is what makes a voice assistant
    # feel alive, so `base` is the default; raise to `small` if accuracy on
    # longer dictation matters more.
    model: str = "base"
    compute_type: str = "int8"
    language: str | None = None
    beam_size: int = 1
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


class AssistantConfig(BaseModel):
    name: str = "Jarvis"
    default_reply_language: Literal["hi", "en", "auto"] = "auto"


class Settings(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    brain: BrainConfig = Field(default_factory=BrainConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
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
