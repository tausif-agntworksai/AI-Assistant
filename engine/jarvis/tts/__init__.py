"""Text to speech backends."""

from __future__ import annotations

import logging

from .base import NullSpeaker, Speaker, chunk_text

log = logging.getLogger(__name__)

__all__ = ["Speaker", "NullSpeaker", "chunk_text", "create_speaker"]


def create_speaker(cfg=None, player=None) -> Speaker:
    """Build the configured speaker, degrading rather than failing.

    Order matters: Edge needs the network, SAPI is offline but English-only,
    and NullSpeaker keeps the assistant usable (silently) if both fail.
    """
    if cfg is None:
        from ..config import settings

        cfg = settings.tts

    if cfg.backend == "none":
        return NullSpeaker()

    if cfg.backend == "edge":
        try:
            from .edge import EdgeSpeaker

            return EdgeSpeaker(
                voice_hi=cfg.voice_hi,
                voice_en=cfg.voice_en,
                rate=cfg.rate,
                volume=cfg.volume,
                player=player,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Edge TTS unavailable (%s) - falling back to Windows SAPI", exc)

    try:
        from .sapi import SapiSpeaker

        return SapiSpeaker()
    except Exception as exc:  # noqa: BLE001
        log.error("No speech output available (%s) - running silently", exc)
        return NullSpeaker()
