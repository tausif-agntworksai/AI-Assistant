"""Speech to text backends."""

from __future__ import annotations

import logging

from .base import Transcriber, Transcript

log = logging.getLogger(__name__)

__all__ = ["Transcriber", "Transcript", "create_transcriber"]


def create_transcriber(cfg=None) -> Transcriber:
    """Build the configured recogniser.

    Three layers, each one only reached when the one before it came up short:

        fast local model → accurate local model → cloud (only if a key is set)

    Local-first by design: audio only leaves the machine after both local
    passes have already failed to make sense of it, and only when the user has
    deliberately configured a cloud key.
    """
    if cfg is None:
        from ..config import settings

        cfg = settings.stt

    from .local_whisper import LocalWhisper

    local: Transcriber = LocalWhisper(
        model=cfg.model,
        compute_type=cfg.compute_type,
        cpu_threads=cfg.cpu_threads,
        language=cfg.language,
        beam_size=cfg.beam_size,
        best_of=cfg.best_of,
    )

    if cfg.escalate and cfg.accurate_model and cfg.accurate_model != cfg.model:
        from .tiered import TieredTranscriber

        accurate = LocalWhisper(
            model=cfg.accurate_model,
            compute_type=cfg.compute_type,
            cpu_threads=cfg.cpu_threads,
            language=cfg.language,
            # The second pass exists to be right, not quick — a wider beam is
            # exactly what we're paying for by escalating at all.
            beam_size=max(5, cfg.beam_size),
            best_of=max(5, cfg.best_of),
        )
        local = TieredTranscriber(local, accurate, min_confidence=cfg.min_confidence)
        log.info("Speech recognition: %s → %s on low confidence",
                 cfg.model, cfg.accurate_model)
    else:
        log.info("Speech recognition: local %s (no escalation)", cfg.model)

    if not cfg.cloud_api_key:
        if cfg.backend == "cloud":
            log.warning("stt.backend is 'cloud' but CLOUD_STT_API_KEY is unset — "
                        "staying local")
        return local

    try:
        from .cloud import CloudTranscriber, FallbackTranscriber

        cloud = CloudTranscriber(cfg.cloud_provider, cfg.cloud_api_key, cfg.language)
    except Exception as exc:  # noqa: BLE001
        log.warning("Cloud speech recognition unavailable (%s) — staying local", exc)
        return local

    if cfg.backend == "cloud":
        log.info("Speech recognition: %s (cloud only)", cloud.name)
        return cloud

    log.info("Speech recognition: local, falling back to %s", cloud.name)
    return FallbackTranscriber(local, cloud)
