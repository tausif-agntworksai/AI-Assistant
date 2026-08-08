"""Speech to text backends."""

from __future__ import annotations

import logging

from .base import Transcriber, Transcript

log = logging.getLogger(__name__)

__all__ = ["Transcriber", "Transcript", "create_transcriber"]


def create_transcriber(cfg=None) -> Transcriber:
    """Build the configured recogniser.

    Local-first by design: the cloud path is only wired up when a key is
    present, and even then it only runs after the local model has already
    failed to make sense of an utterance.
    """
    if cfg is None:
        from ..config import settings

        cfg = settings.stt

    from .local_whisper import LocalWhisper

    local = LocalWhisper(
        model=cfg.model,
        compute_type=cfg.compute_type,
        cpu_threads=cfg.cpu_threads,
        language=cfg.language,
        beam_size=cfg.beam_size,
    )

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

    log.info("Speech recognition: local %s, falling back to %s", cfg.model, cloud.name)
    return FallbackTranscriber(local, cloud)
