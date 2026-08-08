"""Logging: rotating file + console, with UTF-8 forced on Windows.

The UTF-8 part is not incidental — this assistant logs Devanagari transcripts,
and the default Windows console encoding (cp1252) raises UnicodeEncodeError on
them, which would otherwise crash the audio thread mid-utterance.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys

from . import paths

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: int | str = logging.INFO, quiet: bool = False) -> None:
    paths.ensure_dirs()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)

    file_handler = logging.handlers.RotatingFileHandler(
        paths.ENGINE_LOG, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
    root.addHandler(file_handler)

    if not quiet:
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except (ValueError, OSError):
                    pass
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        root.addHandler(console)

    # These are chatty at DEBUG and tell us nothing we want.
    for noisy in ("urllib3", "httpx", "httpcore", "asyncio", "faster_whisper", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
