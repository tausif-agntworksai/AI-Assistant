"""PyInstaller entry point.

A module (`python -m jarvis`) can't be frozen directly, so the packaged build
starts here and hands straight over to the same CLI.
"""

from __future__ import annotations

import multiprocessing
import sys

if __name__ == "__main__":
    # Without this, every worker process a frozen build spawns would re-run
    # the whole app instead of the worker function.
    multiprocessing.freeze_support()

    from jarvis.__main__ import main

    sys.exit(main())
