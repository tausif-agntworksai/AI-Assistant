# -*- coding: utf-8 -*-
"""Test isolation.

`jarvis.paths` resolves the data directory at import time, so this has to run
before anything imports the package — which is exactly what pytest guarantees
for a root conftest. Without it the suite writes test timers, notes and audit
entries into the user's real %LOCALAPPDATA%\\Jarvis.
"""

import os
import tempfile
from pathlib import Path

_TEST_DATA_DIR = Path(tempfile.gettempdir()) / "jarvis-tests"
_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
os.environ["JARVIS_DATA_DIR"] = str(_TEST_DATA_DIR)

# Keep the suite offline and deterministic: no key means no API calls, even if
# the developer has one exported.
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("CLOUD_STT_API_KEY", None)
