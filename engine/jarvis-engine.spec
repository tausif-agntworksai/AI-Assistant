# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Jarvis engine.

One-folder, not one-file: onnxruntime and ctranslate2 ship native DLLs that a
one-file build unpacks to a temp directory on every launch, which is slow and
a common source of load failures. The folder is bundled into the installer as
an extraResource, so the user never sees it.
"""

from PyInstaller.utils.hooks import collect_all

# These carry native libraries and/or on-disk model assets that PyInstaller's
# default analysis misses, so pull each one in wholesale.
BUNDLE = (
    "openwakeword",       # bundled melspectrogram / embedding ONNX graphs
    "faster_whisper",     # assets + tokenizer data
    "onnxruntime",        # native inference DLLs
    "ctranslate2",        # native Whisper runtime
    "av",                 # audio decoding
    "edge_tts",
    "sounddevice",        # bundles the PortAudio DLL
    "indic_transliteration",
    "pycaw",
    "comtypes",
    "anthropic",
)

datas, binaries, hiddenimports = [], [], []
for package in BUNDLE:
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

# Every skill module is imported by name at runtime, so static analysis can't
# see them.
hiddenimports += [
    f"jarvis.skills.{name}"
    for name in (
        "apps", "web", "system", "windows_mgr", "media", "volume",
        "display", "device", "files", "productivity", "messaging", "knowledge",
    )
]
hiddenimports += [
    "win32com.client", "pythoncom", "pywintypes", "win32gui", "win32process",
    "win32api", "win32con", "wmi", "uvicorn.logging", "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto", "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]

datas += [("config.example.yaml", "."), (".env.example", ".")]

a = Analysis(
    ["run_engine.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pandas", "scipy.spatial.cKDTree", "PyQt5", "PySide6"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="jarvis-engine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX corrupts some ONNX runtime DLLs
    console=True,       # the desktop app captures stdout for its log view
    disable_windowed_traceback=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="jarvis-engine",
)
