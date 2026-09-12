"""Index of launchable applications.

Whether "chrome kholo" feels like magic or like a toy comes down to this file.
Four sources are merged, because no single one is complete:

  * Start Menu shortcuts  - most desktop programs
  * Get-StartApps         - Store/UWP apps (WhatsApp, Spotify, Settings), which
                            have no .exe path and must be launched by AUMID
  * App Paths registry    - things registered for `Run` but not pinned
  * Web aliases           - "youtube" is a site, not an installed program

Results are fuzzy-matched, so the user never has to say the exact display name.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import paths, winutil

log = logging.getLogger(__name__)

CACHE_TTL_SEC = 7 * 24 * 3600

#: A fuzzy score at or above this is the answer rather than a candidate. An
#: exact name or alias scores 100; the nineties are a near-exact match with a
#: word order or a suffix out of place, which is still not a real question.
CONFIDENT = 92.0

#: How alike two strings must be, as whole words, when neither contains the
#: other. "whats app"/"whatsapp" is 94 and "note pad"/"notepad" is 93; the
#: thing this keeps out is "photoshop"/"photos" at 80.
_WHOLE_WORD_FLOOR = 88.0

# Shortcut names that are never what someone means by "open X".
_NOISE = (
    "uninstall", "readme", "read me", "release notes", "documentation", "help",
    "license", "changelog", "website", "support", "manual", "报告", "debug",
    "safe mode", "repair", "troubleshoot", "command prompt for",
)

# Executables that sit beside a program without being it. These are only
# filtered out of the install-directory scan, where there is no curated
# display name to go on — a Start Menu shortcut called "Update" is already
# caught by _NOISE.
_NOT_THE_APP = (
    "setup", "install", "uninstall", "update", "upgrade", "crash", "report",
    "helper", "service", "daemon", "launcher_", "unins", "vcredist", "dotnet",
    "python", "node", "cleanup", "diagnos", "elevate", "broker", "handler",
)

# Spoken names that don't match the installed display name.
_ALIASES: dict[str, tuple[str, ...]] = {
    "chrome": ("google chrome", "browser"),
    "edge": ("microsoft edge",),
    "vscode": ("visual studio code", "vs code", "code"),
    "explorer": ("file explorer", "files", "my computer", "this pc"),
    "settings": ("windows settings", "control panel", "setting"),
    "cmd": ("command prompt", "terminal"),
    "powershell": ("windows powershell", "ps"),
    "task manager": ("taskmgr", "task manger"),
    "whatsapp": ("whats app", "watsapp", "vhatsapp"),
    "spotify": ("spotifi",),
    "calculator": ("calc", "calculater"),
    "notepad": ("note pad",),
    "camera": ("webcam",),
}

# Web destinations. Many "apps" people name are really websites, and opening
# the site is what they actually want.
WEB_APPS: dict[str, str] = {
    "youtube": "https://www.youtube.com",
    "gmail": "https://mail.google.com",
    "google": "https://www.google.com",
    "maps": "https://www.google.com/maps",
    "google maps": "https://www.google.com/maps",
    "drive": "https://drive.google.com",
    "google drive": "https://drive.google.com",
    "whatsapp web": "https://web.whatsapp.com",
    "instagram": "https://www.instagram.com",
    "facebook": "https://www.facebook.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "linkedin": "https://www.linkedin.com",
    "reddit": "https://www.reddit.com",
    "github": "https://github.com",
    "chatgpt": "https://chat.openai.com",
    "claude": "https://claude.ai",
    "gemini": "https://gemini.google.com",
    "netflix": "https://www.netflix.com",
    "amazon": "https://www.amazon.in",
    "flipkart": "https://www.flipkart.com",
    "translate": "https://translate.google.com",
    "google translate": "https://translate.google.com",
    "calendar": "https://calendar.google.com",
    "stackoverflow": "https://stackoverflow.com",
    "stack overflow": "https://stackoverflow.com",
    "hotstar": "https://www.hotstar.com",
    "prime video": "https://www.primevideo.com",
    "zomato": "https://www.zomato.com",
    "swiggy": "https://www.swiggy.com",
}


def _contains_word(haystack: str, needle: str) -> bool:
    """Is `needle` present in `haystack` as a whole word?

    Plain substring containment is what let "photos" match "photoshop". The
    word boundary is the whole difference between a name that is part of
    another name and a name that merely starts the same way.
    """
    return re.search(rf"\b{re.escape(needle)}\b", haystack) is not None


@dataclass
class AppEntry:
    name: str
    launch: str
    kind: str  # shortcut | uwp | exe | web
    source: str = ""
    aliases: list[str] = field(default_factory=list)

    @property
    def search_terms(self) -> list[str]:
        base = [self.name] + self.aliases
        # A display name like "Google Chrome" should also match on "Chrome".
        words = self.name.split()
        if len(words) > 1:
            base.append(words[-1])
        return [t.lower() for t in base if t]


class AppIndex:
    def __init__(self) -> None:
        self.entries: list[AppEntry] = []
        self.built_at: float = 0.0

    # -- building -----------------------------------------------------------

    def build(self, force: bool = False) -> list[AppEntry]:
        if not force and self._load_cache():
            return self.entries

        t0 = time.perf_counter()
        merged: dict[str, AppEntry] = {}

        for entry in (*self._scan_start_menu(), *self._scan_uwp(),
                      *self._scan_app_paths(), *self._scan_desktop(),
                      *self._scan_install_dirs(), *self._web_entries()):
            key = entry.name.lower()
            existing = merged.get(key)
            # Prefer a real installed app over a web fallback of the same name.
            if existing is None or (existing.kind == "web" and entry.kind != "web"):
                merged[key] = entry

        for key, extra in _ALIASES.items():
            for entry in merged.values():
                if key in entry.name.lower() or any(a in entry.name.lower() for a in extra):
                    entry.aliases = sorted({*entry.aliases, key, *extra})

        self.entries = sorted(merged.values(), key=lambda e: e.name.lower())
        self.built_at = time.time()
        self._save_cache()
        log.info("App index: %d entries in %.2fs", len(self.entries), time.perf_counter() - t0)
        return self.entries

    def _scan_start_menu(self) -> list[AppEntry]:
        # Both variables are Windows-only. `Path(os.environ.get("APPDATA", ""))`
        # is `Path(".")` everywhere else, which resolves the glob below against
        # the current working directory — harmless thanks to the `.exists()`
        # guard, but it means the scan was quietly walking the wrong tree.
        start_menu = "Microsoft/Windows/Start Menu/Programs"
        roots = [
            Path(root) / start_menu
            for root in (os.environ.get("ProgramData"), os.environ.get("APPDATA"))
            if root
        ]
        found: list[AppEntry] = []
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.suffix.lower() not in (".lnk", ".url"):
                    continue
                name = path.stem.strip()
                if not name or any(n in name.lower() for n in _NOISE):
                    continue
                found.append(AppEntry(
                    name=name,
                    launch=str(path),
                    kind="shortcut",
                    source=str(root),
                ))
        return found

    def _scan_desktop(self) -> list[AppEntry]:
        """Shortcuts on the Desktop, including the public one.

        Plenty of installers drop a Desktop icon and nothing in the Start Menu,
        and portable programs that were never installed at all often live here
        and nowhere else. If it is on the desktop the user thinks of it as an
        app on this machine, which is the whole test.
        """
        roots = [
            Path(root) / "Desktop"
            for root in (os.environ.get("USERPROFILE"), os.environ.get("PUBLIC"))
            if root
        ]
        found: list[AppEntry] = []
        for root in roots:
            if not root.exists():
                continue
            for path in root.glob("*"):
                if path.suffix.lower() not in (".lnk", ".url"):
                    continue
                name = path.stem.strip()
                if not name or any(n in name.lower() for n in _NOISE):
                    continue
                found.append(AppEntry(name=name, launch=str(path),
                                      kind="shortcut", source="Desktop"))
        return found

    def _scan_install_dirs(self) -> list[AppEntry]:
        """Executables sitting in the usual install roots.

        The last resort, and deliberately shallow: two levels down from
        Program Files finds `Fooin
oo.exe` without walking a tree that can
        hold hundreds of thousands of files. Anything found here is a fallback
        — the Start Menu and App Paths scans already cover the same programs
        with better names, and merge order keeps theirs.
        """
        roots = [
            Path(root) for root in (
                os.environ.get("ProgramFiles"),
                os.environ.get("ProgramFiles(x86)"),
                os.environ.get("LOCALAPPDATA", "") and
                str(Path(os.environ["LOCALAPPDATA"]) / "Programs"),
            ) if root
        ]
        found: list[AppEntry] = []
        for root in roots:
            if not root.exists():
                continue
            for depth in ("*/*.exe", "*/*/*.exe"):
                for path in root.glob(depth):
                    name = path.stem.strip()
                    lowered = name.lower()
                    if not name or any(n in lowered for n in _NOISE):
                        continue
                    # Installers, updaters and crash handlers ship beside the
                    # program and would otherwise answer to its name.
                    if any(w in lowered for w in _NOT_THE_APP):
                        continue
                    found.append(AppEntry(name=name, launch=str(path),
                                          kind="exe", source="install-dir"))
        return found

    def _scan_uwp(self) -> list[AppEntry]:
        """Store/UWP apps. These have no .exe — only an AppUserModelID."""
        data = winutil.powershell_json("Get-StartApps | Select-Object Name, AppID", timeout=30)
        if isinstance(data, dict):
            data = [data]

        found: list[AppEntry] = []
        for item in data or []:
            name = str(item.get("Name") or "").strip()
            app_id = str(item.get("AppID") or "").strip()
            if not name or not app_id or any(n in name.lower() for n in _NOISE):
                continue
            # Entries with a backslash-free ID that ends in .exe are plain
            # executables already covered by the Start Menu scan.
            if "!" in app_id or not app_id.lower().endswith(".exe"):
                found.append(AppEntry(
                    name=name,
                    launch=f"shell:AppsFolder\\{app_id}",
                    kind="uwp",
                    source="Get-StartApps",
                ))
        return found

    def _scan_app_paths(self) -> list[AppEntry]:
        found: list[AppEntry] = []
        try:
            import winreg
        except ImportError:
            return found

        key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(hive, key_path) as key:
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            sub = winreg.EnumKey(key, i)
                            with winreg.OpenKey(key, sub) as subkey:
                                target = winreg.QueryValueEx(subkey, "")[0]
                        except OSError:
                            continue
                        if not target or not sub.lower().endswith(".exe"):
                            continue
                        found.append(AppEntry(
                            name=Path(sub).stem,
                            launch=str(target).strip('"'),
                            kind="exe",
                            source="App Paths",
                        ))
            except OSError:
                continue
        return found

    @staticmethod
    def _web_entries() -> list[AppEntry]:
        return [
            AppEntry(name=name, launch=url, kind="web", source="web-alias")
            for name, url in WEB_APPS.items()
        ]

    # -- searching ----------------------------------------------------------

    def search(self, query: str, limit: int = 5, min_score: int = 70
               ) -> list[tuple[AppEntry, float]]:
        """Fuzzy-match `query`, best first.

        The ranking mechanism is shared with contact lookup (`jarvis.resolve`);
        what stays here is everything specific to applications — the
        word-boundary scorer and the preference for something actually
        installed over a web fallback of the same name.
        """
        from rapidfuzz import fuzz

        from .resolve import rank

        if not self.entries:
            self.build()

        def adjust(entry: AppEntry, score: float) -> float:
            # A web fallback shouldn't beat the real program the user has
            # installed, and a Store app is usually the one they mean.
            if entry.kind == "web":
                return score - 4
            if entry.kind == "uwp":
                return score + 2
            return score

        candidates = rank(
            query, self.entries,
            terms=lambda e: e.search_terms,
            name=lambda e: e.name,
            score=lambda needle, term: self._score(needle, term, fuzz),
            floor=min_score,
            adjust=adjust,
            limit=limit,
        )
        return [(c.item, c.score) for c in candidates]

    @staticmethod
    def _score(needle: str, term: str, fuzz) -> float:
        """How well a spoken name matches one of an entry's search terms.

        `WRatio` alone scores "photoshop" against "photos" at 92, because it
        rewards a shared prefix as if it were a partial match. On a machine
        without Photoshop that silently opened the Photos app — and a wrong
        launch is worse than admitting the program isn't installed.

        So a partial match only counts when one string contains the other *as
        a whole word*: "chrome" in "google chrome" and "code" in "visual
        studio code" qualify, "photos" inside "photoshop" does not. Failing
        that, the two have to be close as whole words, which still recovers
        the mishearings that matter ("whats app", "note pad", "calculater").
        """
        score = float(fuzz.WRatio(needle, term))
        if _contains_word(term, needle) or _contains_word(needle, term):
            return score
        if fuzz.ratio(needle, term) >= _WHOLE_WORD_FLOOR:
            return score
        # Not the same word. Keep it far enough down that `resolve` won't take
        # it, while leaving it visible to a wider search.
        return min(score, 60.0)

    def resolve(self, query: str, min_score: int = 68) -> AppEntry | None:
        hits = self.search(query, limit=1, min_score=min_score)
        return hits[0][0] if hits else None

    def ambiguous(self, query: str, margin: float = 6.0,
                  confident_at: float = CONFIDENT) -> list[AppEntry]:
        """Candidates too close to call — the caller should ask which one.

        A strong top match ends the question. "open chrome" matches Google
        Chrome's alias exactly; that some other Chrome-ish shortcut also scores
        in the nineties is not a reason to stop and ask, and asking anyway was
        turning "open vs code" into "did you mean Visual Studio Code, Visual
        Studio Code Insiders?" on machines that had both. Ambiguity is only
        worth raising when the best guess is itself a guess.
        """
        from .resolve import Candidate, decide

        hits = self.search(query, limit=4)
        outcome = decide(
            [Candidate(item=e, name=e.name, score=s) for e, s in hits],
            strong=confident_at, tie_window=margin,
        )
        if not outcome.ambiguous:
            return []
        return [outcome.best.item, *(c.item for c in outcome.considered[1:]
                                     if c.name in outcome.rivals)]

    # -- cache --------------------------------------------------------------

    def _load_cache(self) -> bool:
        try:
            if not paths.APP_INDEX_CACHE.exists():
                return False
            payload = json.loads(paths.APP_INDEX_CACHE.read_text(encoding="utf-8"))
            if time.time() - payload.get("built_at", 0) > CACHE_TTL_SEC:
                return False
            self.entries = [AppEntry(**e) for e in payload.get("entries", [])]
            self.built_at = payload["built_at"]
            log.debug("Loaded %d apps from cache", len(self.entries))
            return bool(self.entries)
        except Exception as exc:  # noqa: BLE001
            log.debug("App index cache unusable: %s", exc)
            return False

    def _save_cache(self) -> None:
        try:
            paths.ensure_dirs()
            paths.APP_INDEX_CACHE.write_text(
                json.dumps(
                    {"built_at": self.built_at, "entries": [asdict(e) for e in self.entries]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            log.debug("Could not write app index cache: %s", exc)


app_index = AppIndex()
