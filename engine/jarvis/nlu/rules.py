"""Offline intent matching.

Two passes, cheapest first:

  1. Regex rules over normalised text. Hindi and English are handled by the
     same rules because `normalize` has already collapsed them onto one
     spelling, and because each intent declares both word orders — English puts
     the verb first ("open chrome"), Hindi puts it last ("chrome kholo").
  2. Fuzzy matching against the `examples` every skill declares, which catches
     phrasings nobody wrote a regex for.

Anything that scores below the configured threshold is handed to the LLM. The
point of this module is that "chrome kholo" never needs a network round-trip.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .normalize import normalize, parse_number, parse_percentage

log = logging.getLogger(__name__)


@dataclass
class Intent:
    skill: str
    args: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    matched_by: str = "rule"
    text: str = ""

    def __str__(self) -> str:
        return f"{self.skill}({self.args}) [{self.matched_by} {self.confidence:.0f}]"


# --- verb vocabulary -------------------------------------------------------
#
# One entry per canonical action, listing every way it gets said in Hindi,
# English and the mix. This table is where new phrasings get added; the rules
# below rarely need to change.

VERBS: dict[str, list[str]] = {
    "open": [
        "open", "launch", "start", "run", "bring up", "fire up", "load",
        "kholo", "khol do", "khol de", "khol", "kholna", "kholiye",
        "chalu karo", "chalu kar do", "chalu", "chala do", "chalao",
        "shuru karo", "shuru kar do", "shuru", "start karo", "on karo",
        "la do", "lao", "nikalo", "dikhao", "dikha do",
    ],
    "close": [
        "close", "quit", "exit", "kill", "terminate", "shut",
        "band karo", "band kar do", "band kardo", "band kar", "band",
        "bandh karo", "bandh kar do", "bandh", "bnd karo",
        "rok do", "roko", "hatao", "hata do", "off karo", "off kar do",
    ],
    "increase": [
        "increase", "raise", "turn up", "up", "louder", "more", "boost",
        "badhao", "badha do", "badhado", "badha", "badhaiye", "bada karo",
        "tez karo", "tez kar do", "tez", "zyada karo", "zyada kar do",
        "zyada", "upar karo", "aur karo",
    ],
    "decrease": [
        "decrease", "lower", "reduce", "turn down", "down", "quieter", "less",
        "kam karo", "kam kar do", "kam kardo", "kam", "ghatao", "ghata do",
        "ghata", "dhima karo", "dheere karo", "chhota karo", "neeche karo",
    ],
    "set": [
        "set", "make it", "change to", "put",
        "kar do", "kardo", "karo", "set karo", "set kar do", "pe karo",
        "par karo", "pe laga do", "kar de", "rakh do",
    ],
    "mute": [
        "mute", "silence", "silent", "shut up",
        "chup karo", "chup", "awaz band", "avaz band", "sound band",
        "mute karo", "mute kar do", "gungi karo",
    ],
    "unmute": ["unmute", "unmute karo", "awaz chalu", "avaz chalu", "sound on"],
    "play": [
        "play", "resume", "chalao", "chala do", "bajao", "baja do", "bajaao",
        "play karo", "play kar do", "shuru karo",
    ],
    "pause": [
        "pause", "hold", "ruko", "rok do", "roko", "thamo", "pause karo",
        "pause kar do", "thoda ruko",
    ],
    "next": ["next", "skip", "forward", "agla", "aage", "agla gana", "next karo"],
    "previous": [
        "previous", "prev", "back", "pichla", "piche", "peeche", "wapas",
        "pichla gana", "last wala",
    ],
    "stop": ["stop", "band karo", "rok do", "stop karo", "khatam karo"],
    "sleep": [
        "sleep", "suspend", "standby", "sula do", "sula de", "sulao",
        "sone do", "so jao", "sula dena", "sleep karo", "sleep kar do",
        "go to sleep", "goto sleep", "send to sleep", "put to sleep",
    ],
    "lock": ["lock", "lock karo", "lock kar do", "taala lagao", "band kar lo"],
    "shutdown": [
        "shutdown", "shut down", "power off", "turn off",
        "shutdown karo", "shut down karo", "band kar do computer",
    ],
    "restart": [
        "restart", "reboot", "restart karo", "reboot karo", "dobara chalu karo",
        "phir se chalu karo",
    ],
    "tell": [
        "what is", "whats", "what", "tell me", "tell", "how much", "how many",
        "show me", "show",
        "batao", "bata do", "bata", "kitna", "kitni", "kitne", "kya hai",
        "kya", "dikhao", "dikha do",
    ],
}


def V(*names: str) -> str:
    """Alternation of every synonym for these verbs, longest phrase first.

    Longest-first matters: without it "band" would match before "band karo"
    and swallow the rest of the sentence.
    """
    words: list[str] = []
    for name in names:
        words.extend(VERBS[name])
    words.sort(key=len, reverse=True)
    return "|".join(re.escape(w) for w in words)


# Words that sit between a verb and its object in Hindi and carry no meaning.
PARTICLES = r"(?:\s+(?:ko|ka|ke|ki|me|mein|par|pe|se|wala|wali|ye|yeh|is|this|the|a|an))?"
TARGET = r"(?P<target>[\w\s.'-]+?)"

# Guards the verb-last catch-alls ("<thing> kholo") against questions.
# Without it, "what windows are open" parses as a request to launch an app
# called "what windows are", which then fuzzy-matches something unrelated.
NOT_QUESTION = (
    r"(?!(?:what|whats|which|who|why|how|hows|when|where|is|are|do|does|can|"
    r"kya|kaun|kaunsa|kaunse|kitna|kitni|kitne|kab|kahan|kaise)\b)"
)


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# --- rule table ------------------------------------------------------------
#
# (pattern, skill, argument builder). First match wins, so more specific
# patterns must come first.

Builder = Callable[[re.Match[str], str], dict[str, Any]]


class SkipRule(Exception):
    """Raised by a builder when the rule matched but the arguments don't hold up."""


_app = lambda m, t: {"app": m.group("target").strip()}  # noqa: E731
_none = lambda m, t: {}  # noqa: E731


def _level(m: re.Match[str], text: str) -> dict[str, Any]:
    """Extract a 0-100 level, or bail so a later rule can claim the utterance.

    Without the bail, "volume kam karo" would match a set-level pattern and
    silently become "set volume to 50" instead of turning it down.
    """
    value = parse_percentage(text)
    if value is None:
        raise SkipRule("no level in utterance")
    return {"level": value}


def _minutes(m: re.Match[str], text: str) -> dict[str, Any]:
    value = parse_number(text)
    if value is None:
        raise SkipRule("no duration in utterance")
    return {"minutes": value}


# Folder names that are never installed applications.
_FOLDERS = r"downloads?|documents?|desktop|pictures|photos|music|videos?|recycle bin"


RULES: list[tuple[re.Pattern[str], str, Builder]] = [
    # --- system power (before app rules: "computer band karo" is not an app) --
    (_rx(rf"^(?:{V('shutdown')})(?:\s+(?:the\s+)?(?:pc|computer|laptop|system))?$"),
     "shutdown_pc", _none),
    (_rx(rf"^(?:pc|computer|laptop|system)\s+(?:{V('shutdown')})$"), "shutdown_pc", _none),
    (_rx(rf"^(?:{V('restart')})(?:\s+(?:the\s+)?(?:pc|computer|laptop|system))?$"),
     "restart_pc", _none),
    (_rx(rf"^(?:pc|computer|laptop|system)\s+(?:{V('restart')})$"), "restart_pc", _none),
    # "computer band kar do" is a shutdown, not closing an app called computer.
    (_rx(rf"^(?:pc|computer|laptop|system)\s+(?:{V('close')})$"), "shutdown_pc", _none),
    (_rx(rf"^(?:{V('sleep')})(?:\s+(?:the\s+)?(?:pc|computer|laptop|system))?$"),
     "sleep_pc", _none),
    (_rx(rf"^(?:pc|computer|laptop|system){PARTICLES}\s+(?:{V('sleep')})$"), "sleep_pc", _none),
    (_rx(rf"^(?:{V('lock')})(?:\s+(?:the\s+)?(?:pc|computer|laptop|screen|system))?$"),
     "lock_screen", _none),
    (_rx(rf"^(?:pc|computer|laptop|screen){PARTICLES}\s+(?:{V('lock')})$"),
     "lock_screen", _none),
    (_rx(r"^(?:sign out|log out|logout|logoff|log off)(?:\s+\w+)?$"), "sign_out", _none),
    # "screen band karo" means lock, not close an app called Screen.
    (_rx(rf"^(?:screen|display|monitor)\s+(?:{V('close')}|{V('lock')})$"),
     "lock_screen", _none),

    # --- volume ---
    # Up/down/mute come first: they're unambiguous, and a level rule that
    # matched "volume kam karo" would turn a "turn it down" into "set to 50".
    (_rx(rf"^(?:{V('increase')})\s+(?:the\s+)?(?:volume|sound|awaz|avaz)$"),
     "volume_up", _none),
    (_rx(rf"^(?:volume|sound|awaz|avaz)\s+(?:{V('increase')})$"), "volume_up", _none),
    (_rx(rf"^(?:{V('decrease')})\s+(?:the\s+)?(?:volume|sound|awaz|avaz)$"),
     "volume_down", _none),
    (_rx(rf"^(?:volume|sound|awaz|avaz)\s+(?:{V('decrease')})$"), "volume_down", _none),
    (_rx(rf"^(?:{V('mute')})(?:\s+(?:the\s+)?(?:volume|sound|audio|awaz|avaz))?$"),
     "mute_audio", _none),
    (_rx(rf"^(?:volume|sound|audio|awaz|avaz)\s+(?:{V('mute')}|{V('close')})$"),
     "mute_audio", _none),
    (_rx(rf"^(?:{V('unmute')})(?:\s+(?:the\s+)?(?:volume|sound|audio))?$"),
     "unmute_audio", _none),
    # Any "volume" utterance carrying a number is a set-to-level. Whisper
    # renders spoken numerals as digits and drops small words, so
    # "वॉल्यूम पचास कर दो" can arrive as "volume 50 do" — matching on the
    # number rather than the trailing verb survives that.
    (_rx(r"^(?:\w+\s+){0,2}?volume\b.*\b\d{1,3}\b.*$"), "set_volume", _level),
    (_rx(r"^volume\b.*$"), "set_volume", _level),
    (_rx(rf"^(?:{V('set')})\s+(?:the\s+)?volume{PARTICLES}\s+.*$"), "set_volume", _level),
    # --- brightness ---
    (_rx(rf"^(?:{V('increase')})\s+(?:the\s+)?brightness$"), "brightness_up", _none),
    (_rx(rf"^brightness\s+(?:{V('increase')})$"), "brightness_up", _none),
    (_rx(rf"^(?:{V('decrease')})\s+(?:the\s+)?brightness$"), "brightness_down", _none),
    (_rx(rf"^brightness\s+(?:{V('decrease')})$"), "brightness_down", _none),
    (_rx(r"^(?:\w+\s+){0,2}?brightness\b.*\b\d{1,3}\b.*$"), "set_brightness", _level),
    (_rx(rf"^(?:{V('set')})\s+(?:the\s+)?brightness{PARTICLES}\s+.*$"),
     "set_brightness", _level),
    (_rx(r"^(?:toggle\s+)?night\s*light(?:\s+\w+)?$"), "toggle_night_light", _none),

    # --- media ---
    (_rx(rf"^(?:{V('next')})(?:\s+(?:song|track|gana|video))?$"), "media_next", _none),
    (_rx(rf"^(?:song|track|gana|video)\s+(?:{V('next')})$"), "media_next", _none),
    (_rx(rf"^(?:{V('previous')})(?:\s+(?:song|track|gana|video))?$"),
     "media_previous", _none),
    (_rx(rf"^(?:{V('pause')})(?:\s+(?:the\s+)?(?:song|music|track|gana|video))?$"),
     "media_play_pause", _none),
    (_rx(rf"^(?:{V('play')})\s+(?:the\s+)?(?:song|music|track|gana|video)$"),
     "media_play_pause", _none),
    (_rx(rf"^(?:song|music|gana|video)\s+(?:{V('play')}|{V('pause')})$"),
     "media_play_pause", _none),
    # "gana band karo" is stopping the music, not closing an app called Song.
    # ("gana" normalises to "song", so it would otherwise hit the app rules.)
    (_rx(rf"^(?:song|music|gana|track|video)\s+(?:{V('close')}|{V('stop')})$"),
     "media_stop", _none),

    # --- youtube / web search (before generic app open) ---
    (_rx(rf"^(?:{V('play')}|{V('open')})?\s*(?P<target>.+?)\s+(?:on|pe|par)\s+youtube"
         rf"(?:\s+(?:{V('play')}|{V('open')}))?$"),
     "youtube_search", lambda m, t: {"query": m.group("target").strip()}),
    (_rx(rf"^youtube\s+(?:pe|par|on)\s+(?P<target>.+?)\s*(?:{V('play')}|{V('open')})?$"),
     "youtube_search", lambda m, t: {"query": m.group("target").strip()}),
    (_rx(r"^(?:search|google|dhundo|dhoondo|khojo|search karo)\s+(?:for\s+)?(?P<target>.+)$"),
     "web_search", lambda m, t: {"query": m.group("target").strip()}),
    (_rx(r"^(?P<target>.+?)\s+(?:search karo|google karo|dhundo|dhoondo)$"),
     "web_search", lambda m, t: {"query": m.group("target").strip()}),

    # --- windows ---
    (_rx(rf"^(?:{V('close')})\s+(?:this|is|yeh|ye)\s*(?:window|windo|tab)?$"),
     "close_window", _none),
    (_rx(rf"^(?:this|is|yeh|ye)\s*(?:window|windo)?\s+(?:{V('close')})$"),
     "close_window", _none),
    (_rx(r"^minimi[sz]e(?:\s+(?:all|everything|sab))$"), "minimize_all", _none),
    (_rx(r"^(?:minimi[sz]e|chhota karo|niche karo)(?:\s+(?:this|window|windo))?$"),
     "minimize_window", _none),
    (_rx(r"^(?:maximi[sz]e|full screen|fullscreen|bada karo|pura karo)"
         r"(?:\s+(?:this|window|windo))?$"), "maximize_window", _none),
    (_rx(r"^(?:show|dikhao)\s+(?:the\s+)?desktop$"), "minimize_all", _none),
    # Excludes system nouns so "go to sleep" isn't read as switching to an app.
    (_rx(rf"^(?:switch|jao|jaao|badlo|go)\s+(?:to\s+)?"
         rf"(?!(?:sleep|bed|desktop|settings)\b){TARGET}$"),
     "switch_to_app", _app),
    # Hindi puts the verb last: "chrome pe jao".
    (_rx(rf"^{NOT_QUESTION}{TARGET}\s+(?:pe|par|pr)?\s*(?:jao|jaao|chalo|switch)$"),
     "switch_to_app", _app),

    # --- screenshot & files ---
    (_rx(r"^(?:take\s+)?(?:a\s+)?screenshot(?:\s+\w+)?$"), "take_screenshot", _none),
    (_rx(r"^screen\s*shot\s+(?:lo|le lo|liya|karo|le)$"), "take_screenshot", _none),
    (_rx(rf"^(?:{V('open')})\s+(?:the\s+)?(?:folder|directory)\s+{TARGET}$"),
     "open_folder", lambda m, t: {"name": m.group("target").strip()}),
    # Well-known folders, before the generic app rules — "open downloads"
    # means a folder, and there is no app by that name to find.
    (_rx(rf"^(?:{V('open')})\s+(?:the\s+|my\s+)?(?P<target>{_FOLDERS})(?:\s+folder)?$"),
     "open_folder", lambda m, t: {"name": m.group("target").strip()}),
    (_rx(rf"^(?P<target>{_FOLDERS})(?:\s+folder)?\s+(?:{V('open')})$"),
     "open_folder", lambda m, t: {"name": m.group("target").strip()}),
    (_rx(r"^(?:find|search|dhundo|dhoondo)\s+(?:file|files|fail)\s+"
         r"(?:called|named|naam|ka naam)?\s*(?P<target>.+)$"),
     "search_files", lambda m, t: {"query": m.group("target").strip()}),

    # --- device status ---
    (_rx(rf"^(?:{V('tell')})?\s*(?:the\s+)?battery(?:\s+\w+)*$"), "get_battery", _none),
    (_rx(r"^battery\s+(?:kitni|kitna|percentage|percent|level|status|bachi|baki).*$"),
     "get_battery", _none),
    (_rx(rf"^(?:{V('tell')})?\s*(?:the\s+)?(?:time|samay|baja|baje)(?:\s+\w+)*$"),
     "get_time", _none),
    (_rx(rf"^(?:{V('tell')})?\s*(?:the\s+)?(?:date|tareekh|tarikh|din)(?:\s+\w+)*$"),
     "get_date", _none),
    (_rx(r"^(?:system|pc|computer|laptop)\s+(?:status|info|information|health|haal).*$"),
     "get_system_status", _none),
    # "open sound settings" is a Settings page, not an app called that.
    (_rx(rf"^(?:{V('open')})\s+(?P<target>display|sound|audio|bluetooth|wifi|network|"
         rf"battery|apps|privacy|update|storage)\s+settings?$"),
     "open_settings", lambda m, t: {"page": m.group("target").strip()}),
    (_rx(rf"^(?:{V('open')})\s+(?:the\s+)?settings?$"), "open_settings", _none),
    (_rx(r"^(?:toggle\s+)?wi\s*-?\s*fi(?:\s+(?:on|off|chalu|band))?$"),
     "toggle_wifi", _none),
    (_rx(r"^(?:toggle\s+)?bluetooth(?:\s+(?:on|off|chalu|band))?$"), "toggle_bluetooth", _none),

    # --- timers & reminders ---
    (_rx(r"^(?:set\s+(?:a\s+)?)?(?:timer|alarm)\s+(?:for\s+)?(?P<target>.+)$"),
     "set_timer", _minutes),
    (_rx(r"^(?P<target>.+?)\s+(?:ka\s+)?timer\s+(?:laga do|lagao|laga|set karo|set)$"),
     "set_timer", _minutes),
    # Read-back forms first: "reminder batao" is a request to list them, which
    # the add-rule below would otherwise store as a reminder named "batao".
    (_rx(r"^(?:my\s+)?(?:reminders?)\s*(?:batao|bata do|padho|sunao|dikhao|list|"
         r"kya hain|kya hai)?$"), "list_reminders", _none),
    (_rx(r"^(?:my\s+)?(?:notes?)\s*(?:batao|bata do|padho|sunao|dikhao|list|"
         r"kya hain|kya hai)?$"), "list_notes", _none),
    (_rx(r"^(?:remind me|remind|yaad dila do|yaad dilana|reminder)\s+(?P<target>.+)$"),
     "add_reminder", lambda m, t: {"text": m.group("target").strip()}),
    (_rx(r"^(?:note|note down|likh lo|likho|note karo)\s+(?:that\s+|ki\s+)?(?P<target>.+)$"),
     "add_note", lambda m, t: {"text": m.group("target").strip()}),

    # --- knowledge ---
    # City-specific forms first — the catch-alls below would swallow the city
    # and silently report the weather wherever the IP says you are.
    (_rx(r"^(?:whats|what is|hows|how is)?\s*(?:the\s+)?(?:mausam|weather)\s+"
         r"(?:in|at|for|of)\s+(?P<target>[\w\s'-]+?)(?:\s+(?:today|now|abhi))?$"),
     "get_weather", lambda m, t: {"city": m.group("target").strip()}),
    (_rx(r"^(?P<target>[\w\s'-]+?)\s+(?:ka|ki|me|mein)\s+(?:mausam|weather)"
         r"(?:\s+\w+)*$"),
     "get_weather", lambda m, t: {"city": m.group("target").strip()}),
    (_rx(r"^(?:whats|what is|hows|how is)\s+the\s+weather(?:\s+.*)?$"), "get_weather", _none),
    (_rx(r"^(?:aj|aaj|today)?\s*(?:ka\s+)?(?:mausam|weather)(?:\s+.*)?$"),
     "get_weather", _none),
    (_rx(r"^(?:news|khabar|khabrein|headlines)$"), "get_news", _none),
    (_rx(r"^(?:translate|anuvad karo)\s+(?P<target>.+?)\s+(?:to|into|in|me|mein)\s+"
         r"(?P<lang>hindi|english|hinglish|urdu|marathi|tamil|telugu|bengali|"
         r"punjabi|gujarati|french|spanish|german|japanese)$"),
     "translate_text",
     lambda m, t: {"text": m.group("target").strip(),
                   "target_language": m.group("lang").strip()}),
    (_rx(r"^(?:translate|anuvad karo|hindi me bolo|english me bolo)\s+(?P<target>.+)$"),
     "translate_text", lambda m, t: {"text": m.group("target").strip()}),
    (_rx(r"^(?:summari[sz]e|summary)\s+(?:the\s+)?(?:clipboard|copied|selection|this)$"),
     "summarize_clipboard", _none),

    # --- apps (last: the catch-all "<verb> <thing>" shapes) ---
    (_rx(rf"^(?:{V('open')})\s+(?:the\s+)?(?:app\s+)?{NOT_QUESTION}{TARGET}$"),
     "open_app", _app),
    (_rx(rf"^{NOT_QUESTION}{TARGET}{PARTICLES}\s+(?:{V('open')})$"), "open_app", _app),
    (_rx(rf"^(?:{V('close')})\s+(?:the\s+)?(?:app\s+)?{NOT_QUESTION}{TARGET}$"),
     "close_app", _app),
    (_rx(rf"^{NOT_QUESTION}{TARGET}{PARTICLES}\s+(?:{V('close')})$"), "close_app", _app),
]


def match_rules(text: str) -> Intent | None:
    """First regex rule that matches, or None."""
    normalised = normalize(text)
    if not normalised:
        return None

    for pattern, skill, builder in RULES:
        m = pattern.match(normalised)
        if not m:
            continue
        try:
            args = builder(m, normalised)
        except SkipRule:
            continue  # matched the shape but not the arguments; try later rules
        except Exception as exc:  # noqa: BLE001 - a bad builder shouldn't kill routing
            log.debug("Rule builder failed for %s: %s", skill, exc)
            continue

        # A captured target that is only filler means the rule over-matched.
        target = args.get("app") or args.get("query") or ""
        if target and len(str(target).strip()) < 2:
            continue

        log.debug("Rule hit: %r -> %s %s", normalised, skill, args)
        return Intent(skill=skill, args=args, confidence=95.0,
                      matched_by="rule", text=normalised)
    return None


# Words too common to count as evidence that two phrases mean the same thing.
_STOPWORDS = frozenset(
    "a an the is are was were be am do does did to for of on in at it this that "
    "my me you your i we and or if what whats how when where which who why can "
    "could would should will shall get got have has had make made let us please "
    "ka ke ki ko se me mein par pe hai hain ho tha thi kya kaise kaun kab kahan "
    "mera meri mujhe aap tum ye yeh wo woh ek do bata batao karo kar de dena".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in text.split() if w not in _STOPWORDS and len(w) > 1}


def match_examples(text: str, min_score: float = 78.0) -> Intent | None:
    """Fuzzy-match against every example phrase declared by every skill.

    Scored with `token_sort_ratio` rather than `WRatio`: WRatio rewards partial
    substring overlap, which made unrelated questions ("why is the sky blue")
    score high against short examples. A shared content word is also required,
    so agreement on filler alone can't carry a match.
    """
    from rapidfuzz import fuzz, process

    from ..skills.registry import registry

    normalised = normalize(text)
    if not normalised:
        return None

    choices: dict[str, str] = {}
    for spec in registry.all():
        for example in spec.examples:
            choices[normalize(example)] = spec.name
    if not choices:
        return None

    spoken_words = _content_words(normalised)
    if not spoken_words:
        return None

    hits = process.extract(normalised, list(choices), scorer=fuzz.token_sort_ratio,
                           score_cutoff=min_score, limit=5)
    for phrase, score, _ in hits:
        if not (spoken_words & _content_words(phrase)):
            continue
        skill = choices[phrase]
        log.debug("Example match: %r ~ %r -> %s (%.0f)", normalised, phrase, skill, score)
        # Deliberately no args: a fuzzy phrase match tells us the intent, not
        # the slot values. Parameterised skills need a rule or the LLM.
        return Intent(skill=skill, args={}, confidence=float(score),
                      matched_by="example", text=normalised)
    return None


def route(text: str, threshold: float = 78.0) -> Intent | None:
    """Best offline interpretation, or None if the LLM should handle it."""
    intent = match_rules(text)
    if intent:
        return intent

    intent = match_examples(text, min_score=threshold)
    if intent and _needs_arguments(intent.skill):
        log.debug("Example match for %s dropped: it needs arguments", intent.skill)
        return None
    return intent


def _needs_arguments(skill_name: str) -> bool:
    from ..skills.registry import registry

    spec = registry.get(skill_name)
    if spec is None:
        return False
    return any(
        p.default is p.empty and name not in ("ctx", "self")
        for name, p in spec.signature.parameters.items()
    )
