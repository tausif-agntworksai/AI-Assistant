"""Text normalisation — the layer that makes bilingual matching tractable.

Whisper returns Hindi as Devanagari when it's confident the utterance is Hindi,
but as romanised text when the sentence is mixed ("chrome kholo"). Both must
reach the rule matcher in one form, so everything is transliterated to Roman
and then canonicalised.

Transliteration is deliberately lossy. "खोलो" only has to land close enough to
"kholo" for a fuzzy match to fire — phonetic exactness would buy nothing.
"""

from __future__ import annotations

import functools
import logging
import re
import unicodedata

log = logging.getLogger(__name__)

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_PUNCT = re.compile(r"[^\w\s%°+-]", re.UNICODE)
_APOSTROPHES = re.compile(r"['‘’ʼ`]")
_WS = re.compile(r"\s+")

# Devanagari digits share meaning with ASCII ones.
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI.search(text))


@functools.lru_cache(maxsize=2048)
def transliterate(text: str) -> str:
    """Devanagari to lowercase Roman, tuned for Hinglish command matching."""
    if not has_devanagari(text):
        return text

    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate as _tr

        romanised = _tr(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    except Exception as exc:  # noqa: BLE001
        log.debug("Transliteration unavailable (%s); passing text through", exc)
        return text

    return _clean_itrans(romanised)


# Vowel signs ITRANS leaves untransliterated, plus their inherent-'a' carrier.
_LEFTOVER_SIGNS = {
    "aॉ": "o",  # ॉ  candra O, as in लैपटॉप / वॉल्यूम
    "aॅ": "e",  # ॅ  candra E
    "ॉ": "o",
    "ॅ": "e",
    "़": "",    # ़  nukta
}

# Nasalisation markers -> plain 'n'.
_NASALS = {".N": "n", ".n": "n", "~N": "n", "~n": "n", "M": "n", "H": ""}


def _clean_itrans(text: str) -> str:
    """Turn ITRANS output into the way people actually spell Hinglish.

    Order matters here. ITRANS is case-sensitive: lowercase 'a' is the inherent
    schwa, uppercase 'A' is a long aa. Hindi deletes the word-final schwa
    ("baMda" is said "band") but never the long aa ("gAnA" is "gaana"), so
    schwa deletion has to happen *before* case folding — otherwise both look
    identical and "gaana" wrongly becomes "gaan".
    """
    out = text
    for sign, replacement in _LEFTOVER_SIGNS.items():
        out = out.replace(sign, replacement)
    for marker, replacement in _NASALS.items():
        out = out.replace(marker, replacement)
    out = out.replace("RRi", "ri").replace("R^i", "ri")

    # ITRANS modifiers ('.', '^', '\', '_') are structural, not phonetic.
    out = re.sub(r"[.^\\_|]", "", out)
    out = re.sub(r"[^\w\s]", " ", out)

    words = []
    for word in out.split():
        if len(word) >= 3 and word.endswith("a") and word[-2] not in "aAeEiIoOuU":
            word = word[:-1]
        words.append(word)
    out = " ".join(words)

    # Only now fold case and collapse long vowels to their short spellings.
    out = out.replace("A", "a").replace("I", "i").replace("U", "u")
    out = out.replace("T", "t").replace("D", "d").replace("N", "n").replace("S", "s")
    out = out.lower()
    out = out.replace("chh", "ch").replace("aa", "a").replace("ii", "i").replace("uu", "u")
    return _WS.sub(" ", out).strip()


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


# Loanwords written in Devanagari come back phonetically spelled ("क्रोम" ->
# "krom"). Mapping them to their English form lets one rule cover both
# languages instead of every rule needing a Hindi twin.
_SPELLING_FIXES = {
    "krom": "chrome", "kroam": "chrome",
    "yutyub": "youtube", "yutub": "youtube",
    "vhatsaepp": "whatsapp", "vhatsapp": "whatsapp", "vatsaepp": "whatsapp",
    "volyum": "volume", "valyum": "volume", "vollyum": "volume",
    "skrinashot": "screenshot", "skrinshot": "screenshot", "skrin": "screen",
    "kanpyutar": "computer", "kampyutar": "computer",
    "laipatop": "laptop", "laiptop": "laptop",
    "taimar": "timer", "timar": "timer",
    "braitanes": "brightness", "braitnes": "brightness",
    "myuzik": "music", "gan": "song", "gana": "song",
    "spikar": "speaker", "fail": "file", "folder": "folder",
    "brauzar": "browser", "vindo": "window", "vindoz": "windows",
    "nots": "notes", "not": "note", "rimaindar": "reminder",
    "sistam": "system", "baitari": "battery", "minat": "minute",
    "aelarm": "alarm", "alarm": "alarm", "meil": "email", "imel": "email",
    "instagram": "instagram", "netflix": "netflix",

    # Mishearings observed from the fast Whisper model on this microphone.
    # Every entry here is one command the user would otherwise have had to
    # repeat, so this table is worth growing whenever a miss is noticed.
    "crome": "chrome", "chrom": "chrome", "krome": "chrome", "chroma": "chrome",
    "youtub": "youtube", "utube": "youtube", "yutube": "youtube",
    "watsapp": "whatsapp", "whatsap": "whatsapp", "watsap": "whatsapp",
    "spotifai": "spotify", "spotifi": "spotify",
    "settingz": "settings", "seting": "settings", "setings": "settings",
    "skreenshot": "screenshot", "screenshort": "screenshot",
    "wolume": "volume", "volum": "volume", "walyum": "volume",
    "brightnes": "brightness", "brighness": "brightness",
    "kolo": "kholo", "kolho": "kholo", "cholo": "kholo",
    # Every Whisper size mangles "sula do" (put to sleep) the same handful of
    # ways, and none of them are English words, so mapping them is safe. This
    # was the one command all four model sizes failed in the benchmark — a
    # bigger model did not fix it and this line does.
    "solado": "sulao", "sulado": "sulao", "soulado": "sulao",
    "sulade": "sulao", "sulao": "sulao", "sulaado": "sulao",
    "caro": "karo", "kro": "karo", "karro": "karo",
    "tamar": "timer", "taymar": "timer", "minakt": "minute", "minit": "minute",
    "paj": "panch", "panch": "panch", "pach": "panch",
    "bandh": "band", "bund": "band",
    "karado": "kardo", "kardo": "kardo",
    "batao": "batao", "bathao": "batao", "batado": "batao",
}

# Fixes that span a word boundary, applied before the per-word table. Whisper
# splits compound product names as often as it joins them.
_PHRASE_FIXES = (
    (re.compile(r"\byou\s+tube\b"), "youtube"),
    (re.compile(r"\bwhats\s+app\b"), "whatsapp"),
    (re.compile(r"\bv\s*s\s+code\b"), "vscode"),
    (re.compile(r"\bvisual\s+studio\s+code\b"), "vscode"),
    (re.compile(r"\bfile\s+explorer\b"), "explorer"),
    (re.compile(r"\bcontrol\s+panel\b"), "controlpanel"),
    (re.compile(r"\bscreen\s+shot\b"), "screenshot"),
    (re.compile(r"\bnight\s+lite\b"), "night light"),
    (re.compile(r"\bwi\s*fi\b"), "wifi"),
    (re.compile(r"\bblue\s+tooth\b"), "bluetooth"),
    (re.compile(r"\bshut\s+down\b"), "shutdown"),
    (re.compile(r"\bre\s+start\b"), "restart"),
    (re.compile(r"\bkhol\s+do\b"), "kholo"),
    (re.compile(r"\bband\s+kar\s+do\b"), "band karo"),
)


def normalize(text: str) -> str:
    """Full pipeline: transliterate, lowercase, strip punctuation and filler."""
    if not text:
        return ""
    text = text.translate(_DEV_DIGITS)
    text = transliterate(text)
    text = strip_accents(text).lower()
    # Contractions must close up, not split: the generic punctuation pass turns
    # "what's" into "what s", which then matches nothing.
    text = _APOSTROPHES.sub("", text)
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    for pattern, replacement in _PHRASE_FIXES:
        text = pattern.sub(replacement, text)
    text = " ".join(_SPELLING_FIXES.get(w, w) for w in text.split())
    return _strip_filler(text)


# Politeness terms that carry no intent wherever they appear.
_FILLER = (
    "please", "plz", "kindly", "umm", "uh", "arey", "toh",
    "zara sa", "zara", "thoda", "yaar", "bhai", "sir", "madam",
    "can you", "could you", "would you", "i want you to", "i want to",
)

# Greetings and the assistant's own name. These are only filler at the *start*
# of an utterance — stripped everywhere, "translate hello to hindi" loses the
# very word being translated.
_LEADING_FILLER = (
    "hey", "hi", "hello", "ok", "okay", "yo", "acha", "achha",
    "jarvis", "javis", "jaarvis", "jervis", "jarwis",
)


def _alternation(words: tuple[str, ...]) -> str:
    return "|".join(sorted((re.escape(w) for w in words), key=len, reverse=True))


_FILLER_RE = re.compile(rf"\b({_alternation(_FILLER)})\b")
_LEADING_RE = re.compile(rf"^(?:(?:{_alternation(_LEADING_FILLER)})\b[\s,]*)+")


def _strip_filler(text: str) -> str:
    cleaned = _LEADING_RE.sub("", text)
    cleaned = _WS.sub(" ", _FILLER_RE.sub(" ", cleaned)).strip()
    # Never return nothing: "jarvis" alone is a valid (if empty) utterance.
    return cleaned or text


# --- numbers ---------------------------------------------------------------

_ENGLISH_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_ENGLISH_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

# Romanised Hindi numerals, including the spellings Whisper tends to produce.
_HINDI_NUMBERS = {
    "shunya": 0, "sifar": 0,
    "ek": 1, "do": 2, "teen": 3, "tin": 3, "char": 4, "chaar": 4,
    "panch": 5, "paanch": 5, "chhe": 6, "che": 6, "chah": 6,
    "sat": 7, "saat": 7, "ath": 8, "aath": 8, "nau": 9, "no": 9,
    "das": 10, "dus": 10, "gyarah": 11, "barah": 12, "terah": 13, "chaudah": 14,
    "pandrah": 15, "solah": 16, "satrah": 17, "atharah": 18, "unnis": 19,
    "bees": 20, "bis": 20, "pachchis": 25, "pachees": 25, "pachchees": 25,
    "tees": 30, "tis": 30, "chalis": 40, "chalees": 40,
    "pachas": 50, "pachaas": 50, "pachchas": 50,
    "saath": 60, "sath": 60, "sattar": 70, "assi": 80, "asi": 80,
    "nabbe": 90, "navve": 90, "sau": 100, "so": 100,
}

_MULTIPLIERS = {"hundred": 100, "sau": 100, "thousand": 1000, "hazar": 1000, "hazaar": 1000}

# "do" is Hindi for 2 but also an English verb; "no" is 9 in Hindi but usually
# a refusal. Only read them as numbers next to a unit word.
_CONTEXT_ONLY = {"do", "no", "so", "sat", "che", "ath"}


def parse_number(text: str, default: int | None = None) -> int | None:
    """Extract the first number, written in digits, English, or Hindi."""
    if not text:
        return default

    text = normalize(text)

    digits = re.search(r"\b(\d{1,4})\b", text)
    if digits:
        return int(digits.group(1))

    words = text.split()
    total = 0
    current = 0
    found = False

    for i, word in enumerate(words):
        if word in _MULTIPLIERS:
            current = (current or 1) * _MULTIPLIERS[word]
            total += current
            current = 0
            found = True
            continue

        value = _ENGLISH_UNITS.get(word)
        if value is None:
            value = _ENGLISH_TENS.get(word)
        if value is None and word in _HINDI_NUMBERS:
            if word in _CONTEXT_ONLY and not _has_numeric_neighbour(words, i):
                continue
            value = _HINDI_NUMBERS[word]

        if value is None:
            if found:
                break  # numbers are contiguous; stop at the first non-number
            continue

        current += value
        found = True

    if not found:
        return default
    return total + current


def _has_numeric_neighbour(words: list[str], index: int) -> bool:
    """True if an adjacent word is a unit that makes a numeric reading likely."""
    neighbours = words[max(0, index - 1) : index] + words[index + 1 : index + 2]
    units = {
        "minute", "minutes", "min", "mins", "second", "seconds", "sec",
        "hour", "hours", "ghante", "ghanta", "minat", "minute", "baje",
        "percent", "%", "ka", "ke", "baar", "bar",
    }
    return any(n in units for n in neighbours)


# Romanised Hindi words distinctive enough that seeing one is strong evidence
# the utterance is Hindi. Every entry has to be checked against English:
# Hindi "the" (were) and "band" (closed) are both ordinary English words and
# were flagging plain English commands as Hindi, so neither is listed here —
# the verbs that accompany them ("karo") carry the signal instead.
_HINDI_MARKERS = frozenset("""
kholo khol kholna chalu chalao chala bajao bandh karo kar karna kardo
batao bata dikhao dikha suno sunao likho likh dhundo dhoondo lagao laga
badhao badha ghatao ghata zyada tez dhima sula sulao
kitna kitni kitne kaise kaisa kaun kahan kyun kyu kab
mujhe mera meri tumhara aapka hume hamara
hai hain tha thi hoga hogi raha rahi rahe
nahi haan bilkul theek accha achha zara thoda phir abhi
mausam tareekh khabar khabrein baje ghante
aur bhi kuch sab yahan wahan idhar udhar
""".split())

# Whisper often glues Hindi words together ("sula do" -> "sulado"), which
# exact matching misses. These stems are distinctive enough to match a prefix.
_HINDI_STEMS = (
    "khol", "sula", "chalu", "badha", "ghata", "dikha", "batao", "sunao",
    "lagao", "lagad", "kardo", "karado", "bandh",
)


def looks_hindi(text: str) -> bool:
    """True when an utterance is Hindi or Hinglish.

    Whisper's own language field is unreliable on short romanised commands —
    it labelled "Chrome kholo" as Polish in testing — so this backs it up.
    Getting it wrong means answering a Hindi question in English.
    """
    if has_devanagari(text):
        return True
    words = normalize(text).split()
    if set(words) & _HINDI_MARKERS:
        return True
    return any(len(w) >= 5 and w.startswith(_HINDI_STEMS) for w in words)


# Words that only appear in English sentences.
#
# Two deliberate exclusions. Anything that doubles as a romanised Hindi word is
# out — "the" is Hindi for "were", "band" is "closed", "do" is "two", "so" is
# "hundred", "me" is "in". So is every technology loanword: "battery",
# "volume", "screen", "file" and "window" get used constantly *inside* Hindi
# sentences ("battery kitni bachi hai"), so their presence says nothing about
# which language a sentence is in. Listing them made a bare "volume" read as
# English in the middle of a Hindi conversation.
_ENGLISH_MARKERS = frozenset("""
what which who whose why how when where whats hows
is are was were am been being have has had does did
can could would should will shall must might may
please tell show open close turn set make give find search play stop start
explain describe create write read send remind translate summarise summarize
about after all also always another any because before between both
my your our their his her its this that these those there here
but from into with without over under just only very
""".split())

# Whisper regularly labels short romanised Hindi as one of these. They all mean
# the same thing for our purposes: the person was not speaking English.
_HINDI_ADJACENT = frozenset({"hi", "ur", "mr", "ne", "sa", "bn", "pa", "gu"})

# Utterances of at most this many words are treated as too short to judge on
# their own, so they inherit the language of the conversation. Two, because
# that covers the real follow-ups — "aur?", "spotify", "band karo" — while a
# three-word sentence has room for Hindi evidence and its absence means
# something.
_AMBIGUOUS_MAX_WORDS = 2


def detect_language(
    text: str,
    whisper_language: str | None = None,
    whisper_probability: float = 0.0,
    previous: str = "en",
) -> str:
    """Decide which language to answer in. Returns "hi" or "en".

    Whisper's label is one input, not the answer. On a two-word romanised
    command it is close to a coin flip — the model has almost no acoustic
    context to work with and no script to read — so the text itself is
    consulted first and the label is only trusted when it is both confident
    and about a language we care about.

    The order below is the whole policy:

      1. Devanagari on screen — settled, nothing else can outweigh it.
      2. Distinctive romanised Hindi ("kholo", "kitna", "karo") — Hinglish,
         which is what most commands actually are.
      3. A confident Whisper label of Hindi, or of a language people's Hindi
         gets mistaken for (Urdu, Marathi, Nepali…).
      4. Distinctive English words, with no Hindi markers present.
      5. A confident Whisper label of English.
      6. A whole sentence with no Hindi evidence anywhere in it — English.
      7. Only then: whatever language the conversation was already in.

    Step 6 is what stops step 7 overreaching. Inheriting the previous turn is
    right for "aur?" or "spotify", which are too short to judge on their own —
    but it was also catching "explain quantum computing to me", which is
    plainly English and merely happened to follow a Hindi command. A sentence
    long enough to carry Hindi evidence and carrying none is English; anything
    shorter is genuinely ambiguous and inherits.
    """
    label = (whisper_language or "").lower()[:2]
    previous = "hi" if (previous or "en").lower().startswith("hi") else "en"

    if has_devanagari(text):
        return "hi"

    normalised = normalize(text)
    words = set(normalised.split())
    hindi_words = bool(words & _HINDI_MARKERS) or any(
        len(w) >= 5 and w.startswith(_HINDI_STEMS) for w in words
    )
    if hindi_words:
        return "hi"

    confident = whisper_probability >= 0.6
    if confident and label in _HINDI_ADJACENT:
        return "hi"

    if words & _ENGLISH_MARKERS:
        return "en"

    if confident and label == "en":
        return "en"

    if len(words) >= _AMBIGUOUS_MAX_WORDS + 1:
        return "en"

    return previous


def parse_percentage(text: str) -> int | None:
    """Extract a 0-100 value, tolerating 'percent'/'pratishat' suffixes."""
    value = parse_number(text)
    if value is None:
        return None
    return max(0, min(100, value))
