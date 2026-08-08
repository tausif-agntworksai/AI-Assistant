"""Conversation, questions, translation, weather and news."""

from __future__ import annotations

import logging
import urllib.parse

from ..permissions import Risk
from .registry import SkillContext, fail, ok, skill

log = logging.getLogger(__name__)


@skill(
    name="answer_question",
    description=(
        "Answer a general knowledge question, explain something, or hold a "
        "conversation. Use this only when no other skill fits."
    ),
    risk=Risk.SAFE,
    category="knowledge",
    params={"question": "The user's question, in their own words"},
    examples=[
        "who invented the telephone", "explain quantum computing simply",
        "python kya hota hai", "tell me a joke", "ek joke sunao",
    ],
)
def answer_question(question: str, ctx: SkillContext = None) -> object:
    from ..nlu.llm import brain

    language = ctx.language if ctx else "en"
    if not brain.available:
        return fail(
            "I need an Anthropic API key to answer that. Add one to the .env file.",
            "Iska jawab dene ke liye Anthropic API key chahiye. .env me daal dijiye.",
        )

    answer = brain.ask(question, language)
    if not answer:
        return fail("I couldn't work that one out.", "Ye samajh nahi aaya.")
    return ok(answer, answer)


@skill(
    name="translate_text",
    description="Translate text between Hindi and English (or another named language)",
    risk=Risk.SAFE,
    category="knowledge",
    params={"text": "The text to translate",
            "target_language": "Language to translate into, e.g. hindi, english"},
    examples=[
        "translate good morning to hindi", "how do you say thank you in hindi",
        "iska english matlab batao", "hindi me bolo how are you",
    ],
)
def translate_text(text: str, target_language: str = "", ctx: SkillContext = None) -> object:
    from ..nlu.llm import brain

    content = (text or "").strip()
    if not content:
        return fail("What should I translate?", "Kya translate karun?")
    if not brain.available:
        return fail("Translation needs an Anthropic API key.",
                    "Translate karne ke liye API key chahiye.")

    language = ctx.language if ctx else "en"
    # Default to the opposite of whatever was spoken.
    target = (target_language or "").strip() or ("English" if language.startswith("hi") else "Hindi")

    prompt = (
        f"Translate this into {target}. Reply with only the translation, nothing else. "
        f"If the target is Hindi, use Devanagari script.\n\n{content}"
    )
    answer = brain.ask(prompt, "hi" if target.lower().startswith("hi") else "en")
    if not answer:
        return fail("I couldn't translate that.", "Translate nahi kar paya.")
    return ok(answer, answer, detail=f"-> {target}")


@skill(
    name="summarize_clipboard",
    description="Summarise whatever text is currently on the clipboard",
    risk=Risk.SAFE,
    category="knowledge",
    examples=[
        "summarize the clipboard", "summarise what i copied",
        "clipboard ka summary do", "jo copy kiya hai uska matlab batao",
    ],
)
def summarize_clipboard(ctx: SkillContext = None) -> object:
    import pyperclip

    from ..nlu.llm import brain

    try:
        content = (pyperclip.paste() or "").strip()
    except Exception as exc:  # noqa: BLE001
        log.error("Clipboard read failed: %s", exc)
        return fail("I couldn't read the clipboard.", "Clipboard nahi padh paya.")

    if len(content) < 40:
        return fail("There isn't enough on the clipboard to summarise.",
                    "Clipboard me summarise karne layak kuch nahi hai.")
    if not brain.available:
        return fail("Summarising needs an Anthropic API key.",
                    "Summary ke liye API key chahiye.")

    language = ctx.language if ctx else "en"
    answer = brain.ask(
        f"Summarise the following in two or three spoken sentences:\n\n{content[:6000]}",
        language,
    )
    if not answer:
        return fail("I couldn't summarise that.", "Summary nahi bana paya.")
    return ok(answer, answer, detail=f"{len(content)} chars")


@skill(
    name="get_weather",
    description="Report the current weather for a city",
    risk=Risk.SAFE,
    category="knowledge",
    params={"city": "City name; leave blank for the current location"},
    examples=[
        "what's the weather", "aaj mausam kaisa hai", "weather in delhi",
        "mumbai ka mausam batao", "is it going to rain",
    ],
)
def get_weather(city: str = "", ctx: SkillContext = None) -> object:
    import requests

    place = (city or "").strip()
    url = f"https://wttr.in/{urllib.parse.quote(place)}?format=j1"

    try:
        # wttr.in is free and needs no key; it geolocates by IP when city is blank.
        resp = requests.get(url, timeout=12, headers={"User-Agent": "curl/8"})
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        log.error("Weather lookup failed: %s", exc)
        return fail("I couldn't reach the weather service.",
                    "Mausam ki jaankari nahi mil payi.")

    try:
        current = data["current_condition"][0]
        temp = current["temp_C"]
        feels = current["FeelsLikeC"]
        # wttr.in pads its descriptions ("Partly cloudy "), which reads as a
        # stumble when spoken.
        desc = current["weatherDesc"][0]["value"].strip().lower()
        humidity = current["humidity"]
        area = data.get("nearest_area", [{}])[0]
        name = (area.get("areaName", [{}])[0].get("value") or place or "your area").strip()
    except (KeyError, IndexError, TypeError) as exc:
        log.error("Unexpected weather payload: %s", exc)
        return fail("The weather data came back in a shape I didn't expect.",
                    "Mausam ka data samajh nahi aaya.")

    return ok(
        f"{name} is {temp} degrees, feels like {feels}, {desc}, "
        f"humidity {humidity} percent.",
        f"{name} me {temp} degree hai, mehsoos {feels} degree hota hai, "
        f"{desc}, humidity {humidity} percent.",
        temp_c=temp, description=desc, location=name,
    )


@skill(
    name="get_news",
    description="Read out the current top news headlines",
    risk=Risk.SAFE,
    category="knowledge",
    params={"topic": "Optional topic, e.g. technology, sports, business"},
    examples=[
        "what's in the news", "read me the headlines", "aaj ki khabar batao",
        "technology news", "khabrein sunao",
    ],
)
def get_news(topic: str = "", ctx: SkillContext = None) -> object:
    import xml.etree.ElementTree as ET

    import requests

    subject = (topic or "").strip()
    if subject:
        url = ("https://news.google.com/rss/search?q="
               + urllib.parse.quote_plus(subject) + "&hl=en-IN&gl=IN&ceid=IN:en")
    else:
        url = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"

    try:
        resp = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as exc:  # noqa: BLE001
        log.error("News fetch failed: %s", exc)
        return fail("I couldn't fetch the news.", "Khabrein nahi mil payin.")

    titles: list[str] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if title:
            # Google appends " - Publisher"; the publisher isn't worth speaking.
            titles.append(title.rsplit(" - ", 1)[0])
        if len(titles) >= 5:
            break

    if not titles:
        return fail("No headlines came back.", "Koi headline nahi mili.")

    spoken = ". ".join(f"{i}. {t}" for i, t in enumerate(titles, 1))
    label = f" on {subject}" if subject else ""
    label_hi = f" {subject} ki" if subject else ""
    return ok(
        f"Top headlines{label}: {spoken}.",
        f"Mukhya{label_hi} khabrein: {spoken}.",
        headlines=titles,
    )
