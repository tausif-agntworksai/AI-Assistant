"""Websites, searches and YouTube."""

from __future__ import annotations

import urllib.parse

from .. import winutil
from ..app_index import WEB_APPS
from ..permissions import Risk
from .registry import fail, ok, skill


def _normalize_url(target: str) -> str:
    target = target.strip()
    if target.startswith(("http://", "https://")):
        return target
    known = WEB_APPS.get(target.lower())
    if known:
        return known
    if "." in target and " " not in target:
        return f"https://{target}"
    return ""


@skill(
    name="open_website",
    description="Open a website in the default browser",
    risk=Risk.SAFE,
    category="web",
    params={"site": "Website name or URL, e.g. youtube, github.com"},
    examples=[
        "open youtube", "youtube kholo", "go to github.com", "open gmail",
        "instagram kholo", "open netflix",
    ],
)
def open_website(site: str) -> object:
    url = _normalize_url(site or "")
    if not url:
        return web_search(site)  # not a URL — treat it as something to look up
    if not winutil.shell_open(url):
        return fail(f"I couldn't open {site}.", f"{site} nahi khul paya.")
    return ok(f"Opening {site}.", f"{site} khol raha hoon.", url=url)


@skill(
    name="web_search",
    description="Search the web for something and open the results",
    risk=Risk.SAFE,
    category="web",
    params={"query": "What to search for"},
    examples=[
        "search for python tutorials", "google the weather in delhi",
        "python tutorial dhundo", "search karo best laptops",
    ],
)
def web_search(query: str) -> object:
    query = (query or "").strip()
    if not query:
        return fail("What should I search for?", "Kya search karun?")
    url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(query)
    if not winutil.shell_open(url):
        return fail("I couldn't open the browser.", "Browser nahi khul paya.")
    return ok(f"Searching for {query}.", f"{query} search kar raha hoon.", url=url)


@skill(
    name="youtube_search",
    description="Search YouTube and open the results, or play a song or video",
    risk=Risk.SAFE,
    category="web",
    params={"query": "Song, video or channel to look for"},
    examples=[
        "play lofi beats on youtube", "youtube pe arijit singh chalao",
        "search youtube for python tutorial", "youtube pe gana chalao",
    ],
)
def youtube_search(query: str) -> object:
    query = (query or "").strip()
    if not query:
        return open_website("youtube")
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)
    if not winutil.shell_open(url):
        return fail("I couldn't open YouTube.", "YouTube nahi khul paya.")
    return ok(f"Searching YouTube for {query}.",
              f"YouTube pe {query} dhoondh raha hoon.", url=url)


@skill(
    name="open_url",
    description="Open an exact URL",
    risk=Risk.SAFE,
    category="web",
    params={"url": "The full URL to open"},
    examples=["open https://claude.ai", "go to example.com"],
    hidden=True,
)
def open_url(url: str) -> object:
    url = _normalize_url(url or "")
    if not url:
        return fail("That doesn't look like a link.", "Ye link jaisa nahi lag raha.")
    winutil.shell_open(url)
    return ok("Opening it now.", "Khol raha hoon.", url=url)
