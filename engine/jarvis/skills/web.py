"""Websites, searches and YouTube."""

from __future__ import annotations

import re
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
    return ok(f"Opening {site}.{_offline_note(url)}",
              f"{site} khol raha hoon.{_offline_note_hi(url)}", url=url)


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
    needs_network=True,
    network_label="Searching the web",
    network_label_hi="Web par search karna",
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
    needs_network=True,
    network_label="YouTube",
    network_label_hi="YouTube",
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
    return ok(f"Opening it now.{_offline_note(url)}",
              f"Khol raha hoon.{_offline_note_hi(url)}", url=url)


# Hosts that need no internet to answer. Opening one of these offline is a
# perfectly ordinary thing to do — a local dev server, the router's admin page,
# a machine on the same network — so the note below must not fire for them.
_LOCAL_HOST = re.compile(
    r"^https?://(localhost|127\.\d+\.\d+\.\d+|\[::1\]|0\.0\.0\.0"
    r"|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+"
    r"|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+"
    r"|[^/]*\.local)(:\d+)?(/|$)",
    re.IGNORECASE,
)


def _is_local(url: str) -> bool:
    return bool(_LOCAL_HOST.match(url or ""))


def _offline_note(url: str) -> str:
    """Warn that the page will be blank, rather than refusing to open it.

    The browser does show its own error page, but the user asked the assistant
    and it is the assistant that should explain — otherwise a dead tab looks
    like the command failed.
    """
    from .. import net

    if _is_local(url) or net.is_online():
        return ""
    return " There's no internet right now, so it may not load."


def _offline_note_hi(url: str) -> str:
    from .. import net

    if _is_local(url) or net.is_online():
        return ""
    return " Abhi internet nahi hai, to page khali aa sakta hai."
