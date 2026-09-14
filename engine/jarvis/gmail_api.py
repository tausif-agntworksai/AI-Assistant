"""Sending mail through Gmail, without opening a browser.

Until now "email Sana" opened a pre-filled compose window and stopped there,
which is honest but leaves the last step to the user every single time. With
the Gmail API the message actually goes.

Three decisions worth stating, because each is a place this could have been
built worse:

**The scope is `gmail.send` and nothing else.** It permits creating and
sending mail and grants no ability to read a single message. An assistant that
asks for mailbox access in order to send mail is asking for the wrong thing,
and the consent screen is where the user finds out what they agreed to.

**Authorisation is a thing the user does once, deliberately.** No flow is
started in the middle of a spoken command — being sent to a browser consent
page because you said "email Sana" would be startling, and a voice turn is the
worst possible moment to read a permissions dialog. Until the user has run the
authorisation, every send falls back to the compose window that has always
worked.

**Absence is not failure.** Every import here is lazy and wrapped: an engine
installed without the Google libraries, or a user who never authorised,
composes mail in the browser exactly as before.
"""

from __future__ import annotations

import base64
import logging
import threading
from email.message import EmailMessage
from pathlib import Path

from . import paths

log = logging.getLogger(__name__)

#: Send only. Deliberately not `gmail.compose` (which can also read drafts)
#: and emphatically not `gmail.modify` or full mailbox access.
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

#: Where the user drops the OAuth client they created in Google Cloud. Not
#: shipped with the app: a desktop client secret embedded in a distributed
#: binary is a secret in name only, and this way each install is the user's
#: own project with its own quota.
CLIENT_FILE = "gmail_client_secret.json"

#: The refresh token, written after a successful consent.
TOKEN_FILE = "gmail_token.json"

_lock = threading.Lock()


def client_path() -> Path:
    return paths.DATA_DIR / CLIENT_FILE


def token_path() -> Path:
    return paths.DATA_DIR / TOKEN_FILE


def available() -> bool:
    """Whether the Google libraries are installed at all."""
    try:
        import google_auth_oauthlib.flow  # noqa: F401
        import googleapiclient.discovery  # noqa: F401
    except ImportError:
        return False
    return True


def configured() -> bool:
    """Whether the user has supplied an OAuth client to authorise against."""
    return client_path().exists()


def authorised() -> bool:
    """Whether a usable token exists. Never starts a consent flow."""
    return token_path().exists() and _credentials() is not None


def _credentials(refresh: bool = True):
    """Load stored credentials, refreshing them if that is all they need.

    Returns None rather than raising: every caller's fallback is the compose
    window, and a broken token should cost a slower path, not a failed turn.
    """
    if not available() or not token_path().exists():
        return None

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    try:
        creds = Credentials.from_authorized_user_file(str(token_path()), SCOPES)
    except (OSError, ValueError) as exc:
        log.warning("Gmail token unreadable (%s) — falling back to the browser", exc)
        return None

    if creds.valid:
        return creds
    if refresh and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as exc:  # noqa: BLE001 - any refresh failure is the same to us
            log.warning("Gmail token could not be refreshed (%s)", exc)
            return None
        _save(creds)
        return creds
    return None


def _save(creds) -> None:
    try:
        paths.ensure_dirs()
        path = token_path()
        path.write_text(creds.to_json(), encoding="utf-8")
        # Best effort on Windows, meaningful on everything else.
        try:
            path.chmod(0o600)
        except OSError:
            pass
    except OSError as exc:
        log.error("Could not store the Gmail token (%s) — you will be asked again", exc)


def authorise(open_browser: bool = True) -> tuple[bool, str]:
    """Run the one-time consent flow. Returns (ok, message for the user).

    Called from the command line, never from a spoken turn.
    """
    if not available():
        return False, ("The Google libraries aren't installed. Run: "
                       "pip install google-api-python-client google-auth-oauthlib")
    if not configured():
        return False, (
            f"No OAuth client found at {client_path()}.\n"
            "  Create one at https://console.cloud.google.com/apis/credentials\n"
            "  (Desktop app), enable the Gmail API, download the JSON, and save\n"
            f"  it to that path."
        )

    from google_auth_oauthlib.flow import InstalledAppFlow

    try:
        flow = InstalledAppFlow.from_client_secrets_file(str(client_path()), SCOPES)
        creds = flow.run_local_server(port=0, open_browser=open_browser,
                                      prompt="consent")
    except Exception as exc:  # noqa: BLE001 - consent can fail a dozen ways
        return False, f"Authorisation did not complete: {exc}"

    _save(creds)
    return True, f"Gmail authorised. The token is at {token_path()}."


def sign_out() -> bool:
    """Forget the token. The next send falls back to the compose window."""
    try:
        token_path().unlink(missing_ok=True)
        return True
    except OSError as exc:
        log.error("Could not remove the Gmail token: %s", exc)
        return False


def send(to: str, subject: str, body: str, sender: str = "me") -> tuple[bool, str]:
    """Send one plain-text message. Returns (sent, detail).

    `detail` is for the log and the audit trail, not to be spoken: it carries
    the message id on success and the API's own complaint on failure.
    """
    creds = _credentials()
    if creds is None:
        return False, "not authorised"

    message = EmailMessage()
    message["To"] = to
    message["Subject"] = subject or ""
    message.set_content(body or "")

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")

    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    # Serialised because the Google client is not thread-safe, and a spoken
    # turn and a queued retry can otherwise arrive together.
    with _lock:
        try:
            # cache_discovery=False: the default file cache warns noisily on
            # every call under a frozen build, and we build one service per
            # send anyway.
            service = build("gmail", "v1", credentials=creds, cache_discovery=False)
            sent = service.users().messages().send(
                userId=sender, body={"raw": raw},
            ).execute()
        except HttpError as exc:
            log.error("Gmail API refused the message: %s", exc)
            return False, f"gmail api: {getattr(exc, 'status_code', '')} {exc.reason}".strip()
        except Exception as exc:  # noqa: BLE001 - network, auth, anything
            log.error("Gmail send failed: %s", exc)
            return False, f"{type(exc).__name__}: {exc}"

    message_id = str(sent.get("id", ""))
    log.info("Gmail message sent (id %s)", message_id or "unknown")
    return True, f"gmail id={message_id}"


def status() -> dict:
    """What the HUD and `--doctor` show. Never includes the token itself."""
    return {
        "libraries": available(),
        "client_configured": configured(),
        "authorised": authorised(),
        "client_path": str(client_path()),
    }
