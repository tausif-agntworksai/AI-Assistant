# -*- coding: utf-8 -*-
"""Sending mail through Gmail, and falling back when we cannot.

The fallback is the important half. Email worked before this existed by
opening a pre-filled compose window, and it has to keep working that way for
anyone who never authorises, whose token expires, or who is offline — an
unauthorised install must not lose a capability it already had.
"""

import base64

import pytest

from jarvis import gmail_api
from jarvis.skills import messaging as messaging_skill


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """Point the token and client files at a temp directory."""
    monkeypatch.setattr(gmail_api.paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gmail_api.paths, "ensure_dirs", lambda: None)
    return tmp_path


# --- authorisation state ---------------------------------------------------


def test_nothing_is_authorised_on_a_fresh_install(data_dir):
    assert gmail_api.authorised() is False
    assert gmail_api.configured() is False


def test_status_never_leaks_the_token(data_dir):
    """`status` feeds the HUD and --doctor, both of which are shown and shared."""
    secret = "ya29-DO-NOT-SHOW-THIS"
    (data_dir / gmail_api.TOKEN_FILE).write_text(
        '{"token": "%s"}' % secret, encoding="utf-8")
    reported = " ".join(str(v) for v in gmail_api.status().values())
    assert secret not in reported


def test_authorising_without_a_client_explains_what_to_create(data_dir):
    ok, message = gmail_api.authorise(open_browser=False)
    assert ok is False
    assert gmail_api.CLIENT_FILE in message
    assert "console.cloud.google.com" in message


def test_the_scope_is_send_only():
    """An assistant that wants to read your mail in order to send mail is
    asking for the wrong thing, and the consent screen is where the user finds
    out what they agreed to."""
    assert gmail_api.SCOPES == ["https://www.googleapis.com/auth/gmail.send"]
    assert all("readonly" not in s and "modify" not in s for s in gmail_api.SCOPES)


def test_signing_out_removes_the_token(data_dir):
    token = data_dir / gmail_api.TOKEN_FILE
    token.write_text("{}", encoding="utf-8")
    assert gmail_api.sign_out() is True
    assert not token.exists()


def test_an_unreadable_token_does_not_raise(data_dir):
    (data_dir / gmail_api.TOKEN_FILE).write_text("not json", encoding="utf-8")
    assert gmail_api.authorised() is False


# --- sending ---------------------------------------------------------------


def test_sending_without_authorisation_is_refused_not_attempted(data_dir):
    sent, detail = gmail_api.send("sana@example.com", "Hi", "Running late.")
    assert sent is False
    assert detail == "not authorised"


class _FakeGmail:
    """Stands in for the Google client, and records what it was handed."""

    def __init__(self, fail_with=None):
        self.sent = None
        self.fail_with = fail_with

    def users(self):
        return self

    def messages(self):
        return self

    def send(self, userId, body):  # noqa: N803 - the API's own spelling
        self.sent = (userId, body)
        return self

    def execute(self):
        if self.fail_with:
            raise self.fail_with
        return {"id": "18f0abc"}


def _authorise(monkeypatch, service):
    monkeypatch.setattr(gmail_api, "_credentials", lambda refresh=True: object())
    monkeypatch.setattr("googleapiclient.discovery.build",
                        lambda *a, **k: service)


def test_a_sent_message_is_a_real_rfc822_mail(monkeypatch, data_dir):
    service = _FakeGmail()
    _authorise(monkeypatch, service)

    sent, detail = gmail_api.send("sana@example.com", "Late", "I'll be 20 minutes.")

    assert sent is True and "18f0abc" in detail
    user_id, body = service.sent
    assert user_id == "me"
    raw = base64.urlsafe_b64decode(body["raw"]).decode("utf-8")
    assert "To: sana@example.com" in raw
    assert "Subject: Late" in raw
    assert "I'll be 20 minutes." in raw


def test_an_api_failure_is_reported_not_raised(monkeypatch, data_dir):
    _authorise(monkeypatch, _FakeGmail(fail_with=RuntimeError("network down")))
    sent, detail = gmail_api.send("sana@example.com", "Hi", "Hello")
    assert sent is False
    assert "network down" in detail


# --- the subject nobody dictates -------------------------------------------


@pytest.mark.parametrize("body, expected", [
    ("I'll send the report tomorrow.", "I'll send the report tomorrow"),
    ("Running late. Start without me.", "Running late"),
    ("", "(no subject)"),
    ("one two three four five six seven eight nine ten",
     "one two three four five six seven eight…"),
])
def test_a_subject_is_taken_from_the_first_clause(body, expected):
    """Nobody says "subject colon" out loud, and asking turns a one-sentence
    errand into an interview."""
    assert messaging_skill._subject_from(body) == expected


# --- the fallback ----------------------------------------------------------


def test_the_send_path_declines_when_unauthorised(monkeypatch):
    """False here means the compose window opens, exactly as it used to."""
    monkeypatch.setattr(gmail_api, "authorised", lambda: False)
    sent, detail = messaging_skill._send_by_api("sana@example.com", "Hello")
    assert sent is False
    assert detail == "not authorised"


def test_a_failed_api_send_still_falls_back(monkeypatch):
    monkeypatch.setattr(gmail_api, "authorised", lambda: True)
    monkeypatch.setattr(gmail_api, "send", lambda *a, **k: (False, "gmail api: 403"))
    sent, _ = messaging_skill._send_by_api("sana@example.com", "Hello")
    assert sent is False
