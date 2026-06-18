"""
Email Parser — converts raw RFC822 bytes into a clean dict.
Handles plain-text, HTML-only, and multipart emails.
"""

import email
import email.message
import logging
import re
from email.header import decode_header
from html.parser import HTMLParser

logger = logging.getLogger(__name__)


class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str):
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(html: str) -> str:
    s = _HTMLStripper()
    s.feed(html)
    return re.sub(r"\s+", " ", s.get_text()).strip()


def _decode_str(value: str | bytes | None, charset: str | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode(charset or "utf-8", errors="replace")
        except Exception:
            return value.decode("latin-1", errors="replace")
    return value


def _decode_header_value(raw: str) -> str:
    parts = []
    for chunk, charset in decode_header(raw):
        parts.append(_decode_str(chunk, charset))
    return "".join(parts)


def _get_body(msg: email.message.Message) -> str:
    """Extract readable body text, preferring text/plain over text/html."""
    plain_parts: list[str] = []
    html_parts:  list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if "attachment" in cd:
                continue
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            charset = part.get_content_charset() or "utf-8"
            text = _decode_str(payload, charset)
            if ct == "text/plain":
                plain_parts.append(text)
            elif ct == "text/html":
                html_parts.append(text)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            text = _decode_str(payload, charset)
            if msg.get_content_type() == "text/html":
                html_parts.append(text)
            else:
                plain_parts.append(text)

    if plain_parts:
        return "\n".join(plain_parts).strip()
    if html_parts:
        return _strip_html("\n".join(html_parts))
    return ""


def parse_raw_email(raw_bytes: bytes) -> dict:
    """
    Parse raw email bytes into a dict with keys:
      email_id, sender, recipient, subject, body, date, message_id
    Returns None on parse failure.
    """
    try:
        msg = email.message_from_bytes(raw_bytes)

        sender    = _decode_header_value(msg.get("From",    ""))
        recipient = _decode_header_value(msg.get("To",      ""))
        subject   = _decode_header_value(msg.get("Subject", ""))
        date      = msg.get("Date", "")
        msg_id    = msg.get("Message-ID", "")
        body      = _get_body(msg)

        return {
            "sender":     sender,
            "recipient":  recipient,
            "subject":    subject,
            "body":       body,
            "date":       date,
            "message_id": msg_id.strip(),
        }

    except Exception as exc:
        logger.error("Failed to parse email: %s", exc)
        return None