"""
Gmail IMAP Poller

Connects to Gmail via IMAP SSL, polls for UNSEEN emails in INBOX,
and yields them as parsed dicts for the LangGraph pipeline.

Setup:
  1. Enable IMAP in Gmail → Settings → See all settings → Forwarding and POP/IMAP
  2. Enable 2-Step Verification on your Google Account
  3. Go to Google Account → Security → 2-Step Verification → App passwords
  4. Create an App Password (select "Mail" + "Windows Computer")
  5. Put the 16-char password in your .env as GMAIL_APP_PASSWORD
"""

import imaplib
import logging
import time
from typing import Generator

from app.intake.email_parser import parse_raw_email

logger = logging.getLogger(__name__)

IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993


class GmailIMAP:
    def __init__(self, address: str, app_password: str):
        self._address      = address
        self._app_password = app_password
        self._conn: imaplib.IMAP4_SSL | None = None

    # ---------------------------------------------------------------- connect
    def connect(self) -> None:
        logger.info("Connecting to Gmail IMAP as %s …", self._address)
        self._conn = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        self._conn.login(self._address, self._app_password)
        logger.info("Connected.")

    def disconnect(self) -> None:
        if self._conn:
            try:
                self._conn.logout()
            except Exception:
                pass
            self._conn = None

    def _reconnect(self) -> None:
        self.disconnect()
        time.sleep(3)
        self.connect()

    # --------------------------------------------------------- fetch unseen
    def fetch_unseen(self) -> list[dict]:
        """
        Fetch all UNSEEN emails from INBOX.
        Marks each email as SEEN after successfully parsing it.
        Returns a list of parsed email dicts.
        """
        if not self._conn:
            self.connect()

        results = []
        try:
            self._conn.select("INBOX")
            status, data = self._conn.search(None, "UNSEEN")
            if status != "OK" or not data[0]:
                return results

            email_ids = data[0].split()
            logger.info("Found %d unseen email(s).", len(email_ids))

            for eid in email_ids:
                try:
                    status, raw_data = self._conn.fetch(eid, "(RFC822)")
                    if status != "OK":
                        continue

                    raw_bytes = raw_data[0][1]
                    parsed    = parse_raw_email(raw_bytes)

                    if parsed:
                        results.append(parsed)
                        # Mark as SEEN so we don't reprocess it
                        self._conn.store(eid, "+FLAGS", "\\Seen")
                    else:
                        logger.warning("Skipping email %s — parse failed.", eid)

                except Exception as exc:
                    logger.error("Error fetching email %s: %s", eid, exc)

        except (imaplib.IMAP4.abort, imaplib.IMAP4.error, OSError) as exc:
            logger.warning("IMAP connection error: %s — reconnecting.", exc)
            self._reconnect()

        return results
