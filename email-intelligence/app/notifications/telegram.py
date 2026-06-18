"""
Telegram Notification — Email Intelligence Alerts

Sends a concise, formatted analysis report to your Telegram chat
after each email is processed by the LangGraph pipeline.

Setup:
  1. Message @BotFather on Telegram → /newbot → copy the token
  2. Message @userinfobot → /start → copy your Chat ID
  3. Add to .env:
       TELEGRAM_BOT_TOKEN=123456789:ABCdef...
       TELEGRAM_CHAT_ID=987654321
       TELEGRAM_NOTIFY_ALL=false   # true = all emails, false = threats only
"""

import html
import logging
import os
import time
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"

# ── emoji maps ────────────────────────────────────────────────────────────────
_CLS_EMOJI = {
    "phishing":        "🚨",
    "spam":            "🗑️",
    "complaint":       "😤",
    "inquiry":         "❓",
    "invoice":         "🧾",
    "support_request": "🛠️",
    "legitimate":      "✅",
}
_SEV_EMOJI  = {"high": "🔴", "medium": "🟡", "low": "🟢"}
_URG_EMOJI  = {"critical": "🚨", "high": "🔴", "medium": "🟡", "low": "🟢"}
_SEN_EMOJI  = {"negative": "😠", "neutral": "😐", "positive": "😊"}


def _e(text: str) -> str:
    """HTML-escape so Telegram doesn't choke on <, >, & characters."""
    return html.escape(str(text), quote=False)


def _threat_score_bar(score: float) -> str:
    filled = int(score * 10)
    return "█" * filled + "░" * (10 - filled) + f"  {score:.0%}"


def format_message(email_data: dict, analysis: dict) -> str:
    """
    Build a Telegram HTML message from the pipeline output.
    Keeps it concise — key facts only, no walls of text.
    """
    ar  = analysis.get("analysis_result", analysis)
    cls = ar.get("classification", "unknown").lower()
    sev = ar.get("severity",       "low").lower()
    urg = ar.get("urgency",        "low").lower()
    sen = ar.get("sentiment",      "neutral").lower()

    is_threat    = cls in ("phishing", "spam")
    score        = ar.get("threat_score", 0.0)
    confidence   = ar.get("confidence",  0.0)
    phishing_type = ar.get("phishing_type", "—")
    summary      = ar.get("summary", "").strip()

    sender  = _e(email_data.get("sender",  "Unknown"))
    subject = _e(email_data.get("subject", "(no subject)"))
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── header ────────────────────────────────────────────────────────────
    cls_emoji = _CLS_EMOJI.get(cls, "📧")
    if is_threat:
        header = f"<b>{cls_emoji} THREAT DETECTED — {cls.upper()}</b>"
    else:
        header = f"<b>{cls_emoji} Email Processed — {cls.upper()}</b>"

    # ── build lines ───────────────────────────────────────────────────────
    lines = [
        header,
        "",
        f"<b>📨 From:</b>    {sender}",
        f"<b>📌 Subject:</b> {subject}",
        "",
        "┄" * 20,
    ]

    if is_threat:
        lines += [
            f"<b>🏷  Type:</b>         {_e(phishing_type)}",
        ]

    lines += [
        f"<b>{_SEV_EMOJI.get(sev,'🔵')} Severity:</b>     {sev.upper()}",
        f"<b>{_URG_EMOJI.get(urg,'🔵')} Urgency:</b>      {urg.upper()}",
        f"<b>{_SEN_EMOJI.get(sen,'😐')} Sentiment:</b>    {sen.capitalize()}",
    ]

    if is_threat:
        lines += [
            f"<b>☠️  Threat Score:</b> <code>{_threat_score_bar(score)}</code>",
            f"<b>🎯 Confidence:</b>   {confidence:.0%}",
        ]

    lines += ["┄" * 20]

    # ── summary ───────────────────────────────────────────────────────────
    if summary:
        # Truncate long summaries for Telegram readability
        short = summary if len(summary) <= 300 else summary[:297] + "…"
        lines += ["", f"<b>📝 Summary:</b>", _e(short)]

    lines += ["", f"<i>⏰ {ts}</i>"]

    return "\n".join(lines)


def send_notification(
    email_data: dict,
    analysis:   dict,
    token:      str,
    chat_id:    str,
    retries:    int = 3,
) -> bool:
    """
    Send the formatted message to Telegram.
    Returns True on success, False if all retries fail.
    """
    message = format_message(email_data, analysis)
    url     = _API.format(token=token)
    payload = {
        "chat_id":                  chat_id,
        "text":                     message,
        "parse_mode":               "HTML",
        "disable_web_page_preview": True,
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info("Telegram notification sent (chat %s)", chat_id)
                return True

            # 429 = rate-limited by Telegram
            if resp.status_code == 429:
                wait = resp.json().get("parameters", {}).get("retry_after", 5)
                logger.warning("Telegram rate-limited. Waiting %ds …", wait)
                time.sleep(wait)
                continue

            logger.error(
                "Telegram API error %d: %s",
                resp.status_code, resp.text[:200],
            )

        except requests.RequestException as exc:
            logger.warning("Telegram attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)

    logger.error("Telegram notification failed after %d attempts.", retries)
    return False


def maybe_notify(email_data: dict, analysis: dict) -> None:
    """
    Called from worker.py after each pipeline run.
    Reads token / chat_id / notify_all from environment.
    Sends if:
      - TELEGRAM_NOTIFY_ALL=true  → always send
      - TELEGRAM_NOTIFY_ALL=false → only send for threats (phishing / spam)
    """
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID",   "").strip()

    if not token or not chat_id:
        logger.debug("Telegram not configured — skipping notification.")
        return

    ar         = analysis.get("analysis_result", analysis)
    cls        = ar.get("classification", "").lower()
    is_threat  = cls in ("phishing", "spam")
    notify_all = os.getenv("TELEGRAM_NOTIFY_ALL", "false").lower() == "true"

    if not notify_all and not is_threat:
        logger.debug("Email is not a threat — skipping Telegram notification.")
        return

    send_notification(email_data, analysis, token, chat_id)