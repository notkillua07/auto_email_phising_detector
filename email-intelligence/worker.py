"""
Email Intelligence Worker

Polls Gmail for new emails every GMAIL_POLL_INTERVAL seconds,
runs each one through the full LangGraph pipeline, and prints
the analysis results to the terminal.

Usage:
    D:\miniconda3\envs\mlenv\python.exe worker.py

.env keys:
    GMAIL_ADDRESS         your Gmail address
    GMAIL_APP_PASSWORD    16-char App Password from Google Account
    GMAIL_POLL_INTERVAL   seconds between polls (default 60)
    OLLAMA_BASE_URL       Ollama server (default http://localhost:11434)
    GENERAL_LLM_MODEL     general model (default llama3.2)
    THREAT_LLM_MODEL      fine-tuned model (default phishing-detector)
"""

import json
import logging
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# ── config ────────────────────────────────────────────────────────────────────
GMAIL_ADDRESS      = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
POLL_INTERVAL      = int(os.getenv("GMAIL_POLL_INTERVAL", "60"))
RESULTS_DIR        = Path(__file__).parent / "outputs" / "processed_emails"

_SEV = {"high": "red", "medium": "yellow", "low": "green"}
_URG = {"critical": "red", "high": "orange3", "medium": "yellow", "low": "green"}
_SEN = {"negative": "red", "neutral": "yellow", "positive": "green"}
_CLS = {"phishing": "red", "spam": "orange3", "legitimate": "green"}


# ── graceful shutdown ─────────────────────────────────────────────────────────
_running = True

def _handle_signal(sig, frame):
    global _running
    console.print("\n[yellow]Shutting down…[/yellow]")
    _running = False

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── display ───────────────────────────────────────────────────────────────────
def _col(mapping, key, default="white"):
    return mapping.get(str(key).lower(), default)


def display_result(email_data: dict, analysis: dict):
    ar  = analysis.get("analysis_result", analysis)
    cls = ar.get("classification", "unknown")

    # ── header ────────────────────────────────────────────────────────────
    is_threat = cls.lower() in ("phishing", "spam")
    icon      = "⚠️  THREAT" if is_threat else "✅  CLEAN"
    console.print(Panel(
        Text(f"{icon}  —  {email_data.get('sender', '?')}",
             style=f"bold {'red' if is_threat else 'green'} on white",
             justify="center"),
        subtitle=f"[dim]{email_data.get('subject', '(no subject)')}[/dim]",
        border_style="red" if is_threat else "green",
    ))

    # ── metrics ───────────────────────────────────────────────────────────
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style="bold dim", width=18)
    t.add_column(style="bold")
    t.add_column(style="bold dim", width=18)
    t.add_column(style="bold")

    score = ar.get("threat_score", 0.0)
    bar   = "█" * int(score * 16) + "░" * (16 - int(score * 16))

    t.add_row("Classification",
              Text(cls.upper(), style=f"bold {_col(_CLS, cls)}"),
              "Phishing Type",
              ar.get("phishing_type", "—"))
    t.add_row("Sentiment",
              Text(ar.get("sentiment", "—").upper(),
                   style=f"bold {_col(_SEN, ar.get('sentiment',''))}"),
              "Urgency",
              Text(ar.get("urgency", "—").upper(),
                   style=f"bold {_col(_URG, ar.get('urgency',''))}"))
    t.add_row("Severity",
              Text(ar.get("severity", "—").upper(),
                   style=f"bold {_col(_SEV, ar.get('severity',''))}"),
              "Confidence",
              f"{ar.get('confidence', 0):.0%}")
    t.add_row("Threat Score",
              Text(f"{bar}  {score:.0%}",
                   style="bold red" if score >= 0.6 else "bold green"),
              "Received",
              email_data.get("date", "—")[:30])

    console.print(t)
    if ar.get("summary"):
        console.print(Panel(ar["summary"], title="[bold]Summary[/bold]",
                            border_style="dim", padding=(0, 2)))


# ── persistence ───────────────────────────────────────────────────────────────
def save_result(email_data: dict, analysis: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    email_id = analysis.get("email_id", str(uuid.uuid4()))
    path     = RESULTS_DIR / f"{ts}_{email_id[:8]}.json"
    record   = {**email_data, "analysis": analysis.get("analysis_result", {})}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    logger.debug("Saved → %s", path)


# ── main loop ─────────────────────────────────────────────────────────────────
def run_worker():
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        console.print(Panel(
            "[red]GMAIL_ADDRESS and GMAIL_APP_PASSWORD must be set in .env[/red]\n\n"
            "Steps:\n"
            "  1. Enable IMAP in Gmail → Settings → See all settings →\n"
            "     Forwarding and POP/IMAP → Enable IMAP\n"
            "  2. Turn on 2-Step Verification on your Google Account\n"
            "  3. Google Account → Security → App passwords\n"
            "  4. Create an App Password for Mail on Windows\n"
            "  5. Add GMAIL_ADDRESS and GMAIL_APP_PASSWORD to .env",
            title="[red]Gmail credentials missing[/red]",
            border_style="red",
        ))
        sys.exit(1)

    from app.intake.gmail_imap import GmailIMAP
    from app.graph.workflow import graph
    from app.notifications.telegram import maybe_notify

    console.print(Panel(
        "[bold white]Email Intelligence Worker[/bold white]\n"
        f"[dim]Polling [cyan]{GMAIL_ADDRESS}[/cyan] "
        f"every [cyan]{POLL_INTERVAL}s[/cyan][/dim]",
        border_style="blue", padding=(1, 4), expand=False,
    ))

    imap = GmailIMAP(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
    imap.connect()

    processed = 0

    while _running:
        try:
            console.print(Rule(f"[dim]{datetime.now().strftime('%H:%M:%S')}  polling inbox…[/dim]"))
            emails = imap.fetch_unseen()

            if not emails:
                console.print(f"[dim]No new emails. Next check in {POLL_INTERVAL}s.[/dim]")
            else:
                for email_data in emails:
                    console.print(f"\n[cyan]Processing:[/cyan] {email_data['subject'][:70]}")

                    initial_state = {
                        "email_id":    str(uuid.uuid4()),
                        "raw_subject": email_data["subject"],
                        "raw_body":    email_data["body"],
                        "errors":      [],
                    }

                    with console.status("[cyan]Running pipeline…[/cyan]", spinner="dots"):
                        final_state = graph.invoke(initial_state)

                    # Attach sender metadata to display
                    email_data["email_id"] = initial_state["email_id"]
                    display_result(email_data, final_state)
                    save_result(email_data, final_state)
                    maybe_notify(email_data, final_state)
                    processed += 1

        except KeyboardInterrupt:
            break
        except Exception as exc:
            logger.error("Worker loop error: %s", exc, exc_info=True)

        # ── wait for next poll ────────────────────────────────────────────
        for _ in range(POLL_INTERVAL):
            if not _running:
                break
            time.sleep(1)

    imap.disconnect()
    console.print(f"\n[green]Worker stopped. Processed {processed} email(s).[/green]")


if __name__ == "__main__":
    run_worker()