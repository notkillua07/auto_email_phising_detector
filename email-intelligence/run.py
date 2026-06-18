"""
Email Intelligence Pipeline — Interactive Runner

Usage:
    D:\miniconda3\envs\mlenv\python.exe run.py

Paste an email (type END to finish), and the full LangGraph pipeline
runs: preprocess → classify → [sentiment | urgency | threat | summary] → aggregate.
"""

import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# ── colour helpers ────────────────────────────────────────────────────────
_SEV  = {"high": "red",    "medium": "yellow",  "low": "green"}
_URG  = {"critical": "red","high": "orange3",   "medium": "yellow", "low": "green"}
_SEN  = {"negative": "red","neutral": "yellow", "positive": "green"}
_CLS  = {"phishing": "red","spam": "orange3",   "legitimate": "green"}


def _col(mapping, key, default="white"):
    return mapping.get(str(key).lower(), default)


def display_result(result: dict):
    ar = result.get("analysis_result", result)

    cls       = ar.get("classification",  "unknown").upper()
    ptype     = ar.get("phishing_type",   "—")
    sentiment = ar.get("sentiment",       "—")
    urgency   = ar.get("urgency",         "—")
    severity  = ar.get("severity",        "—")
    score     = ar.get("threat_score",    0.0)
    conf      = ar.get("confidence",      0.0)
    summary   = ar.get("summary",         "—")
    errors    = ar.get("errors",          [])

    # ── verdict banner ────────────────────────────────────────────────────
    is_phishing = cls.lower() in ("phishing", "spam")
    banner_icon = "⚠️  THREAT DETECTED" if is_phishing else "✅  NO THREAT DETECTED"
    console.print()
    console.print(Panel(
        Text(banner_icon, style=f"bold {'red' if is_phishing else 'green'} on white",
             justify="center"),
        border_style="red" if is_phishing else "green",
        padding=(0, 6),
    ))

    # ── metrics table ─────────────────────────────────────────────────────
    table = Table(box=box.ROUNDED, show_header=True, padding=(0, 2),
                  border_style="dim", header_style="bold dim")
    table.add_column("Field",     style="bold dim",  width=18)
    table.add_column("Value",     style="bold",      width=22)
    table.add_column("Field",     style="bold dim",  width=18)
    table.add_column("Value",     style="bold",      width=22)

    bar_len  = 18
    def _bar(val):
        filled = int(val * bar_len)
        return f"{'█' * filled}{'░' * (bar_len - filled)}  {val:.0%}"

    table.add_row(
        "Classification", Text(cls, style=f"bold {_col(_CLS, cls)}"),
        "Phishing Type",  ptype,
    )
    table.add_row(
        "Sentiment",  Text(sentiment.upper(), style=f"bold {_col(_SEN, sentiment)}"),
        "Urgency",    Text(urgency.upper(),   style=f"bold {_col(_URG, urgency)}"),
    )
    table.add_row(
        "Severity",   Text(severity.upper(),  style=f"bold {_col(_SEV, severity)}"),
        "Confidence", f"{conf:.0%}",
    )
    table.add_row(
        "Threat Score",
        Text(_bar(score), style="bold red" if score >= 0.6 else "bold green"),
        "Token Count",
        str(ar.get("token_count", "—")),
    )

    console.print(table)

    # ── summary panel ─────────────────────────────────────────────────────
    console.print(Panel(summary, title="[bold]Summary[/bold]",
                        border_style="dim", padding=(0, 2)))

    # ── errors ────────────────────────────────────────────────────────────
    if errors:
        console.print(Panel(
            "\n".join(f"• {e}" for e in errors),
            title="[yellow]Warnings[/yellow]", border_style="yellow",
        ))


def read_email() -> str | None:
    console.print(Rule("[dim]New Analysis[/dim]"))
    console.print(
        "[bold cyan]Paste the email below.[/bold cyan]  "
        "Type [bold]END[/bold] on its own line when done, "
        "or [bold]quit[/bold] to exit.\n"
    )
    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            return None
        if line.strip().lower() in ("quit", "exit"):
            return None
        if line.strip().lower() == "end":
            break
        lines.append(line)
    text = "\n".join(lines).strip()
    if not text:
        console.print("[yellow]No text entered.[/yellow]\n")
        return read_email()
    return text


def main():
    console.clear()
    console.print(Panel(
        "[bold white]Email Intelligence Pipeline[/bold white]\n"
        "[dim]LangGraph · Llama 3.2 3B · Multi-Agent Analysis[/dim]",
        border_style="blue", padding=(1, 4), expand=False,
    ))

    # Import here so loading messages appear after the banner
    from app.graph.workflow import graph

    console.print(f"[green]✓ Graph ready[/green]  "
                  f"[dim]nodes: preprocess → classify → "
                  f"[sentiment|urgency|threat|summary] → aggregate[/dim]\n")

    while True:
        raw_email = read_email()
        if raw_email is None:
            console.print("\n[dim]Goodbye.[/dim]")
            break

        # Extract optional subject line (first line if it starts with "Subject:")
        lines   = raw_email.splitlines()
        subject = ""
        body    = raw_email
        if lines and lines[0].lower().startswith("subject:"):
            subject = lines[0][8:].strip()
            body    = "\n".join(lines[1:]).strip()

        initial_state = {
            "email_id":    str(uuid.uuid4()),
            "raw_subject": subject,
            "raw_body":    body,
            "errors":      [],
        }

        with console.status("[cyan]Running pipeline…[/cyan]", spinner="dots"):
            final_state = graph.invoke(initial_state)

        display_result(final_state)
        console.print()


if __name__ == "__main__":
    main()