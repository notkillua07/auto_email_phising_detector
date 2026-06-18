"""
Interactive Email Phishing Test Console

Loads the fine-tuned Llama 3.2 3B adapter and lets you paste
one real email at a time to test the model's classification ability.

Usage:
    D:\miniconda3\envs\mlenv\python.exe test_model.py

Controls inside the console:
    - Paste / type your email, then type END on a new line and press Enter
    - Type 'quit' or 'exit' to close
    - Type 'clear' to clear the screen
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)

# ----------------------------------------------------------------- rich UI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich import box
from rich.prompt import Prompt
from rich.padding import Padding

console = Console()

# ----------------------------------------------------------------- config
_HERE        = Path(__file__).parent
_ADAPTER_DIR = _HERE.parent / "outputs" / "llama3.2-3b-phishing" / "final_adapter"
_BASE_MODEL  = "meta-llama/Llama-3.2-3B-Instruct"

SYSTEM_PROMPT = (
    "You are a cybersecurity analyst specializing in email threat detection.\n"
    "When given the body of an email, you must:\n"
    "  1. Determine whether it is a phishing attempt or a legitimate email.\n"
    "  2. Identify the specific phishing category (if applicable).\n"
    "  3. Assess the severity of the threat.\n"
    "  4. Provide a confidence score between 0.0 and 1.0.\n"
    "  5. Give a clear, concise explanation of your reasoning.\n"
    "Always structure your response in the exact JSON format requested.\n"
)

USER_TEMPLATE = (
    'Analyze the following email and respond ONLY with a JSON object containing these keys:\n'
    '  - "classification": "phishing" or "legitimate"\n'
    '  - "phishing_type": the phishing category, or "legitimate" if not phishing\n'
    '  - "severity": "high", "medium", or "low"\n'
    '  - "confidence": a float between 0.0 and 1.0\n'
    '  - "explanation": a brief explanation of your reasoning\n\n'
    'Email:\n"""\n{email}\n"""'
)

# ----------------------------------------------------------------- severity colours
_SEV_COLOUR = {"high": "red", "medium": "yellow", "low": "green"}
_CLS_COLOUR = {"phishing": "red", "legitimate": "green"}


def _extract_json(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}


def load_model():
    console.print(Rule("[bold blue]Loading Model[/bold blue]"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    hf_token = os.environ.get("HF_TOKEN")

    if not _ADAPTER_DIR.exists():
        console.print(f"[red]Adapter not found at {_ADAPTER_DIR}[/red]")
        console.print("[yellow]Run train.bat first to fine-tune the model.[/yellow]")
        sys.exit(1)

    console.print(f"[dim]Adapter : {_ADAPTER_DIR}[/dim]")
    console.print(f"[dim]Base    : {_BASE_MODEL}[/dim]\n")

    with console.status("[cyan]Loading tokenizer…[/cyan]"):
        tokenizer = AutoTokenizer.from_pretrained(_ADAPTER_DIR)

    with console.status("[cyan]Loading base model (bfloat16)…[/cyan]"):
        base = AutoModelForCausalLM.from_pretrained(
            _BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
        )

    with console.status("[cyan]Attaching LoRA adapter…[/cyan]"):
        model = PeftModel.from_pretrained(base, str(_ADAPTER_DIR))
        model.eval()

    gpu = ""
    if hasattr(base, "hf_device_map"):
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)

    console.print(f"[green]✓ Model ready[/green]  [dim]{gpu}[/dim]\n")
    return model, tokenizer


def run_inference(model, tokenizer, email_text: str) -> tuple[str, float]:
    import torch

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": USER_TEMPLATE.format(email=email_text)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
    elapsed = time.perf_counter() - t0

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True)
    return raw_output, elapsed


def display_result(raw_output: str, elapsed: float, email_text: str):
    parsed = _extract_json(raw_output)

    if not parsed:
        console.print(Panel(
            f"[red]Could not parse JSON from model output.[/red]\n\n"
            f"[dim]Raw output:[/dim]\n{raw_output}",
            title="[red]Parse Error[/red]", border_style="red",
        ))
        return

    cls   = parsed.get("classification", "unknown").upper()
    ptype = parsed.get("phishing_type",  "unknown")
    sev   = parsed.get("severity",       "unknown").lower()
    conf  = parsed.get("confidence",     0.0)
    expl  = parsed.get("explanation",    "")

    cls_colour = _CLS_COLOUR.get(cls.lower(), "white")
    sev_colour = _SEV_COLOUR.get(sev, "white")

    # ── verdict banner ───────────────────────────────────────────────────
    if cls.lower() == "phishing":
        banner_style = "bold red on white"
        icon = "⚠️  PHISHING DETECTED"
    else:
        banner_style = "bold green on white"
        icon = "✅  LEGITIMATE EMAIL"

    console.print()
    console.print(Panel(
        Text(icon, style=banner_style, justify="center"),
        border_style=cls_colour, padding=(0, 4),
    ))

    # ── details table ────────────────────────────────────────────────────
    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2),
                  border_style="dim", min_width=56)
    table.add_column("Field",  style="bold dim", width=18)
    table.add_column("Value",  style="bold")

    table.add_row("Classification",
                  Text(cls, style=f"bold {cls_colour}"))
    table.add_row("Phishing Type",  ptype)
    table.add_row("Severity",
                  Text(sev.upper(), style=f"bold {sev_colour}"))

    # confidence bar
    bar_filled = int(conf * 20)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    table.add_row("Confidence",
                  f"[bold]{conf:.0%}[/bold]  [dim]{bar}[/dim]")

    table.add_row("Inference time", f"{elapsed:.2f}s")
    console.print(Padding(table, (1, 0, 0, 0)))

    # ── explanation ──────────────────────────────────────────────────────
    console.print(Panel(
        expl,
        title="[bold]Model Reasoning[/bold]",
        border_style="dim",
        padding=(0, 2),
    ))

    # ── raw JSON (collapsed by default) ─────────────────────────────────
    console.print(Panel(
        raw_output.strip(),
        title="[dim]Raw JSON Output[/dim]",
        border_style="dim",
        expand=False,
    ))


def read_email_from_user() -> str | None:
    """
    Reads multi-line email input from the user.
    User types or pastes the email, then types END alone on a new line.
    Returns None if the user wants to quit.
    """
    console.print(Rule("[dim]New Test[/dim]"))
    console.print(
        "[bold cyan]Paste your email below.[/bold cyan]  "
        "Type [bold]END[/bold] on a new line when done, "
        "or [bold]quit[/bold] to exit.\n"
    )

    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            return None

        stripped = line.strip().lower()
        if stripped in ("quit", "exit"):
            return None
        if stripped == "clear":
            console.clear()
            return read_email_from_user()
        if stripped == "end":
            break
        lines.append(line)

    email_text = "\n".join(lines).strip()
    if not email_text:
        console.print("[yellow]No email text entered. Try again.[/yellow]\n")
        return read_email_from_user()

    return email_text


def main():
    console.clear()
    console.print(Panel(
        "[bold white]Email Phishing Detector[/bold white]\n"
        "[dim]Llama 3.2 3B · QLoRA fine-tuned · Interactive Test Console[/dim]",
        border_style="blue", padding=(1, 4), expand=False,
    ))
    console.print()

    model, tokenizer = load_model()

    while True:
        email_text = read_email_from_user()

        if email_text is None:
            console.print("\n[dim]Goodbye.[/dim]")
            break

        with console.status("[cyan]Analyzing email…[/cyan]", spinner="dots"):
            raw_output, elapsed = run_inference(model, tokenizer, email_text)

        display_result(raw_output, elapsed, email_text)
        console.print()


if __name__ == "__main__":
    main()