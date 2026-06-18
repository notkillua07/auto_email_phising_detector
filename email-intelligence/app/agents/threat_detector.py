"""
Threat Detection Agent

Calls the fine-tuned phishing-detector model via Ollama's /api/generate endpoint
using the exact Llama 3.2 Instruct chat template the model was trained on.

Why /api/generate instead of /api/chat:
  ChatOllama (/api/chat) can conflict with the Modelfile's SYSTEM and STOP
  parameters, producing malformed or truncated JSON.  The generate endpoint
  gives us full control of the prompt and reliably returns clean output.
"""

import json
import logging
import os
import re

import requests

from app.graph.state import EmailGraphState

logger = logging.getLogger(__name__)

# ── Llama 3.2 Instruct chat template ─────────────────────────────────────────
# This is the exact format used during fine-tuning.
_LLAMA_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "{system}"
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{user}"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

_SYSTEM = (
    "You are a cybersecurity analyst specializing in email threat detection.\n"
    "When given the body of an email, you must:\n"
    "  1. Determine whether it is a phishing attempt or a legitimate email.\n"
    "  2. Identify the specific phishing category (if applicable).\n"
    "  3. Assess the severity of the threat.\n"
    "  4. Provide a confidence score between 0.0 and 1.0.\n"
    "  5. Give a clear, concise explanation of your reasoning.\n"
    "Always structure your response in the exact JSON format requested."
)

_USER_TEMPLATE = (
    "Analyze the following email and respond ONLY with a JSON object "
    "containing these keys:\n"
    '  - "classification": "phishing" or "legitimate"\n'
    '  - "phishing_type": the phishing category, or "legitimate" if not phishing\n'
    '  - "severity": "high", "medium", or "low"\n'
    '  - "confidence": a float between 0.0 and 1.0\n'
    '  - "explanation": a brief explanation of your reasoning\n\n'
    "Email:\n\"\"\"\n{email}\n\"\"\""
)


def _build_prompt(email: str) -> str:
    return _LLAMA_TEMPLATE.format(
        system=_SYSTEM,
        user=_USER_TEMPLATE.format(email=email),
    )


def _extract_json(text: str) -> dict:
    """Robust JSON extraction — handles markdown fences and stray text."""
    text = text.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)

    # Find outermost { ... }
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object found in: {text[:200]!r}")

    candidate = text[start:end]
    return json.loads(candidate)


def _call_ollama(prompt: str) -> str:
    """POST to /api/generate and return the response text."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model    = os.getenv("THREAT_LLM_MODEL", "phishing-detector")

    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "repeat_penalty": 1.1,
            "stop": ["<|eot_id|>", "<|end_of_text|>"],
        },
    }

    resp = requests.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def threat_node(state: EmailGraphState) -> dict:
    logger.info("Threat analysis for email %s", state.get("email_id", "?"))

    subject = state.get("cleaned_subject", "")
    body    = state.get("cleaned_body", "")
    email   = f"Subject: {subject}\n\n{body}" if subject else body

    prompt = _build_prompt(email)

    for attempt in range(2):
        try:
            raw = _call_ollama(prompt)
            logger.debug("Threat raw response: %s", raw[:300])

            data = _extract_json(raw)

            sev_map = {"high": 0.85, "medium": 0.50, "low": 0.15}
            sev     = data.get("severity", "low").lower()
            is_phishing = data.get("classification", "").lower() == "phishing"
            threat_score = sev_map.get(sev, 0.0) if is_phishing else 0.0

            return {
                "threat_score":  threat_score,
                "phishing_type": data.get("phishing_type", "legitimate"),
                "severity":      sev,
                "confidence":    float(data.get("confidence", 0.5)),
                "errors":        [],
            }

        except Exception as exc:
            logger.warning("Threat detector attempt %d failed: %s", attempt + 1, exc)
            if attempt == 1:
                logger.error("Threat detector failed after 2 attempts")
                return {
                    "threat_score":  0.0,
                    "phishing_type": "unknown",
                    "severity":      "low",
                    "confidence":    0.0,
                    "errors":        [f"threat_detector: {exc}"],
                }