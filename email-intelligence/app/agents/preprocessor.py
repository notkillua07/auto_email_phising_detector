"""
Preprocessing Agent

Cleans and normalises the raw email before it reaches any LLM.
This is pure Python — no LLM call needed here.
"""

import logging
import re
import unicodedata
import uuid

from app.graph.state import EmailGraphState

logger = logging.getLogger(__name__)

# Patterns stripped from email bodies before LLM analysis
_TRACKING_URL   = re.compile(r'https?://[^\s]+track[^\s]*', re.I)
_HTML_TAG       = re.compile(r'<[^>]+>')
_DISCLAIMER     = re.compile(
    r'(this (email|message) (is|was) (sent|intended)|confidentiality notice|'
    r'legal disclaimer|unsubscribe|privacy policy)',
    re.I,
)
_SIGNATURE      = re.compile(r'(^|\n)(regards|best|cheers|sincerely|thanks)[,.]?\s*\n.*', re.I | re.S)
_MULTI_NEWLINE  = re.compile(r'\n{3,}')
_MULTI_SPACE    = re.compile(r'[ \t]{2,}')

# Prompt-injection patterns — filtered to protect downstream LLMs
_INJECT         = re.compile(
    r'(ignore (previous|all|prior|above)|disregard|forget (your|the)|'
    r'you are now|new (instruction|role|persona)|<\|.*?\|>)',
    re.I,
)


def _strip_html(text: str) -> str:
    return _HTML_TAG.sub(' ', text)


def _remove_tracking(text: str) -> str:
    return _TRACKING_URL.sub('[URL]', text)


def _remove_disclaimer(text: str) -> str:
    return _DISCLAIMER.sub('', text)


def _remove_signature(text: str) -> str:
    return _SIGNATURE.sub('', text)


def _filter_injection(text: str) -> str:
    return _INJECT.sub('[FILTERED]', text)


def _normalize(text: str) -> str:
    text = unicodedata.normalize('NFKD', text)
    text = _MULTI_NEWLINE.sub('\n\n', text)
    text = _MULTI_SPACE.sub(' ', text)
    return text.strip()


def _estimate_tokens(text: str) -> int:
    # Rough approximation: 1 token ≈ 4 chars
    return max(1, len(text) // 4)


def _truncate(text: str, max_tokens: int = 800) -> str:
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[... truncated]"


def preprocess_node(state: EmailGraphState) -> dict:
    logger.info("Preprocessing email %s", state.get("email_id", "?"))

    subject = state.get("raw_subject", "")
    body    = state.get("raw_body", "")

    try:
        # Subject: strip HTML only
        clean_subject = _normalize(_strip_html(subject))

        # Body: full pipeline
        clean_body = body
        clean_body = _strip_html(clean_body)
        clean_body = _remove_tracking(clean_body)
        clean_body = _remove_disclaimer(clean_body)
        clean_body = _remove_signature(clean_body)
        clean_body = _filter_injection(clean_body)
        clean_body = _normalize(clean_body)
        clean_body = _truncate(clean_body)

        token_count = _estimate_tokens(clean_body)

        return {
            "email_id":        state.get("email_id") or str(uuid.uuid4()),
            "cleaned_subject": clean_subject,
            "cleaned_body":    clean_body,
            "token_count":     token_count,
            "errors":          [],
        }

    except Exception as exc:
        logger.error("Preprocessor error: %s", exc)
        return {
            "cleaned_subject": subject,
            "cleaned_body":    body,
            "token_count":     _estimate_tokens(body),
            "errors":          [f"preprocessor: {exc}"],
        }