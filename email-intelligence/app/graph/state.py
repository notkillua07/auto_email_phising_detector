import operator
from typing import Annotated, TypedDict


class EmailGraphState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────
    email_id:        str
    raw_subject:     str
    raw_body:        str

    # ── After preprocessing ───────────────────────────────────────────────
    cleaned_subject: str
    cleaned_body:    str
    token_count:     int

    # ── After classification ──────────────────────────────────────────────
    classification:  str   # spam | phishing | complaint | inquiry | invoice | support_request | legitimate

    # ── Parallel agent outputs ────────────────────────────────────────────
    sentiment:       str   # positive | neutral | negative
    urgency:         str   # low | medium | high | critical
    threat_score:    float # 0.0 – 1.0
    phishing_type:   str
    severity:        str   # low | medium | high
    summary:         str
    confidence:      float

    # ── Final aggregated result ───────────────────────────────────────────
    analysis_result: dict

    # ── Error accumulator (merged from all parallel nodes) ────────────────
    errors: Annotated[list[str], operator.add]