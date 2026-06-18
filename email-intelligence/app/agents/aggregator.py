"""
Aggregation Agent

Runs after all parallel agents complete.
Combines their outputs into one structured analysis_result dict
and determines the final verdict.
"""

import logging
from datetime import datetime, timezone

from app.graph.state import EmailGraphState

logger = logging.getLogger(__name__)


def aggregate_node(state: EmailGraphState) -> dict:
    logger.info("Aggregating results for email %s", state.get("email_id", "?"))

    threat_score  = state.get("threat_score",  0.0)
    classification = state.get("classification", "legitimate")
    phishing_type  = state.get("phishing_type",  "legitimate")
    severity       = state.get("severity",       "low")
    sentiment      = state.get("sentiment",      "neutral")
    urgency        = state.get("urgency",        "low")
    summary        = state.get("summary",        "")
    confidence     = state.get("confidence",     0.0)
    errors         = state.get("errors",         [])

    # Override classification to phishing if threat model is confident
    final_classification = classification
    if threat_score >= 0.6 and final_classification != "phishing":
        final_classification = "phishing"
        logger.info("Overriding classification to phishing (threat_score=%.2f)", threat_score)

    analysis_result = {
        "email_id":        state.get("email_id"),
        "classification":  final_classification,
        "phishing_type":   phishing_type,
        "sentiment":       sentiment,
        "urgency":         urgency,
        "severity":        severity,
        "threat_score":    round(threat_score, 4),
        "confidence":      round(confidence, 4),
        "summary":         summary,
        "token_count":     state.get("token_count", 0),
        "errors":          errors,
        "analyzed_at":     datetime.now(timezone.utc).isoformat(),
    }

    return {
        "classification":  final_classification,
        "analysis_result": analysis_result,
        "errors":          [],
    }