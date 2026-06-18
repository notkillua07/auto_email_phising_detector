"""Urgency Agent — low | medium | high | critical"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_general_llm
from app.graph.state import EmailGraphState

logger = logging.getLogger(__name__)

_SYSTEM = """You are an urgency assessment expert.
Rate how urgently this email requires attention.
Respond ONLY with valid JSON:
{"urgency": "low" | "medium" | "high" | "critical", "confidence": <0.0-1.0>}

Guidelines:
  critical : immediate action required, time-sensitive threat or crisis
  high     : requires response within hours
  medium   : requires response within a day or two
  low      : informational, no action or flexible timeline"""


def urgency_node(state: EmailGraphState) -> dict:
    logger.info("Urgency assessment for email %s", state.get("email_id", "?"))

    body = state.get("cleaned_body", "")
    llm  = get_general_llm()

    for attempt in range(2):
        try:
            response = llm.invoke([
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=f"Email:\n\"\"\"\n{body}\n\"\"\""),
            ])
            text = response.content.strip()
            start, end = text.find("{"), text.rfind("}") + 1
            data = json.loads(text[start:end])
            return {
                "urgency": data.get("urgency", "low"),
                "errors":  [],
            }
        except Exception as exc:
            logger.warning("Urgency attempt %d failed: %s", attempt + 1, exc)
            if attempt == 1:
                return {"urgency": "low", "errors": [f"urgency: {exc}"]}