"""Sentiment Agent — positive | neutral | negative"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_general_llm
from app.graph.state import EmailGraphState

logger = logging.getLogger(__name__)

_SYSTEM = """You are a sentiment analysis expert.
Analyse the tone of the email and respond ONLY with valid JSON:
{"sentiment": "positive" | "neutral" | "negative", "confidence": <0.0-1.0>}"""


def sentiment_node(state: EmailGraphState) -> dict:
    logger.info("Sentiment analysis for email %s", state.get("email_id", "?"))

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
                "sentiment": data.get("sentiment", "neutral"),
                "errors":    [],
            }
        except Exception as exc:
            logger.warning("Sentiment attempt %d failed: %s", attempt + 1, exc)
            if attempt == 1:
                return {"sentiment": "neutral", "errors": [f"sentiment: {exc}"]}