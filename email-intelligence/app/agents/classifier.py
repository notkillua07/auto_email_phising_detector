"""
Classification Agent

Classifies the email into one of:
  spam | phishing | complaint | inquiry | invoice | support_request | legitimate
"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_general_llm
from app.graph.state import EmailGraphState

logger = logging.getLogger(__name__)

_SYSTEM = """You are an expert email classifier.
Classify the email into EXACTLY one category:
  - spam             : unsolicited bulk email, ads, promotions
  - phishing         : attempts to steal credentials or personal data
  - complaint        : customer complaints or negative feedback
  - inquiry          : questions or requests for information
  - invoice          : billing, payment, or financial documents
  - support_request  : requests for help or technical support
  - legitimate       : normal professional or personal communication

Respond ONLY with a valid JSON object:
{"classification": "<category>", "confidence": <0.0-1.0>}"""


def classify_node(state: EmailGraphState) -> dict:
    logger.info("Classifying email %s", state.get("email_id", "?"))

    subject = state.get("cleaned_subject", "")
    body    = state.get("cleaned_body", "")
    email   = f"Subject: {subject}\n\n{body}"

    llm = get_general_llm()

    for attempt in range(2):
        try:
            response = llm.invoke([
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=f"Classify this email:\n\"\"\"\n{email}\n\"\"\""),
            ])
            text = response.content.strip()
            start, end = text.find("{"), text.rfind("}") + 1
            data = json.loads(text[start:end])

            return {
                "classification": data.get("classification", "legitimate"),
                "confidence":     float(data.get("confidence", 0.5)),
                "errors":         [],
            }

        except Exception as exc:
            logger.warning("Classifier attempt %d failed: %s", attempt + 1, exc)
            if attempt == 1:
                logger.error("Classifier failed after 2 attempts")
                return {
                    "classification": "legitimate",
                    "confidence":     0.0,
                    "errors":         [f"classifier: {exc}"],
                }