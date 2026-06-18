"""Summary Agent — executive summary + key insights"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_general_llm
from app.graph.state import EmailGraphState

logger = logging.getLogger(__name__)

_SYSTEM = """You are an executive email analyst.
Write a concise summary of the email and list the key insights or action items.
Respond ONLY with valid JSON:
{
  "summary": "<2-3 sentence summary>",
  "key_insights": ["<insight 1>", "<insight 2>", ...]
}"""


def summary_node(state: EmailGraphState) -> dict:
    logger.info("Summarising email %s", state.get("email_id", "?"))

    subject = state.get("cleaned_subject", "")
    body    = state.get("cleaned_body", "")
    email   = f"Subject: {subject}\n\n{body}" if subject else body

    llm = get_general_llm()

    for attempt in range(2):
        try:
            response = llm.invoke([
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=f"Email:\n\"\"\"\n{email}\n\"\"\""),
            ])
            text = response.content.strip()
            start, end = text.find("{"), text.rfind("}") + 1
            data = json.loads(text[start:end])
            return {
                "summary": data.get("summary", ""),
                "errors":  [],
            }
        except Exception as exc:
            logger.warning("Summariser attempt %d failed: %s", attempt + 1, exc)
            if attempt == 1:
                return {
                    "summary": "Summary unavailable.",
                    "errors":  [f"summarizer: {exc}"],
                }