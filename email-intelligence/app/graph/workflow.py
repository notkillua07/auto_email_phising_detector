"""
LangGraph Workflow — Email Intelligence Pipeline

Flow:
  START
    │
    ▼
  preprocess          (clean + sanitise email)
    │
    ▼
  classify            (spam / phishing / inquiry / …)
    │
    ├──▶ sentiment     ─┐
    ├──▶ urgency       ─┤  (parallel)
    ├──▶ threat        ─┤
    └──▶ summary       ─┘
                        │
                        ▼
                     aggregate   (merge + final verdict)
                        │
                        ▼
                       END
"""

from langgraph.graph import END, START, StateGraph

from app.agents.aggregator    import aggregate_node
from app.agents.classifier    import classify_node
from app.agents.preprocessor  import preprocess_node
from app.agents.sentiment     import sentiment_node
from app.agents.summarizer    import summary_node
from app.agents.threat_detector import threat_node
from app.agents.urgency       import urgency_node
from app.graph.state          import EmailGraphState


def build_graph():
    builder = StateGraph(EmailGraphState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("preprocess", preprocess_node)
    builder.add_node("classify",   classify_node)
    builder.add_node("sentiment",  sentiment_node)
    builder.add_node("urgency",    urgency_node)
    builder.add_node("threat",     threat_node)
    builder.add_node("summary",    summary_node)
    builder.add_node("aggregate",  aggregate_node)

    # ── Sequential edges ──────────────────────────────────────────────────
    builder.add_edge(START,        "preprocess")
    builder.add_edge("preprocess", "classify")

    # ── Fan-out from classify → 4 parallel agents ─────────────────────────
    builder.add_edge("classify",   "sentiment")
    builder.add_edge("classify",   "urgency")
    builder.add_edge("classify",   "threat")
    builder.add_edge("classify",   "summary")

    # ── Fan-in: all 4 must finish before aggregate runs ───────────────────
    builder.add_edge("sentiment",  "aggregate")
    builder.add_edge("urgency",    "aggregate")
    builder.add_edge("threat",     "aggregate")
    builder.add_edge("summary",    "aggregate")

    builder.add_edge("aggregate",  END)

    return builder.compile()


# Module-level compiled graph — import this everywhere
graph = build_graph()