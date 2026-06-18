"""
LLM client factory.

Two clients are used throughout the workflow:
  - general_llm : any Ollama model (default llama3.2) for sentiment,
                  urgency, classification, summary
  - threat_llm  : the fine-tuned phishing-detector model for threat analysis
"""

import os
from functools import lru_cache

from langchain_ollama import ChatOllama


@lru_cache(maxsize=1)
def get_general_llm() -> ChatOllama:
    model = os.getenv("GENERAL_LLM_MODEL", "llama3.2")
    base  = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
    return ChatOllama(model=model, base_url=base, temperature=0)


@lru_cache(maxsize=1)
def get_threat_llm() -> ChatOllama:
    model = os.getenv("THREAT_LLM_MODEL",  "phishing-detector")
    base  = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
    return ChatOllama(model=model, base_url=base, temperature=0)