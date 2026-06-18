import json
import logging
from typing import Callable

from datasets import Dataset

logger = logging.getLogger(__name__)


def load_sharegpt_json(path: str) -> Dataset:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} samples from {path}")
    return Dataset.from_list(data)


def make_chat_formatter(tokenizer) -> Callable:
    """
    Returns a function that converts a ShareGPT sample into a fully formatted
    string using the model's own chat template.

    ShareGPT schema:
        { "system": "...", "conversations": [{"from": "human"|"gpt", "value": "..."}] }
    """
    def format_sample(sample: dict) -> str:
        messages = []
        if system_msg := sample.get("system", "").strip():
            messages.append({"role": "system", "content": system_msg})
        for turn in sample["conversations"]:
            role = "user" if turn["from"] == "human" else "assistant"
            messages.append({"role": role, "content": turn["value"]})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # False = include the final assistant turn
        )

    return format_sample


def prepare_dataset(path: str, tokenizer, num_proc: int = 4) -> Dataset:
    raw = load_sharegpt_json(path)
    formatter = make_chat_formatter(tokenizer)
    formatted = raw.map(
        lambda x: {"text": formatter(x)},
        remove_columns=raw.column_names,
        num_proc=num_proc,
        desc=f"Formatting {path.split('/')[-1]}",
    )
    return formatted