"""
Evaluate the fine-tuned phishing detector on the test set.

Standalone usage:
    python evaluate.py

Can also be imported by train.py to run evaluation on the in-memory model
immediately after training (no reload needed):
    from evaluate import evaluate_model
    metrics = evaluate_model(model, tokenizer, test_file, run_dir=run_dir)
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}


def _build_prompt(sample: dict, tokenizer) -> str:
    messages = []
    if system_msg := sample.get("system", "").strip():
        messages.append({"role": "system", "content": system_msg})
    for turn in sample["conversations"][:-1]:   # exclude the final assistant turn
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.inference_mode()
def _run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 300) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Override generation config to silence warnings from Llama 3.2's default
    # generation_config.json which has temperature=0.6 and top_p=0.9 set — those
    # only apply to sampling; we use greedy decoding (do_sample=False).
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,       # explicitly unset so the saved config default is ignored
        top_p=None,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Wrap in autocast so hidden states stay in bfloat16 to match model weights.
    # This prevents the "expected BFloat16 but found Float" error that occurs when
    # gradient-checkpointing hooks are still attached after training.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.generate(**gen_kwargs)

    new_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def evaluate_model(
    model,
    tokenizer,
    test_file: str,
    n_samples: int = 500,
    run_dir: Optional[str] = None,
) -> dict:
    """
    Run evaluation on the test set using an already-loaded model.

    Called automatically by train.py after training using the in-memory model
    (saves the cost of reloading from disk). Can also be called standalone.

    Args:
        model:      Any HuggingFace or PEFT model already on GPU.
        tokenizer:  Matching tokenizer.
        test_file:  Path to sharegpt_test.json.
        n_samples:  How many test samples to evaluate (default 500).
        run_dir:    If provided, saves eval_report.json here.

    Returns:
        dict with accuracy, json_parse_rate, and per-class metrics.
    """
    from sklearn.metrics import classification_report, accuracy_score

    was_training = model.training
    model.eval()

    # Gradient-checkpointing rewrites the forward pass to recompute activations
    # in float32 for backward, which breaks bfloat16 inference. Disable it here.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    # Enable KV cache (disabled during training to save memory)
    if hasattr(model, "config"):
        model.config.use_cache = True

    with open(test_file, encoding="utf-8") as f:
        test_data = json.load(f)

    eval_samples = test_data[:n_samples]
    logger.info(f"Evaluating on {len(eval_samples)} samples …")

    true_labels, pred_labels = [], []
    parse_failures = 0

    for i, sample in enumerate(eval_samples):
        prompt = _build_prompt(sample, tokenizer)
        ground_truth = _extract_json(sample["conversations"][-1]["value"])
        true_label = ground_truth.get("classification", "unknown")

        output = _run_inference(model, tokenizer, prompt)
        parsed = _extract_json(output)
        pred_label = parsed.get("classification", "unknown")

        if not parsed:
            parse_failures += 1

        true_labels.append(true_label)
        pred_labels.append(pred_label)

        if (i + 1) % 50 == 0:
            running_acc = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
            logger.info(f"  [{i+1}/{len(eval_samples)}]  running accuracy: {running_acc:.3f}")

    # Restore model to training state if it was training before eval
    if was_training:
        model.train()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    acc = accuracy_score(true_labels, pred_labels)
    parse_rate = 1.0 - parse_failures / len(eval_samples)
    report_dict = classification_report(
        true_labels, pred_labels, zero_division=0, output_dict=True
    )

    metrics = {
        "n_eval_samples": len(eval_samples),
        "accuracy": round(acc, 6),
        "json_parse_rate": round(parse_rate, 6),
        "parse_failures": parse_failures,
        "classification_report": report_dict,
    }

    # ----------------------------------------------------------------- Print
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(classification_report(true_labels, pred_labels, zero_division=0))
    print(f"Accuracy        : {acc:.4f}")
    print(f"JSON parse rate : {parse_rate:.4f}  ({parse_failures} failures / {len(eval_samples)})")
    print("=" * 60)

    # ----------------------------------------------------------------- Save
    if run_dir:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        report_path = os.path.join(run_dir, "eval_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Eval report saved → {report_path}")

    return metrics


def main():
    """Standalone entry point: reloads model from disk then evaluates."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    from config import TrainingConfig

    cfg = TrainingConfig()
    adapter_dir = os.path.join(cfg.output_dir, "final_adapter")
    hf_token = os.environ.get("HF_TOKEN")

    if not Path(adapter_dir).exists():
        raise FileNotFoundError(
            f"Adapter not found at {adapter_dir}. Run train.py first."
        )

    logger.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    logger.info(f"Loading base model: {cfg.model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    logger.info(f"Loading LoRA adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_file=cfg.test_file,
        n_samples=cfg.eval_n_samples,
    )


if __name__ == "__main__":
    main()