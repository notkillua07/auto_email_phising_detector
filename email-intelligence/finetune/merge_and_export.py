"""
Merge LoRA adapter → full fp16 model and export for deployment.

Standalone usage:
    python merge_and_export.py

Can also be imported by train.py to run the merge immediately after training:
    from merge_and_export import merge_and_save
    merge_and_save(base_model_name, adapter_dir, run_dir, hf_token)

Produces inside run_dir:
    model/       full safetensors model (HuggingFace format)
    Modelfile    ready-to-use Ollama Modelfile

After this script:
  [Ollama]
    cd <run_dir>
    ollama create phishing-detector -f Modelfile

  [LangGraph / LangChain]
    from langchain_community.chat_models import ChatOllama
    llm = ChatOllama(model="phishing-detector", temperature=0.1)

  [n8n HTTP node]
    POST http://localhost:11434/api/generate
    { "model": "phishing-detector", "prompt": "...", "stream": false }

  [GGUF — optional, for llama.cpp / Ollama with GGUF]
    python llama.cpp/convert_hf_to_gguf.py <run_dir>/model/ --outfile phishing.gguf
    ./llama.cpp/llama-quantize phishing.gguf phishing-q4_k_m.gguf q4_k_m
"""

import gc
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


def _make_modelfile(model_subdir: str = "./model") -> str:
    return f"""\
FROM {model_subdir}

SYSTEM \"\"\"You are a cybersecurity analyst specializing in email threat detection.
When given the body of an email, you must:
  1. Determine whether it is a phishing attempt or a legitimate email.
  2. Identify the specific phishing category (if applicable).
  3. Assess the severity of the threat.
  4. Provide a confidence score between 0.0 and 1.0.
  5. Give a clear, concise explanation of your reasoning.
Always structure your response in the exact JSON format requested.\"\"\"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
"""


def merge_and_save(
    base_model_name: str,
    adapter_dir: str,
    run_dir: str,
    hf_token: Optional[str] = None,
) -> str:
    """
    Merge LoRA weights into the base model and save to run_dir/model/.

    The merge is done entirely on CPU to avoid VRAM OOM — the merge operation
    is simple weight addition, not a forward pass, so CPU is fine.

    Args:
        base_model_name: HuggingFace model ID (e.g. 'meta-llama/Llama-3.2-3B-Instruct').
        adapter_dir:     Path to the saved LoRA adapter directory.
        run_dir:         Destination directory for this training run's artifacts.
        hf_token:        HuggingFace access token.

    Returns:
        Path to the saved merged model directory.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model_dir = os.path.join(run_dir, "model")
    modelfile_path = os.path.join(run_dir, "Modelfile")
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ Load base model in fp16
    logger.info(f"Loading base model in fp16 on CPU for merge: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        token=hf_token,
    )

    # -------------------------------------------- Attach & merge LoRA adapter
    logger.info(f"Attaching LoRA adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir, device_map="cpu")

    logger.info("Merging LoRA weights into base model …")
    model = model.merge_and_unload()
    logger.info("Merge complete.")

    # ------------------------------------------------------------------ Save
    logger.info(f"Saving merged model → {model_dir}")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(model_dir)
    logger.info("Merged model saved (safetensors format).")

    # ------------------------------------------------------------ Modelfile
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(_make_modelfile(model_subdir="./model"))
    logger.info(f"Ollama Modelfile written → {modelfile_path}")

    # --------------------------------------------------------- Free CPU RAM
    del model
    del base_model
    gc.collect()

    return model_dir


def _print_deploy_instructions(run_dir: str) -> None:
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"Run directory : {run_dir}")
    print(f"Merged model  : {os.path.join(run_dir, 'model')}")
    print(f"Ollama file   : {os.path.join(run_dir, 'Modelfile')}")
    print(f"Eval report   : {os.path.join(run_dir, 'eval_report.json')}")
    print()
    print("Deploy with Ollama:")
    print(f"  cd \"{run_dir}\"")
    print("  ollama create phishing-detector -f Modelfile")
    print()
    print("Use in LangGraph:")
    print("  from langchain_community.chat_models import ChatOllama")
    print('  llm = ChatOllama(model="phishing-detector", temperature=0.1)')
    print()
    print("Use in n8n (HTTP Request node):")
    print("  POST http://localhost:11434/api/generate")
    print('  { "model": "phishing-detector", "prompt": "...", "stream": false }')
    print("=" * 60)


def main():
    """Standalone entry point: merges the adapter produced by train.py."""
    from config import TrainingConfig

    cfg = TrainingConfig()
    adapter_dir = os.path.join(cfg.output_dir, "final_adapter")
    hf_token = os.environ.get("HF_TOKEN")

    if not Path(adapter_dir).exists():
        raise FileNotFoundError(
            f"Adapter not found at {adapter_dir}. Run train.py first."
        )

    # Standalone runs go into saved_models/standalone_<timestamp>
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.saved_models_dir, f"standalone_{timestamp}")

    merge_and_save(
        base_model_name=cfg.model_name,
        adapter_dir=adapter_dir,
        run_dir=run_dir,
        hf_token=hf_token,
    )
    _print_deploy_instructions(run_dir)


if __name__ == "__main__":
    main()