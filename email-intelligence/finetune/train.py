"""
QLoRA Fine-tuning — Llama 3.2 3B Instruct → Email Phishing Detector

Environment: conda env at D:\miniconda3\envs\mlenv  (Python 3.11, TRL 0.11, transformers 4.46)
Run via:     train.bat   (sets the conda Python automatically)
Or directly: D:\miniconda3\envs\mlenv\python.exe train.py

What happens automatically after training completes:
  1. LoRA adapter saved to outputs/.../final_adapter/
  2. Evaluation runs on test set  (in-memory model — no reload)
  3. GPU memory freed
  4. LoRA merged into base model on CPU
  5. Full model + Modelfile + eval_report.json saved to:
       outputs/llama3.2-3b-phishing/saved_models/run_<YYYYMMDD_HHMMSS>/
           model/            full safetensors model (Ollama / LangGraph / n8n)
           Modelfile         ollama create phishing-detector -f Modelfile
           eval_report.json  accuracy, F1, JSON parse rate
"""

import gc
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _check_environment() -> None:
    import torch
    if not torch.cuda.is_available():
        raise EnvironmentError("No CUDA GPU detected. Fine-tuning requires a CUDA-capable GPU.")
    gpu = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"Python : {sys.version.split()[0]}")
    logger.info(f"GPU    : {gpu}")
    logger.info(f"VRAM   : {vram_gb:.1f} GB")
    if not os.environ.get("HF_TOKEN"):
        raise EnvironmentError(
            "HF_TOKEN is not set.\n"
            "  1. Copy .env.example → .env\n"
            "  2. Paste your token from https://huggingface.co/settings/tokens"
        )


def main() -> None:
    _check_environment()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    from config import TrainingConfig
    from dataset import prepare_dataset
    from evaluate import evaluate_model
    from merge_and_export import merge_and_save, _print_deploy_instructions

    cfg = TrainingConfig()
    hf_token = os.environ["HF_TOKEN"]

    # Timestamped run directory — created now so eval and merge share the same path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.saved_models_dir, f"run_{timestamp}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")

    # ---------------------------------------------------------------- Tokenizer
    logger.info(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ----------------------------------------------------------------- Datasets
    logger.info("Preparing datasets …")
    train_ds = prepare_dataset(cfg.train_file, tokenizer)
    val_ds = prepare_dataset(cfg.val_file, tokenizer)
    logger.info(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    # ------------------------------------------------------- Completion collator
    # Only compute loss on the assistant's reply tokens (not on the prompt).
    # Llama 3.2 Instruct marks the start of each assistant turn with this sequence.
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    # ------------------------------------------------------------ Quantization
    compute_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )

    # ------------------------------------------------------------------ Model
    logger.info(f"Loading base model (4-bit QLoRA): {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.gradient_checkpointing,
    )

    # ------------------------------------------------------------------- LoRA
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.lora_target_modules,
    )

    # -------------------------------------------------------- Training args
    checkpoint_dir = os.path.join(cfg.output_dir, "checkpoints")
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim="paged_adamw_8bit",
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        evaluation_strategy="steps",    # "eval_strategy" in transformers 5.x; "evaluation_strategy" in 4.x
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=cfg.report_to,
        seed=cfg.seed,
        dataloader_num_workers=0,
    )

    # ----------------------------------------------------------------- Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        packing=False,
        peft_config=lora_cfg,
        args=training_args,
    )

    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    # ================================================================ TRAIN
    logger.info("Starting training …")
    trainer.train()
    logger.info("Training complete.")

    # ======================================== STEP 1 — Save LoRA adapter
    adapter_path = os.path.join(cfg.output_dir, "final_adapter")
    logger.info(f"Saving LoRA adapter → {adapter_path}")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    # ======================================== STEP 2 — Evaluate (in-memory, GPU)
    logger.info("Running post-training evaluation on test set …")
    evaluate_model(
        model=trainer.model,
        tokenizer=tokenizer,
        test_file=cfg.test_file,
        n_samples=cfg.eval_n_samples,
        run_dir=run_dir,
    )

    # ======================================== STEP 3 — Free GPU
    logger.info("Freeing GPU memory before CPU merge …")
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"VRAM freed. Used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ======================================== STEP 4 — Merge on CPU
    logger.info("Merging LoRA adapter into base model (CPU) …")
    merge_and_save(
        base_model_name=cfg.model_name,
        adapter_dir=adapter_path,
        run_dir=run_dir,
        hf_token=hf_token,
    )

    _print_deploy_instructions(run_dir)
    logger.info(f"All artifacts saved → {run_dir}")


if __name__ == "__main__":
    main()