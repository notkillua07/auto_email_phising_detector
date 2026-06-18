from dataclasses import dataclass, field
from pathlib import Path

# Resolve paths relative to this file so scripts can be run from any CWD
_ROOT = Path(__file__).parent.parent.parent  # d:/Anthony/CyberLLM


@dataclass
class TrainingConfig:
    # ------------------------------------------------------------------ Model
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"

    # ---------------------------------------------------------------- Dataset
    train_file: str = str(_ROOT / "dataset" / "Dataset Sharegpt Format" / "sharegpt_train.json")
    val_file: str = str(_ROOT / "dataset" / "Dataset Sharegpt Format" / "sharegpt_val.json")
    test_file: str = str(_ROOT / "dataset" / "Dataset Sharegpt Format" / "sharegpt_test.json")

    # ----------------------------------------------------------------- Output
    output_dir: str = str(_ROOT / "email-intelligence" / "outputs" / "llama3.2-3b-phishing")

    # ------------------------------------------------------------------ LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ----------------------------------------------------------- Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"          # NF4 > int4 for language tasks
    bnb_4bit_use_double_quant: bool = True     # extra 0.4-bit saving
    bnb_4bit_compute_dtype: str = "bfloat16"  # compute in bf16, store in 4-bit

    # --------------------------------------------------------------- Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4      # effective batch = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 1024
    gradient_checkpointing: bool = True

    # ------------------------------------------------------------------- Misc
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    report_to: str = "none"                   # set to "wandb" to enable W&B

    # --------------------------------------------------------- Post-training
    # Each training run saves a timestamped directory here containing the
    # merged model, Ollama Modelfile, and eval_report.json.
    saved_models_dir: str = str(_ROOT / "email-intelligence" / "outputs" / "llama3.2-3b-phishing" / "saved_models")
    eval_n_samples: int = 500                 # test samples to use for evaluation
