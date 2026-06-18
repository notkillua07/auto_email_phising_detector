# Email Intelligence — Automated Email Phishing Detector

A multi-agent AI pipeline that ingests emails, analyzes them for phishing threats, and delivers structured intelligence reports. Built with **LangGraph** for orchestration and a **QLoRA fine-tuned Llama 3.2 3B** model for threat detection.

## How It Works

```
Email Input
    │
    ▼
Preprocess ──► Classify ──┬──► Sentiment  ──┐
                           ├──► Urgency     ─┤  (parallel)
                           ├──► Threat      ─┤
                           └──► Summary     ─┘
                                             │
                                             ▼
                                         Aggregate ──► Final Verdict
```

**7 specialized agents** run in a LangGraph state graph:

| Agent | What It Does |
|---|---|
| **Preprocessor** | Strips HTML, tracking URLs, disclaimers, signatures, and prompt injection patterns |
| **Classifier** | Categorizes email: spam, phishing, complaint, inquiry, invoice, support request, or legitimate |
| **Sentiment** | Detects tone: positive, neutral, or negative |
| **Urgency** | Rates priority: low, medium, high, or critical |
| **Threat Detector** | Uses the fine-tuned phishing model to score threat level (0.0–1.0), phishing type, and severity |
| **Summarizer** | Generates an executive summary of the email |
| **Aggregator** | Merges all outputs into a final verdict — overrides classification to phishing if threat score >= 0.6 |

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph + LangChain |
| LLM (General) | Llama 3.2 via Ollama |
| LLM (Threat) | Fine-tuned Llama 3.2 3B Instruct (QLoRA) |
| Fine-tuning | HuggingFace Transformers + TRL + PEFT + BitsAndBytes |
| Email Intake | Gmail IMAP polling |
| CLI | Rich (interactive terminal UI) |

## Project Structure

```
email-intelligence/
├── app/
│   ├── agents/          # All 7 LangGraph agent nodes
│   ├── core/            # LLM client factory (Ollama)
│   ├── graph/           # State definition + workflow wiring
│   ├── intake/          # Gmail IMAP poller + email parser
│   └── schemas/         # Pydantic v2 schemas
├── finetune/            # QLoRA fine-tuning pipeline
│   ├── train.py         # Training entry point
│   ├── evaluate.py      # Test set evaluation (accuracy, F1, JSON parse rate)
│   ├── merge_and_export.py  # LoRA merge + Ollama Modelfile export
│   ├── dataset.py       # Dataset preparation (ShareGPT format)
│   └── config.py        # Training hyperparameters
├── outputs/             # Checkpoints, adapters, saved models, processed emails
├── run.py               # Interactive CLI runner
dataset/
├── Dataset Sharegpt Format/   # Train/val/test splits (ShareGPT format)
└── Dataset - Qwen/            # Alternative Qwen ChatML format
```

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- CUDA-capable GPU (for fine-tuning only)

### 1. Run the Pipeline (Inference)

Pull a general-purpose model with Ollama and load the fine-tuned phishing detector:

```bash
ollama pull llama3.2
ollama create phishing-detector -f <path-to-Modelfile>
```

Run the interactive CLI:

```bash
cd email-intelligence
pip install langchain-ollama langgraph rich python-dotenv requests
python run.py
```

Paste any email, type `END`, and get a full analysis:

```
┌─────────────────────────────────────┐
│      ⚠️  THREAT DETECTED            │
└─────────────────────────────────────┘
┌──────────────────┬──────────────────┐
│ Classification   │ PHISHING         │
│ Phishing Type    │ credential_theft │
│ Sentiment        │ NEUTRAL          │
│ Urgency          │ HIGH             │
│ Severity         │ HIGH             │
│ Threat Score     │ ████████████░░░░ 85% │
│ Confidence       │ 92%              │
└──────────────────┴──────────────────┘
```

### 2. Fine-Tune the Phishing Model

```bash
cd email-intelligence/finetune
cp .env.example .env
# Add your HuggingFace token to .env

pip install -r requirements.txt
python train.py
```

This will:
1. Load Llama 3.2 3B Instruct in 4-bit QLoRA
2. Train on the phishing dataset (3 epochs, effective batch size 16)
3. Evaluate on the test set (accuracy, F1, JSON parse rate)
4. Merge LoRA into the base model
5. Export a standalone model + Ollama Modelfile

### 3. Gmail Intake (Optional)

To poll emails directly from Gmail:

1. Enable IMAP in Gmail settings
2. Enable 2-Step Verification on your Google Account
3. Create an App Password (Security > 2-Step Verification > App passwords)
4. Set environment variables:
   ```env
   GMAIL_ADDRESS=your.email@gmail.com
   GMAIL_APP_PASSWORD=your-16-char-app-password
   ```

## Fine-Tuning Details

| Parameter | Value |
|---|---|
| Base Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Method | QLoRA (4-bit NF4 + double quantization) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Learning Rate | 2e-4 (cosine schedule) |
| Epochs | 3 |
| Batch Size | 4 x 4 gradient accumulation = 16 effective |
| Max Sequence Length | 1024 tokens |
| Optimizer | Paged AdamW 8-bit |

The model is trained on completion tokens only (loss computed on assistant responses, not prompts) using TRL's `DataCollatorForCompletionOnlyLM`.

## Environment Variables

| Variable | Description |
|---|---|
| `GENERAL_LLM_MODEL` | Ollama model for general agents (default: `llama3.2`) |
| `THREAT_LLM_MODEL` | Ollama model for threat detection (default: `phishing-detector`) |
| `OLLAMA_BASE_URL` | Ollama API endpoint (default: `http://localhost:11434`) |
| `HF_TOKEN` | HuggingFace token (fine-tuning only) |

## License

This project is for educational and research purposes.
