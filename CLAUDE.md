# LangGraph Email Intelligence Automation — CLAUDE.md

## Project Overview

A production-ready Email Intelligence Automation platform that ingests emails, runs multi-agent AI analysis via LangGraph, stores results in PostgreSQL, and delivers Telegram notifications.

**Root directory:** `email-intelligence/`

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI |
| Orchestration | LangGraph + LangChain |
| LLM | Fine-tuned Qwen / Llama / Mistral via Ollama or vLLM |
| Validation | Pydantic v2 |
| ORM | SQLAlchemy + Alembic |
| Queue | Redis + Celery |
| Database | PostgreSQL |
| Notifications | Telegram Bot API |
| Monitoring | LangSmith + Prometheus + Grafana |

---

## Engineering Rules (Non-Negotiable)

- Python 3.10
- Async architecture everywhere possible
- Pydantic v2 for all schemas and output validation
- SQLAlchemy ORM with Alembic migrations
- Repository pattern for all DB access
- Service layer pattern for business logic
- Structured JSON output enforced on all LLM calls
- All secrets via environment variables only — never hardcoded
- Structured logging with JSON format and correlation IDs
- Exception handling on all agent nodes and external calls
- Unit tests for all agents, repositories, and services (minimum 80% coverage)

---

## Directory Structure

```
email-intelligence/
├── app/
│   ├── api/           # FastAPI routers and endpoints
│   ├── core/          # App startup, config loading
│   ├── config/        # Settings (Pydantic BaseSettings)
│   ├── database/      # DB engine, session factory
│   ├── models/        # SQLAlchemy ORM models
│   ├── repositories/  # Repository pattern — all DB queries
│   ├── services/      # Business logic layer
│   ├── agents/        # Individual LangGraph agent nodes
│   ├── graph/         # LangGraph state + graph definition
│   ├── schemas/       # Pydantic request/response schemas
│   ├── workers/       # Celery task definitions
│   └── utils/         # Shared utilities (logging, token count, etc.)
├── tests/
├── migrations/        # Alembic migration files
├── docs/
├── docker/
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## LangGraph State

```python
class EmailGraphState(TypedDict):
    email_id: str
    raw_subject: str
    raw_body: str
    cleaned_subject: str
    cleaned_body: str
    classification: str
    sentiment: str
    urgency: str
    threat_score: float
    summary: str
    confidence: float
    analysis_result: dict
    errors: list[str]
```

---

## Agent Pipeline

```
START → Preprocess → Classify → [Sentiment | Urgency | Threat | Summary] → Aggregate → DB Save → Telegram → END
```

| Agent | Output |
|-------|--------|
| Preprocessing | `cleaned_body`, `cleaned_subject`, `token_count` |
| Classification | `classification`, `confidence` — Spam/Phishing/Complaint/Inquiry/Invoice/Support |
| Sentiment | `sentiment` — Positive/Neutral/Negative |
| Urgency | `urgency` — Low/Medium/High/Critical |
| Threat Detection | `threat_score` (float 0–1) |
| Summary | `summary` (executive summary) |
| Aggregation | Combined structured JSON |

---

## Database Schema

### `email_logs`
`id, sender, recipient, subject, body, received_at, created_at`

### `analysis_results`
`id, email_id (FK), classification, sentiment, urgency, threat_score, summary, confidence, model_version, created_at`

### `system_errors`
`id, service, error_type, message, stacktrace, created_at`

---

## Environment Variables

```env
DATABASE_URL=
REDIS_URL=
LLM_BASE_URL=
LLM_MODEL=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
LANGSMITH_API_KEY=
```

---

## Build Phases

| Phase | Scope |
|-------|-------|
| 1 | Project init — git, venv, linting, Docker, logging |
| 2 | Database layer — models, migrations, repositories |
| 3 | Email intake — Gmail API (preferred) or IMAP polling |
| 4 | Preprocessing agent — HTML strip, cleanup, injection filter, token control |
| 5 | LLM service — inference wrapper, output validation, fallback |
| 6 | LangGraph state design |
| 7 | Agent node implementation (all 7 agents) |
| 8 | LangGraph orchestration — graph wiring, routing, retry, fallback |
| 9 | Telegram notification system |
| 10 | Error handling + observability (LangSmith, Prometheus, Grafana) |
| 11 | Security hardening — prompt injection, rate limiting, MIME validation |
| 12 | Testing — unit + integration, 80% coverage |
| 13 | Deployment — Docker, docker-compose, health checks |
| 14 | Future: Slack/Discord, CRM, RAG, dashboard, auto-reply, multi-tenant |

---

## Telegram Notification Format

```
Email Analysis

Category: Support Request
Sentiment: Negative
Urgency: High
Threat Score: 5%

Summary:
Customer reports recurring issue and requests escalation.
```

---

## Success Criteria

- Emails automatically ingested and cleaned
- LangGraph executes all agents end-to-end
- Fine-tuned LLM returns validated structured JSON
- Results persisted in PostgreSQL
- Telegram notifications delivered
- Failures recover automatically (retry, DLQ, fallback model)
- Logs and traces observable in LangSmith/Prometheus
- Docker deployment works with one command
- Test suite passes at 80%+ coverage

---

## Implementation Notes

- Classify + Sentiment + Urgency + Threat + Summary agents run in parallel after Classification
- LLM output always validated with Pydantic v2; auto-repair malformed JSON before failing
- Preprocessing must filter prompt injection patterns before sending to LLM
- Use Celery workers for async email processing; FastAPI handles intake webhooks only
- All retries use exponential backoff; unrecoverable failures go to Dead Letter Queue
