from pydantic import BaseModel, Field


class PreprocessResult(BaseModel):
    clean_subject: str
    clean_body:    str
    token_count:   int


class ClassificationResult(BaseModel):
    classification: str = Field(
        description="One of: spam, phishing, complaint, inquiry, invoice, support_request, legitimate"
    )
    confidence: float = Field(ge=0.0, le=1.0)


class SentimentResult(BaseModel):
    sentiment:  str   = Field(description="positive, neutral, or negative")
    confidence: float = Field(ge=0.0, le=1.0)


class UrgencyResult(BaseModel):
    urgency:    str   = Field(description="low, medium, high, or critical")
    confidence: float = Field(ge=0.0, le=1.0)


class ThreatResult(BaseModel):
    classification: str   = Field(description="phishing or legitimate")
    phishing_type:  str   = Field(description="phishing category or 'legitimate'")
    severity:       str   = Field(description="high, medium, or low")
    threat_score:   float = Field(ge=0.0, le=1.0)
    confidence:     float = Field(ge=0.0, le=1.0)
    explanation:    str


class SummaryResult(BaseModel):
    summary:      str
    key_insights: list[str]