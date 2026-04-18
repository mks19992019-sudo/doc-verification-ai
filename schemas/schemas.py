from pydantic import BaseModel, Field
from typing import Optional


class FlaggedRegion(BaseModel):
    x: int
    y: int
    w: int
    h: int
    reason: str


class ModelOutput(BaseModel):
    model_id: str
    available: bool
    forgery_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    regions: list[FlaggedRegion] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)


class ModelResultBundle(BaseModel):
    document_id: str
    doc_type: str
    language: str
    model_outputs: dict[str, ModelOutput]


class AgentVerdict(BaseModel):
    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    explanation: str
    evidence: list[str]
    models_used: list[str]
    cnn_available: bool
    request_heatmap: bool
    flagged_regions: list[FlaggedRegion] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    request_id: str
    status: str
    verdict: str
    confidence: float
    summary: str
    explanation: str
    evidence: list[str]
    models_used: list[str]
    cnn_available: bool
    heatmap_url: Optional[str] = None
    flagged_regions: list[FlaggedRegion]
    processing_time_ms: int


class StatusResponse(BaseModel):
    status: str
    models: dict[str, bool]
    agent: str
    version: str