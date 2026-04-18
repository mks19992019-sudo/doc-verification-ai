# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Document forgery detection API for Track C of the ThinkRoot x Vortex Hackathon 2026. A FastAPI backend that analyzes documents through 4 detection models (ELA, FONT, CNN, LAYOUT) and uses a Groq AI agent to produce explainable verdicts (GENUINE/SUSPICIOUS/FORGED/INCONCLUSIVE).

## Running the Server

```bash
cd forgery-detection
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

## Testing the API

```bash
# Health check
curl http://localhost:8000/api/v1/status

# Analyze a document
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@/path/to/test.jpg" \
  -F "doc_type=marksheet" \
  -F "language_hint=en"
```

## Architecture

### Data Flow
```
POST /api/v1/analyze → Preprocessor → PipelineRunner → ForgeryAgent (Groq) → XAI → JSON response
```

### Key Design Patterns

**Singleton injection:** `main.py` creates one instance each of Preprocessor, PipelineRunner, ForgeryAgent, and XAILayer. These are passed to `routers/analyze.py` via `set_dependencies()` — not imported directly — to avoid circular imports.

**Settings centralization:** All config lives in `config.py` using pydantic-settings. Both `main.py` and `routers/analyze.py` import from `config.py`. Never import settings from `main.py`.

**CNN is optional:** CNNModel checks if `weights/resnet50_forgery.pt` exists at init. If not, `available=False`. The pipeline continues with 3 models and the agent handles this gracefully.

**Per-model exception isolation:** PipelineRunner wraps each `model.run()` in try/except. One model failure returns `available=False` with an error signal — it never crashes the pipeline.

### API Endpoints

- `POST /api/v1/analyze` — Upload file, run pipeline, return verdict
- `GET /api/v1/status` — Model availability and system health

### Detection Models

| Model | File | Stub Score | Purpose |
|-------|------|-----------|---------|
| ELA | `models/ela.py` | 0.15 | Compression artifact detection |
| FONT | `models/font.py` | 0.10 | Character-level font inconsistency |
| CNN | `models/cnn.py` | optional | Deep learning classifier (loads if weights exist) |
| LAYOUT | `models/layout.py` | 0.08 | Geometric margin/grid analysis |

### LLM Provider

Uses Groq via `langchain_groq.ChatGroq` with model `llama-3.3-70b-versatile`. API key via `GROQ_API_KEY` in `.env`.

### Schemas

All Pydantic models in `schemas/schemas.py`:
- `FlaggedRegion` — bounding box with reason
- `ModelOutput` — single model result
- `ModelResultBundle` — all model outputs collected
- `AgentVerdict` — AI agent's structured verdict
- `AnalyzeResponse` — final API response
- `StatusResponse` — system status

## Phase 2 Notes

Phase 2 replaces stubs with real implementations:
- Real ELA with OpenCV + scikit-image
- Real font analysis with pdfminer
- Real layout analysis with OpenCV
- Real CNN with PyTorch weights
- PDF conversion with PyMuPDF
- OCR with EasyOCR + Tesseract
- Grad-CAM heatmap

Pipeline runner and agent code do not change when stubs are replaced.