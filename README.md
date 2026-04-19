# Document Forgery Detection API

An explainable AI system that detects forged or manipulated documents (marksheets, certificates, government IDs) and explains exactly why a document was flagged. Supports regional Indian languages and produces outputs that non-technical verification officers can understand.

## Tech Stack

- **Backend:** FastAPI (Python 3.11+)
- **AI Agent:** Anthropic Claude API (claude-sonnet-4-20250514)
- **Image Processing:** Pillow, NumPy, OpenCV
- **Detection Models:** ELA, Font Analysis, CNN (ResNet-50), Layout Analysis
- **OCR:** EasyOCR, Tesseract (Phase 2)

## Project Structure

```
forgery-detection/
├── main.py                  # FastAPI app entry point
├── routers/
│   └── analyze.py           # POST /api/v1/analyze, GET /api/v1/status
├── pipeline/
│   ├── preprocessor.py      # File → normalized image + TextMap
│   ├── runner.py            # Runs all 4 detection models
│   └── xai.py               # Grad-CAM heatmap (Phase 2)
├── models/
│   ├── base.py              # Abstract BaseDetectionModel
│   ├── ela.py               # Error Level Analysis
│   ├── font.py              # Font consistency analysis
│   ├── cnn.py               # CNN deep learning classifier
│   └── layout.py            # Layout structure analysis
├── agent/
│   ├── agent.py             # Claude AI agent
│   └── prompts.py           # System prompt
├── schemas/
│   └── schemas.py           # Pydantic models
└── uploads/                 # Temp uploaded files
```

## Setup

```bash
# Clone the repo
cd forgery-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# Download CNN weights (optional - CNN works without it)
# Place weights at weights/resnet50_forgery.pt

# Run server
uvicorn main:app --reload --port 8000
```

## Environment Variables

Create a `.env` file with:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
CNN_WEIGHTS_PATH=weights/resnet50_forgery.pt
MAX_FILE_SIZE_MB=20
```

Get your Anthropic API key from: [console.anthropic.com](https://console.anthropic.com)

## API Endpoints

### POST /api/v1/analyze

Upload a document for forgery analysis.

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@document.jpg" \
  -F "doc_type=marksheet" \
  -F "language_hint=en"
```

**Request:**
- `file`: PDF, PNG, JPG, or TIFF (max 20MB)
- `doc_type`: `marksheet` | `certificate` | `id` | `other` (default: `other`)
- `language_hint`: `en` | `hi` | `ta` | `bn` | `te` (default: `en`)

**Response:**
```json
{
  "request_id": "uuid",
  "status": "ok",
  "verdict": "GENUINE",
  "confidence": 0.92,
  "summary": "The document appears authentic based on low forgery signals.",
  "explanation": "No compression artifacts or layout irregularities detected.",
  "evidence": ["No significant compression artifacts detected"],
  "models_used": ["ELA", "LAYOUT"],
  "cnn_available": false,
  "heatmap_url": null,
  "flagged_regions": [],
  "processing_time_ms": 1250
}
```

### GET /api/v1/status

Check system health and model availability.

```bash
curl http://localhost:8000/api/v1/status
```

**Response:**
```json
{
  "status": "ok",
  "models": {
    "ELA": true,
    "FONT": true,
    "CNN": false,
    "LAYOUT": true
  },
  "agent": "connected",
  "version": "1.0.0"
}
```

## Verdict meanings

| Verdict | Meaning |
|---------|---------|
| `GENUINE` | Document appears authentic |
| `SUSPICIOUS` | Some forgery signals detected |
| `FORGED` | Strong evidence of tampering |
| `INCONCLUSIVE` | Not enough models available |

## Detection Models

### 1. ELA (Error Level Analysis)
Detects pixel-level tampering using JPEG compression artifact analysis. Compares original image against a re-compressed version to find regions with different compression histories.

### 2. Font Analysis
Detects character-level font inconsistencies. Statistical outlier detection on character sizes to flag tampered text.

### 3. CNN (ResNet-50)
Deep learning classifier trained on genuine vs forged documents. Optional — system works without it.

### 4. Layout Analysis
Checks geometric structural consistency. Analyzes margin widths, text line spacing, and document skew.

## License

MIT
