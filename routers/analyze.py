import uuid
import time
from fastapi import APIRouter, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from schemas.schemas import AnalyzeResponse, StatusResponse

router = APIRouter(prefix="/api/v1")

preprocessor = None
runner = None
agent = None
xai = None


def set_dependencies(prep, run, ag, xai_layer):
    global preprocessor, runner, agent, xai
    preprocessor = prep
    runner = run
    agent = ag
    xai = xai_layer


ALLOWED_TYPES = {"application/pdf", "image/png", "image/jpeg", "image/tiff"}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile,
    doc_type: str = Form(default="other"),
    language_hint: str = Form(default="en")
):
    from config import settings

    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)

    if file_size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(400, f"File too large. Max size is {settings.MAX_FILE_SIZE_MB}MB")

    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            400,
            f"Invalid file type: {content_type}. Allowed: PDF, PNG, JPEG, TIFF"
        )

    request_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1] if "." in file.filename else "bin"
    saved_path = f"uploads/{request_id}.{ext}"

    with open(saved_path, "wb") as f:
        f.write(file_bytes)

    start_time = time.time()

    text_map = await preprocessor.process(saved_path, language_hint)
    bundle = await runner.run_all(request_id, saved_path, doc_type, language_hint, text_map)
    verdict = await agent.analyze(bundle)
    heatmap_path = await xai.generate_heatmap(request_id, bundle.model_outputs, saved_path)

    processing_time_ms = int((time.time() - start_time) * 1000)

    return AnalyzeResponse(
        request_id=request_id,
        status="complete",
        verdict=verdict.verdict,
        confidence=verdict.confidence,
        summary=verdict.summary,
        explanation=verdict.explanation,
        evidence=verdict.evidence,
        models_used=verdict.models_used,
        cnn_available=verdict.cnn_available,
        heatmap_url=None,
        flagged_regions=verdict.flagged_regions,
        processing_time_ms=processing_time_ms
    )


@router.get("/status", response_model=StatusResponse)
async def status():
    return StatusResponse(
        status="ok",
        models=runner.get_model_status(),
        agent="connected",
        version="1.0.0"
    )