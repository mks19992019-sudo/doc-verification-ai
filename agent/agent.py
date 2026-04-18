import json
from langchain_groq import ChatGroq
from agent.prompts import SYSTEM_PROMPT
from schemas.schemas import AgentVerdict, ModelResultBundle, FlaggedRegion
from config import settings


class ForgeryAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.GROQ_API_KEY,
            temperature=0.0
        )

    async def analyze(self, bundle: ModelResultBundle) -> AgentVerdict:
        bundle_json = bundle.model_dump_json(indent=2)
        user_message = f"Here is the detection model output bundle for you to analyze:\n\n{bundle_json}"

        prompt_with_system = f"{SYSTEM_PROMPT}\n\n{user_message}"

        response = self.llm.invoke(prompt_with_system)

        text = response.content
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
            flagged_regions = [
                FlaggedRegion(**r) for r in data.get("flagged_regions", [])
            ]
            return AgentVerdict(
                verdict=data["verdict"],
                confidence=data["confidence"],
                summary=data["summary"],
                explanation=data["explanation"],
                evidence=data["evidence"],
                models_used=data["models_used"],
                cnn_available=data["cnn_available"],
                request_heatmap=data["request_heatmap"],
                flagged_regions=flagged_regions
            )
        except Exception:
            return AgentVerdict(
                verdict="INCONCLUSIVE",
                confidence=0.0,
                summary="Agent failed to parse model outputs.",
                explanation="Internal error in agent response parsing.",
                evidence=[],
                models_used=[],
                cnn_available=False,
                request_heatmap=False,
                flagged_regions=[]
            )