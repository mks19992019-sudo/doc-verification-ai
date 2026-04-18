from models.base import BaseDetectionModel
from schemas.schemas import ModelOutput


class LayoutModel(BaseDetectionModel):
    model_id = "LAYOUT"
    available = True

    def run(self, image_path: str, text_map: dict) -> ModelOutput:
        return ModelOutput(
            model_id=self.model_id,
            available=self.available,
            forgery_score=0.08,
            confidence=0.75,
            regions=[],
            signals=["LAYOUT stub: margin consistency within expected range"]
        )