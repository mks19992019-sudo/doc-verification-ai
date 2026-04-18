from models.base import BaseDetectionModel
from schemas.schemas import ModelOutput


class FontModel(BaseDetectionModel):
    model_id = "FONT"
    available = True

    def run(self, image_path: str, text_map: dict) -> ModelOutput:
        return ModelOutput(
            model_id=self.model_id,
            available=self.available,
            forgery_score=0.10,
            confidence=0.65,
            regions=[],
            signals=["FONT stub: character size variance within normal range"]
        )