from models.base import BaseDetectionModel
from schemas.schemas import ModelOutput


class ELAModel(BaseDetectionModel):
    model_id = "ELA"
    available = True

    def run(self, image_path: str, text_map: dict) -> ModelOutput:
        return ModelOutput(
            model_id=self.model_id,
            available=self.available,
            forgery_score=0.15,
            confidence=0.70,
            regions=[],
            signals=["ELA stub: no significant compression artifacts detected"]
        )