import os
from models.base import BaseDetectionModel
from schemas.schemas import ModelOutput


class CNNModel(BaseDetectionModel):
    model_id = "CNN"

    def __init__(self, weights_path: str = "weights/resnet50_forgery.pt"):
        self._available = os.path.exists(weights_path)

    @property
    def available(self) -> bool:
        return self._available

    def run(self, image_path: str, text_map: dict) -> ModelOutput:
        if not self.available:
            return ModelOutput(
                model_id=self.model_id,
                available=False,
                forgery_score=0.0,
                confidence=0.0,
                regions=[],
                signals=["CNN model not loaded — weights file not found"]
            )
        return ModelOutput(
            model_id=self.model_id,
            available=self.available,
            forgery_score=0.0,
            confidence=0.0,
            regions=[],
            signals=[]
        )