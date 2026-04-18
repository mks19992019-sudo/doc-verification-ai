from abc import ABC, abstractmethod
from schemas.schemas import ModelOutput


class BaseDetectionModel(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return True

    @abstractmethod
    def run(self, image_path: str, text_map: dict) -> ModelOutput:
        raise NotImplementedError