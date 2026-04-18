from models.ela import ELAModel
from models.font import FontModel
from models.cnn import CNNModel
from models.layout import LayoutModel
from schemas.schemas import ModelResultBundle, ModelOutput


class PipelineRunner:
    def __init__(self):
        from config import settings
        self.models = [
            ELAModel(),
            FontModel(),
            CNNModel(weights_path=settings.CNN_WEIGHTS_PATH),
            LayoutModel()
        ]

    async def run_all(self, document_id: str, file_path: str, doc_type: str, language: str, text_map: dict) -> ModelResultBundle:
        model_outputs = {}

        for model in self.models:
            try:
                output = model.run(file_path, text_map)
            except Exception as e:
                output = ModelOutput(
                    model_id=model.model_id,
                    available=False,
                    forgery_score=0.0,
                    confidence=0.0,
                    regions=[],
                    signals=[f"Error running {model.model_id}: {str(e)}"]
                )
            model_outputs[model.model_id] = output

        return ModelResultBundle(
            document_id=document_id,
            doc_type=doc_type,
            language=language,
            model_outputs=model_outputs
        )

    def get_model_status(self) -> dict:
        return {model.model_id: model.available for model in self.models}