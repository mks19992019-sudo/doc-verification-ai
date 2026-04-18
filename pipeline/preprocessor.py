from PIL import Image
from typing import Optional


class Preprocessor:
    async def process(self, file_path: str, language_hint: str) -> dict:
        try:
            img = Image.open(file_path)
            img.verify()
        except Exception:
            pass

        return {
            "language": language_hint,
            "raw_text": "",
            "characters": [],
            "lines": [],
            "blocks": [],
            "image_path": file_path
        }