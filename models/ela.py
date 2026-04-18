import numpy as np
from PIL import Image
import io
from models.base import BaseDetectionModel
from schemas.schemas import ModelOutput, FlaggedRegion


class ELAModel(BaseDetectionModel):
    model_id = "ELA"

    def __init__(self, quality: int = 85, threshold: float = 15.0):
        self.quality = quality
        self.threshold = threshold
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def run(self, image_path: str, text_map: dict) -> ModelOutput:
        try:
            # Step 0: Check if it's a PDF - PIL can't reliably handle PDFs
            if image_path.lower().endswith(".pdf"):
                return ModelOutput(
                    model_id=self.model_id,
                    available=self.available,
                    forgery_score=0.0,
                    confidence=0.50,
                    regions=[],
                    signals=["PDF documents require conversion preprocessing — PDF support will be added in Phase 2"]
                )

            # Step 1: Load the original image
            original = Image.open(image_path)
            if original.mode != "RGB":
                original = original.convert("RGB")

            # Step 2: Re-compress at specified quality level
            # This simulates a "second generation" copy
            buffer = io.BytesIO()
            original.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)
            recompressed = Image.open(buffer)

            # Step 3: Compute absolute pixel difference
            orig_arr = np.array(original, dtype=np.float32)
            comp_arr = np.array(recompressed, dtype=np.float32)
            diff = np.abs(orig_arr - comp_arr)

            # Step 4: Compute ELA score per channel, take max across channels
            ela_map = np.max(diff, axis=2)

            # Step 5: Find bright regions (potential forged areas)
            # Bright = large difference = suspicious compression mismatch
            bright_pixels = ela_map > self.threshold
            bright_count = np.sum(bright_pixels)
            total_pixels = ela_map.size
            bright_ratio = bright_count / total_pixels

            # Forgery score: 0.0 (clean) to 1.0 (heavily forged)
            forgery_score = min(bright_ratio * 50, 1.0)

            # Confidence based on how decisive the signal is
            if bright_ratio < 0.001:
                confidence = 0.85
                signals = ["No significant compression artifacts detected — document appears to be a single-generation scan"]
            elif bright_ratio < 0.01:
                confidence = 0.75
                signals = ["Minor compression inconsistencies found within normal range"]
            else:
                confidence = 0.80
                signals = [f"Compression artifacts detected in {bright_count} pixels ({bright_ratio*100:.2f}% of image)"]

            # Step 6: Find bounding boxes around suspicious regions
            regions = self._find_flagged_regions(ela_map, bright_pixels)

            return ModelOutput(
                model_id=self.model_id,
                available=self.available,
                forgery_score=float(forgery_score),
                confidence=confidence,
                regions=regions,
                signals=signals
            )

        except Exception as e:
            return ModelOutput(
                model_id=self.model_id,
                available=False,
                forgery_score=0.0,
                confidence=0.0,
                regions=[],
                signals=[f"ELA analysis failed: {str(e)}"]
            )

    def _find_flagged_regions(self, ela_map: np.ndarray, bright_mask: np.ndarray) -> list[FlaggedRegion]:
        """
        Cluster bright pixels into bounding boxes.
        Simple approach: divide image into a grid and flag cells with high brightness.
        """
        regions = []
        h, w = ela_map.shape

        # Divide into 8x8 grid cells
        grid_h = h // 8
        grid_w = w // 8

        for i in range(8):
            for j in range(8):
                y_start = i * grid_h
                y_end = (i + 1) * grid_h if i < 7 else h
                x_start = j * grid_w
                x_end = (j + 1) * grid_w if j < 7 else w

                cell = ela_map[y_start:y_end, x_start:x_end]
                cell_bright = cell > self.threshold

                # If more than 30% of cell pixels are bright, flag it
                if np.mean(cell_bright) > 0.30:
                    regions.append(FlaggedRegion(
                        x=int(x_start),
                        y=int(y_start),
                        w=int(x_end - x_start),
                        h=int(y_end - y_start),
                        reason="High compression artifact density — possible pasted region"
                    ))

        return regions[:10]  # Cap at 10 regions