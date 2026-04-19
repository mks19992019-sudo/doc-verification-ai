import cv2
import numpy as np
from models.base import BaseDetectionModel
from schemas.schemas import ModelOutput, FlaggedRegion


class LayoutModel(BaseDetectionModel):
    model_id = "LAYOUT"

    def __init__(self, line_angle_threshold: float = 5.0, margin_deviation_threshold: float = 0.15):
        self.line_angle_threshold = line_angle_threshold  # degrees
        self.margin_deviation_threshold = margin_deviation_threshold  # 15% deviation is suspicious
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def run(self, image_path: str, text_map: dict) -> ModelOutput:
        try:
            # Step 0: Check if it's a PDF - OpenCV can't handle PDFs
            if image_path.lower().endswith(".pdf"):
                return ModelOutput(
                    model_id=self.model_id,
                    available=self.available,
                    forgery_score=0.0,
                    confidence=0.50,
                    regions=[],
                    signals=["PDF documents require OCR preprocessing — PDF conversion will be added in Phase 2"]
                )

            # Step 1: Load image in grayscale
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"Could not load image: {image_path}")

            h, w = gray.shape

            # Step 1.5: Minimum image size check
            # Small images (like 100x100) will have noisy line detection
            if h < 300 or w < 300:
                return ModelOutput(
                    model_id=self.model_id,
                    available=self.available,
                    forgery_score=0.0,
                    confidence=0.50,
                    regions=[],
                    signals=[f"Image too small ({w}x{h}) for reliable layout analysis — need higher resolution scan"]
                )

            # Step 2: Edge detection with Canny
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

            # Step 3: Detect straight lines using HoughLinesP
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=50,
                maxLineGap=20
            )

            if lines is None or len(lines) == 0:
                return ModelOutput(
                    model_id=self.model_id,
                    available=self.available,
                    forgery_score=0.0,
                    confidence=0.60,
                    regions=[],
                    signals=["No structural lines detected — image may be a photograph rather than a document"]
                )

            # Distinguish documents from photographs using line quality
            # Documents have long, axis-aligned lines. Photos have short fragmented edges.
            lengths = [np.sqrt((l[0][2]-l[0][0])**2 + (l[0][3]-l[0][1])**2) for l in lines]
            avg_length = np.mean(lengths)
            max_length = max(lengths) if lengths else 0

            # A real scanned document: 5-30 structural lines (margins, table borders, dividers)
            # A photograph processed with Canny+Hough: 100+ random edge fragments
            # A mobile photo of a document: intermediate, but still has clear long margins
            too_many_lines = len(lines) > 100
            avg_too_short = avg_length < 80
            no_long_lines = max_length < 150

            is_not_document = too_many_lines or avg_too_short or no_long_lines

            if is_not_document:
                return ModelOutput(
                    model_id=self.model_id,
                    available=self.available,
                    forgery_score=0.0,
                    confidence=0.60,
                    regions=[],
                    signals=[f"No document structure detected — image appears to be a photograph (found {len(lines)} edge fragments, avg length {avg_length:.0f}px)"]
                )

            # Need at least 3 lines for meaningful analysis
            if len(lines) < 3:
                return ModelOutput(
                    model_id=self.model_id,
                    available=self.available,
                    forgery_score=0.0,
                    confidence=0.50,
                    regions=[],
                    signals=[f"Only {len(lines)} lines detected — need more structure for reliable analysis"]
                )

            # Step 4: Classify lines as horizontal or vertical
            horizontal_lines = []
            vertical_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                # Normalize angle: horizontal lines are ~0 or ~180, vertical ~90
                if abs(angle) < self.line_angle_threshold or abs(angle - 180) < self.line_angle_threshold:
                    horizontal_lines.append((y1, y2, length))
                elif abs(angle - 90) < self.line_angle_threshold:
                    vertical_lines.append((x1, x2, length))

            # Step 5: Analyze margin consistency using grid regularity
            # Documents have a repeating grid of text lines — check if spacings follow a pattern
            forgery_score = 0.0
            signals = []
            regions = []

            # Use median spacing as reference — robust to margin gaps (outliers)
            if horizontal_lines:
                y_positions = sorted(set([h[0] for h in horizontal_lines]))
                spacings = np.diff(y_positions)

                # Filter out tiny spacings (less than 5 pixels - likely noise)
                valid_spacings = [s for s in spacings if s >= 5]

                if len(valid_spacings) >= 3:
                    # Use median as reference — margins don't affect it
                    ref_spacing = np.median(valid_spacings)
                    deviations = [abs(s - ref_spacing) / ref_spacing for s in valid_spacings]
                    max_deviation = min(max(deviations), 1.0)

                    # Flag only if deviation is severe (> 30%) AND affects a significant portion
                    severe_count = sum(1 for d in deviations if d > 0.30)
                    severe_ratio = severe_count / len(deviations) if deviations else 0

                    if severe_ratio > 0.3:
                        forgery_score = max(forgery_score, max_deviation * 0.5)
                        signals.append(f"Irregular horizontal spacing detected ({severe_count}/{len(valid_spacings)} spacings deviate >30% from grid)")
                    else:
                        signals.append("Horizontal text grid is regular — margins consistent")
                else:
                    signals.append("Insufficient horizontal line spacing for analysis")

            if vertical_lines:
                x_positions = sorted(set([v[0] for v in vertical_lines]))
                spacings = np.diff(x_positions)

                valid_spacings = [s for s in spacings if s >= 5]

                if len(valid_spacings) >= 3:
                    ref_spacing = np.median(valid_spacings)
                    deviations = [abs(s - ref_spacing) / ref_spacing for s in valid_spacings]
                    max_deviation = min(max(deviations), 1.0)

                    severe_count = sum(1 for d in deviations if d > 0.30)
                    severe_ratio = severe_count / len(deviations) if deviations else 0

                    if severe_ratio > 0.3:
                        forgery_score = max(forgery_score, max_deviation * 0.5)
                        signals.append(f"Irregular vertical spacing detected ({severe_count}/{len(valid_spacings)} spacings deviate >30% from grid)")
                    else:
                        signals.append("Vertical column grid is regular")

            # Step 6: Check for rotation/skew
            if horizontal_lines:
                angles = []
                for ln in lines:
                    x1, y1, x2, y2 = ln[0]
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if abs(angle) < 45 or abs(angle - 180) < 45:
                        angles.append(angle)
                if angles:
                    skew = np.std(angles)
                    # Only flag significant skew (> 3 degrees on a real document)
                    if skew > 3.0:
                        forgery_score = max(forgery_score, min(skew / 30.0, 1.0))
                        signals.append(f"Document appears skewed ({skew:.1f}° rotation variance)")

            # Set regions for suspicious areas
            if forgery_score > 0.2 and horizontal_lines:
                mid_y = int(np.median([h[0] for h in horizontal_lines]))
                regions.append(FlaggedRegion(
                    x=0,
                    y=max(0, mid_y - 30),
                    w=w,
                    h=60,
                    reason="Margin irregularity detected"
                ))

            # Confidence calculation
            if len(lines) < 3:
                confidence = 0.50
            elif forgery_score > 0.3:
                confidence = 0.80
            else:
                confidence = 0.75

            if not signals:
                signals = ["Layout structure appears normal"]

            return ModelOutput(
                model_id=self.model_id,
                available=self.available,
                forgery_score=min(float(forgery_score), 1.0),
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
                signals=[f"Layout analysis failed: {str(e)}"]
            )