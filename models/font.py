from models.base import BaseDetectionModel
from schemas.schemas import ModelOutput, FlaggedRegion


class FontModel(BaseDetectionModel):
    model_id = "FONT"

    def __init__(self, outlier_threshold: float = 2.0):
        """
        outlier_threshold: number of standard deviations to flag as outlier
        """
        self.outlier_threshold = outlier_threshold
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def run(self, image_path: str, text_map: dict) -> ModelOutput:
        try:
            # Check if we have OCR data from preprocessor
            if text_map and text_map.get("characters"):
                return self._analyze_from_text_map(text_map)

            # If it's a PDF, try pdfminer
            if image_path.lower().endswith(".pdf"):
                return self._analyze_pdf(image_path)

            # For images without OCR data yet (Phase 2 will add EasyOCR)
            return ModelOutput(
                model_id=self.model_id,
                available=self.available,
                forgery_score=0.0,
                confidence=0.50,
                regions=[],
                signals=["No text extraction available yet — OCR will be added in Phase 2"]
            )

        except Exception as e:
            return ModelOutput(
                model_id=self.model_id,
                available=False,
                forgery_score=0.0,
                confidence=0.0,
                regions=[],
                signals=[f"Font analysis failed: {str(e)}"]
            )

    def _analyze_from_text_map(self, text_map: dict) -> ModelOutput:
        """Analyze font sizes from OCR-generated character data."""
        characters = text_map.get("characters", [])
        if len(characters) < 5:
            return ModelOutput(
                model_id=self.model_id,
                available=self.available,
                forgery_score=0.0,
                confidence=0.50,
                regions=[],
                signals=["Too few characters to analyze font consistency"]
            )

        # Extract font sizes (height of each character bounding box)
        font_sizes = []
        for char in characters:
            if isinstance(char, dict) and "height" in char:
                font_sizes.append(char["height"])

        if len(font_sizes) < 5:
            return ModelOutput(
                model_id=self.model_id,
                available=self.available,
                forgery_score=0.0,
                confidence=0.50,
                regions=[],
                signals=["Insufficient character data for font analysis"]
            )

        return self._detect_outliers(font_sizes, characters)

    def _analyze_pdf(self, pdf_path: str) -> ModelOutput:
        """Extract text from PDF using pdfminer and analyze font sizes."""
        try:
            from pdfminer.high_level import extract_text
            from pdfminer.layout import LTChar, LTTextBox
            from pdfminer.pdfpage import PDFPage
            import io
        except ImportError:
            return ModelOutput(
                model_id=self.model_id,
                available=False,
                forgery_score=0.0,
                confidence=0.0,
                regions=[],
                signals=["pdfminer not installed — run: pip install pdfminer.six"]
            )

        font_sizes = []
        characters_data = []

        with open(pdf_path, "rb") as f:
            for page in PDFPage.get_pages(f):
                # We need to access page.resources to get font info
                # pdfminer.six approach: extract_text with layout mode
                from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
                from pdfminer.converter import PDFPageAggregator
                from pdfminer.layout import LAParams, LTTextBox, LTChar

                resource_manager = PDFResourceManager()
                laparams = LAParams()
                device = PDFPageAggregator(resource_manager, laparams=laparams)
                interpreter = PDFPageInterpreter(resource_manager, device)

                # Process page
                interpreter.process_page(page)
                layout = device.get_result()

                # Walk the layout tree
                for element in layout:
                    if isinstance(element, LTTextBox):
                        for line in element:
                            for char in line:
                                if isinstance(char, LTChar):
                                    font_sizes.append(char.height)
                                    characters_data.append({
                                        "height": char.height,
                                        "x": char.x0,
                                        "y": char.y0,
                                        "text": char.get_text()
                                    })

        if len(font_sizes) < 5:
            return ModelOutput(
                model_id=self.model_id,
                available=self.available,
                forgery_score=0.0,
                confidence=0.50,
                regions=[],
                signals=[f"Extracted {len(font_sizes)} characters — need at least 5 for analysis"]
            )

        return self._detect_outliers(font_sizes, characters_data)

    def _detect_outliers(self, font_sizes: list, characters_data: list) -> ModelOutput:
        """Statistical outlier detection on font sizes."""
        import statistics

        if len(font_sizes) < 2:
            return ModelOutput(
                model_id=self.model_id,
                available=self.available,
                forgery_score=0.0,
                confidence=0.50,
                regions=[],
                signals=["Insufficient data for font analysis"]
            )

        mean_size = statistics.mean(font_sizes)
        stdev_size = statistics.stdev(font_sizes) if len(font_sizes) > 1 else 0

        # Find outliers (characters whose font size is > threshold std devs from mean)
        outlier_indices = []
        outlier_sizes = []

        for i, size in enumerate(font_sizes):
            if stdev_size > 0:
                z_score = abs(size - mean_size) / stdev_size
                if z_score > self.outlier_threshold:
                    outlier_indices.append(i)
                    outlier_sizes.append(size)

        # Forgery score based on proportion of outliers
        outlier_ratio = len(outlier_indices) / len(font_sizes)
        forgery_score = min(outlier_ratio * 10, 1.0)  # Scale up since genuine docs might have some variation

        # Confidence based on sample size and signal strength
        if len(font_sizes) < 20:
            confidence = 0.60
        elif forgery_score > 0.5:
            confidence = 0.85
        else:
            confidence = 0.75

        signals = []
        regions = []

        if len(outlier_indices) == 0:
            signals.append(f"Font size variance within normal range (mean: {mean_size:.1f}, std: {stdev_size:.2f})")
        else:
            outlier_sizes_rounded = [round(s, 1) for s in set(outlier_sizes)]
            signals.append(f"Found {len(outlier_indices)} characters with anomalous font sizes: {outlier_sizes_rounded}")

            # Flag regions with outlier characters
            if characters_data and outlier_indices:
                # Group nearby outliers into regions
                for idx in outlier_indices[:5]:  # Flag up to 5 regions
                    if idx < len(characters_data):
                        char = characters_data[idx]
                        if isinstance(char, dict):
                            regions.append(FlaggedRegion(
                                x=int(char.get("x", 0)),
                                y=int(char.get("y", 0)),
                                w=20,
                                h=20,
                                reason=f"Suspicious font size: {char.get('height', 0):.1f} (expected ~{mean_size:.1f})"
                            ))

        return ModelOutput(
            model_id=self.model_id,
            available=self.available,
            forgery_score=float(forgery_score),
            confidence=confidence,
            regions=regions[:5],  # Max 5 regions
            signals=signals
        )