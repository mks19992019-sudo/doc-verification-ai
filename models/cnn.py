import os
import ssl
import tempfile
import shutil
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import httpx

from models.base import BaseDetectionModel
from schemas.schemas import ModelOutput


def _download_resnet50_weights() -> str:
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
    cached_path = os.path.join(cache_dir, "resnet50-0676ba61.pth")
    if os.path.exists(cached_path):
        return cached_path

    os.makedirs(cache_dir, exist_ok=True)
    url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
    print(f"[CNN] Downloading ResNet-50 weights from {url} ...")
    with httpx.stream("GET", url, timeout=60.0, follow_redirects=True) as resp:
        resp.raise_for_status()
        tmp_path = cached_path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=8192):
                f.write(chunk)
    os.rename(tmp_path, cached_path)
    print(f"[CNN] Downloaded to {cached_path}")
    return cached_path


class CNNModel(BaseDetectionModel):
    model_id = "CNN"

    def __init__(self, weights_path: str = "weights/resnet50_forgery.pt"):
        self._available = False
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._weights_path = weights_path
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._load_model()

    def _load_model(self):
        try:
            weights_file = None

            # 1. Try local fine-tune weights first
            if os.path.exists(self._weights_path):
                weights_file = self._weights_path
                print(f"[CNN] Found fine-tuned weights: {weights_file}")

            # 2. Try downloading ResNet-50 ImageNet weights
            if weights_file is None:
                try:
                    weights_file = _download_resnet50_weights()
                    print(f"[CNN] Loaded ImageNet weights from cache: {weights_file}")
                except Exception as e:
                    print(f"[CNN] Could not download ImageNet weights: {e}")
                    weights_file = None

            # 3. Build model
            if weights_file:
                self._model = models.resnet50(weights=None)
                state_dict = torch.load(weights_file, map_location=self._device, weights_only=True)
                self._model.load_state_dict(state_dict)
            else:
                self._model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

            # Replace classification head
            num_features = self._model.fc.in_features
            self._model.fc = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(num_features, 1),
            )
            self._model = self._model.to(self._device)
            self._model.eval()

            # 4. Load fine-tune weights on top if available
            if os.path.exists(self._weights_path):
                state_dict = torch.load(self._weights_path, map_location=self._device, weights_only=True)
                self._model.load_state_dict(state_dict)
                print(f"[CNN] Fine-tuned weights loaded from {self._weights_path}")

            self._available = True
        except Exception as e:
            print(f"[CNN] Failed to load model: {e}")
            self._available = False
            self._model = None

    @property
    def available(self) -> bool:
        return self._available

    def run(self, image_path: str, text_map: dict) -> ModelOutput:
        if not self.available or self._model is None:
            return ModelOutput(
                model_id=self.model_id,
                available=False,
                forgery_score=0.0,
                confidence=0.0,
                regions=[],
                signals=["CNN model not loaded"]
            )

        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self._transform(img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logit = self._model(img_tensor)
                prob = torch.sigmoid(logit).item()
                confidence = abs(prob - 0.5) * 2

            forgery_score = float(prob)
            signals = self._build_signals(forgery_score, confidence)

            return ModelOutput(
                model_id=self.model_id,
                available=True,
                forgery_score=forgery_score,
                confidence=confidence,
                regions=[],
                signals=signals
            )
        except Exception as e:
            return ModelOutput(
                model_id=self.model_id,
                available=False,
                forgery_score=0.0,
                confidence=0.0,
                regions=[],
                signals=[f"CNN inference error: {e}"]
            )

    def _build_signals(self, score: float, conf: float) -> list:
        signals = []
        if score >= 0.7:
            signals.append(f"CNN confidence {conf:.2f}: High forgery probability detected")
        elif score >= 0.5:
            signals.append(f"CNN confidence {conf:.2f}: Moderate anomaly detected")
        elif score >= 0.3:
            signals.append(f"CNN confidence {conf:.2f}: Minor anomalies detected")
        else:
            signals.append(f"CNN confidence {conf:.2f}: No significant forgery patterns detected")
        return signals
