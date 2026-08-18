"""
Edge Inference Module
======================
Deploys the trained 1D-CNN on edge hardware (Jetson Nano / Raspberry Pi 4)
for real-time, sub-100 ms cyberattack detection.

Key features
  - TorchScript export for deployment without full PyTorch install
  - Sliding-window buffer with configurable step
  - Online StandardScaler that can be updated incrementally
  - Returns structured DetectionResult with class, confidence, latency
"""

import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Re-import the CNN architecture so this module is self-contained
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=k//2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )
    def forward(self, x):
        return self.net(x)


class EdgeCNN1D(nn.Module):
    NUM_CLASSES   = 6
    NUM_FEATURES  = 32

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 64),
            nn.MaxPool1d(2),
            ConvBlock(64, 128),
            nn.MaxPool1d(2),
            ConvBlock(128, 64),
            nn.AdaptiveAvgPool1d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, self.NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x.unsqueeze(1)))


# ---------------------------------------------------------------------------
# Detection result container
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Normal", "FDI", "DoS", "GPS_Spoofing", "Replay", "Meas_Manipulation"]


@dataclass
class DetectionResult:
    predicted_class: int
    class_name:      str
    confidence:      float          # softmax probability of top class
    all_probs:       list           # [float] × NUM_CLASSES
    latency_ms:      float
    attack_detected: bool           # True for any non-zero class
    timestamp:       float          # Unix time of inference


# ---------------------------------------------------------------------------
# Edge inference engine
# ---------------------------------------------------------------------------
class EdgeInferenceEngine:
    """
    Lightweight inference wrapper for the 1D-CNN model.

    Parameters
    ----------
    model_path : str
        Path to the saved model weights (.pt or TorchScript .pts).
    scaler_path : str, optional
        Path to JSON file with mean/std for online normalisation.
    device : str
        "cpu" (default) for edge MCUs; "cuda" when a GPU is available.

    Usage
    -----
    >>> engine = EdgeInferenceEngine("best_model.pt")
    >>> result = engine.predict(feature_vector_32d)
    >>> print(result.class_name, result.latency_ms)
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model  = self._load_model(model_path)
        self.mean, self.std = self._load_scaler(scaler_path)

    # ------------------------------------------------------------------
    def _load_model(self, path: str) -> nn.Module:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        try:
            # Try TorchScript first
            model = torch.jit.load(str(p), map_location=self.device)
        except Exception:
            # Fall back to state-dict loading
            model = EdgeCNN1D().to(self.device)
            state = torch.load(str(p), map_location=self.device)
            model.load_state_dict(state.get("model_state_dict", state))
        model.eval()
        return model

    def _load_scaler(self, path: Optional[str]):
        if path and Path(path).exists():
            with open(path) as f:
                s = json.load(f)
            return (
                np.array(s["mean"], dtype=np.float32),
                np.array(s["std"],  dtype=np.float32),
            )
        # Identity scaler — no normalisation
        return np.zeros(32, dtype=np.float32), np.ones(32, dtype=np.float32)

    # ------------------------------------------------------------------
    def _preprocess(self, x: np.ndarray) -> torch.Tensor:
        x = (x.astype(np.float32) - self.mean) / (self.std + 1e-8)
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    def predict(self, feature_vector: np.ndarray) -> DetectionResult:
        """
        Run inference on a single 32-dimensional feature vector.

        Returns a DetectionResult with class prediction, confidence,
        and wall-clock latency in milliseconds.
        """
        t0 = time.perf_counter()
        x  = self._preprocess(feature_vector)
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        cls  = int(np.argmax(probs))
        return DetectionResult(
            predicted_class = cls,
            class_name      = CLASS_NAMES[cls],
            confidence      = float(probs[cls]),
            all_probs       = probs.tolist(),
            latency_ms      = latency_ms,
            attack_detected = cls != 0,
            timestamp       = time.time(),
        )

    def predict_batch(self, feature_matrix: np.ndarray) -> list:
        """Batch inference — more efficient for offline evaluation."""
        X = np.stack([(row.astype(np.float32) - self.mean) / (self.std + 1e-8)
                      for row in feature_matrix])
        t = torch.tensor(X, dtype=torch.float32).to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            probs = torch.softmax(self.model(t), dim=-1).cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0 / len(X)

        results = []
        for i, p in enumerate(probs):
            cls = int(np.argmax(p))
            results.append(DetectionResult(
                predicted_class = cls,
                class_name      = CLASS_NAMES[cls],
                confidence      = float(p[cls]),
                all_probs       = p.tolist(),
                latency_ms      = latency_ms,
                attack_detected = cls != 0,
                timestamp       = time.time(),
            ))
        return results

    def export_torchscript(self, output_path: str):
        """Export model to TorchScript for bare-metal deployment."""
        dummy = torch.zeros(1, 32).to(self.device)
        scripted = torch.jit.trace(self.model, dummy)
        scripted.save(output_path)
        print(f"TorchScript model saved → {output_path}")
        return output_path


# ---------------------------------------------------------------------------
# Streaming buffer for real-time processing (10 Hz IEC 61850 stream)
# ---------------------------------------------------------------------------
class StreamingDetector:
    """
    Wraps EdgeInferenceEngine in a sliding-window buffer suitable for
    a 10 Hz IEC 61850 data stream.
    """

    def __init__(self, engine: EdgeInferenceEngine, window: int = 1, step: int = 1):
        self.engine  = engine
        self.window  = window
        self.step    = step
        self._buffer: list = []
        self.detections: list = []

    def push(self, feature_vector: np.ndarray) -> Optional[DetectionResult]:
        self._buffer.append(feature_vector)
        if len(self._buffer) >= self.window:
            x = np.mean(self._buffer[-self.window:], axis=0)
            result = self.engine.predict(x)
            self.detections.append(result)
            if len(self._buffer) > self.window * 10:
                self._buffer = self._buffer[-self.window * 2:]
            return result
        return None


# ---------------------------------------------------------------------------
# CLI quick-test (no model file required — uses random weights)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile, os

    # Save a random-weight model for testing
    tmp = tempfile.mktemp(suffix=".pt")
    m = EdgeCNN1D()
    torch.save({"model_state_dict": m.state_dict()}, tmp)

    engine = EdgeInferenceEngine(tmp)
    rng    = np.random.default_rng(99)
    vec    = rng.standard_normal(32).astype(np.float32)

    result = engine.predict(vec)
    print(f"Class: {result.class_name:<22} Conf: {result.confidence:.4f}  "
          f"Latency: {result.latency_ms:.4f} ms  Attack: {result.attack_detected}")
    os.remove(tmp)
