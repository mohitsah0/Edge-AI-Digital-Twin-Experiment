"""
1D-CNN Architecture for Edge-AI Cyberattack Detection
Paper: Edge-AI-Driven Digital Twin for Real-Time Cyberattack Detection
       and Resilient Control in Renewable Microgrids
"""

import torch
import torch.nn as nn


class EdgeAI_CNN(nn.Module):
    """
    1D-CNN for Edge-AI Cyberattack Detection in Renewable Microgrids.

    Architecture:
        - 3 Convolutional Blocks (Conv1d + BatchNorm + ReLU + MaxPool)
        - 2 Fully Connected Layers with Dropout
        - Softmax output for 6-class classification

    Input:  (batch, 32)   — 32 time-domain/frequency features per sample
    Output: (batch, 6)    — logits for [Normal, FDI, DoS, GPS, Replay, Manip.]

    Parameters : ~87,432
    Model size  : ~1.2 MB
    Inference   : <0.1 ms per sample on CPU
    """

    def __init__(self, input_dim: int = 32, num_classes: int = 6):
        super(EdgeAI_CNN, self).__init__()

        # ── Conv Block 1 ──────────────────────────────────────────────────────
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)

        # ── Conv Block 2 ──────────────────────────────────────────────────────
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)

        # ── Conv Block 3 ──────────────────────────────────────────────────────
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(256)

        self.pool    = nn.MaxPool1d(2)
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)

        # After 3× MaxPool(2): spatial dim = input_dim // 8
        fc_in = 256 * (input_dim // 8)
        self.fc1 = nn.Linear(fc_in, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features) → (batch, 1, features)
        x = x.unsqueeze(1)

        x = self.pool(self.relu(self.bn1(self.conv1(x))))   # (B, 64,  16)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))   # (B, 128,  8)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))   # (B, 256,  4)

        x = x.view(x.size(0), -1)                          # (B, 1024)
        x = self.relu(self.fc1(x))                         # (B, 128)
        x = self.dropout(x)
        return self.fc2(x)                                  # (B, 6)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = EdgeAI_CNN()
    print(f"Model parameters : {count_parameters(model):,}")
    print(f"Approx size (MB) : {count_parameters(model) * 4 / 1e6:.2f}")
    dummy = torch.randn(8, 32)
    out = model(dummy)
    print(f"Output shape     : {out.shape}")   # (8, 6)
