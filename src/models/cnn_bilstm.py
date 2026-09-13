"""CNN-BiLSTM delineation model (Section 3.3, Fig. 1 of the paper).

Two 1D convolutional blocks (32 filters/kernel 7, then 64 filters/kernel 5,
each with BatchNorm + ReLU) feed a single-layer bidirectional LSTM (128 hidden
units per direction, 256-dim output), followed by dropout (0.3) and a linear
classifier over the three waveform classes {Background, P, T}.
"""
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.features(x)


class BiLSTMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(64, 128, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.lstm(x)[0]


class RPeakGuidedML2(nn.Module):
    """Baseline / A0-A2 model: single-channel R-centered (or fixed-window) input."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.cnn = CNNFeatureExtractor(channels=1)
        self.bilstm = BiLSTMBlock()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x).permute(0, 2, 1)
        return self.classifier(self.dropout(self.bilstm(x)))


class RPeakTimeML2(nn.Module):
    """A3/A4 (and R6) variant: adds the normalized R-relative time channel."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.cnn = CNNFeatureExtractor(channels=2)
        self.bilstm = BiLSTMBlock()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x).permute(0, 2, 1)
        return self.classifier(self.dropout(self.bilstm(x)))
