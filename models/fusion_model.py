import torch
import torch.nn as nn


def block3d(i, o):
    return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.BatchNorm3d(o), nn.ReLU())


def block2d(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())


class AffectAwareFusionModel(nn.Module):
    """video (B,3,16,112,112) + audio (B,1,80,200) -> 3 logits
    (Mental Demand, Effort, Temporal Demand). About 423k parameters."""

    def __init__(self):
        super().__init__()
        self.video_net = nn.Sequential(
            block3d(3, 16), nn.MaxPool3d(2),
            block3d(16, 32), nn.MaxPool3d(2),
            block3d(32, 64), nn.MaxPool3d(2),
            block3d(64, 128), nn.AdaptiveAvgPool3d(1), nn.Flatten())  # -> 128
        self.audio_net = nn.Sequential(
            block2d(1, 16), nn.MaxPool2d(2),
            block2d(16, 32), nn.MaxPool2d(2),
            block2d(32, 64), nn.MaxPool2d(2),
            block2d(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten())  # -> 128
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, video, audio):
        return self.fc(torch.cat([self.video_net(video), self.audio_net(audio)], 1))
