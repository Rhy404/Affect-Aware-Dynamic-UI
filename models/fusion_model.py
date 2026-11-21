import torch
import torch.nn as nn


def block3d(i, o):
    return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.BatchNorm3d(o), nn.ReLU())


def block2d(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())


class AffectAwareFusionModel(nn.Module):
    """video (B,3,T,H,W) [+ audio (B,1,F,T)] -> 3 logits
    (Mental Demand, Effort, Temporal Demand).

    use_audio=True  -> about 423k parameters (same as before)
    use_audio=False -> video-only, about 309k parameters; `audio` is ignored.
    Works for any H, W, and for T >= 8 (global pooling at the end).
    """

    def __init__(self, use_audio=True):
        super().__init__()
        self.use_audio = use_audio
        self.video_net = nn.Sequential(
            block3d(3, 16), nn.MaxPool3d(2),
            block3d(16, 32), nn.MaxPool3d(2),
            block3d(32, 64), nn.MaxPool3d(2),
            block3d(64, 128), nn.AdaptiveAvgPool3d(1), nn.Flatten())  # -> 128
        if use_audio:
            self.audio_net = nn.Sequential(
                block2d(1, 16), nn.MaxPool2d(2),
                block2d(16, 32), nn.MaxPool2d(2),
                block2d(32, 64), nn.MaxPool2d(2),
                block2d(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten())  # -> 128
        fc_in = 256 if use_audio else 128
        self.fc = nn.Sequential(nn.Linear(fc_in, 128), nn.ReLU(),
                                nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, video, audio=None):
        feats = [self.video_net(video)]
        if self.use_audio:
            feats.append(self.audio_net(audio))
        return self.fc(torch.cat(feats, 1))
