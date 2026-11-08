"""Model 3 architecture used by the backend to load Member 2's checkpoint.

This mirrors the MultimodalFusionModel from the Plan of Action (section I, Step 3),
with the syntax error in the plan fixed (the fusion_mlp block closed with `}`).

IMPORTANT: if Member 2 changes the architecture in models/train_multimodal.py,
copy the class here (or import it) so the checkpoint's state_dict keys match.
Layer sizes are read from the checkpoint itself (see inference.py), so changing
only the input dimensions does NOT require editing this file.
"""

import torch
import torch.nn as nn


class MultimodalFusionModel(nn.Module):
    def __init__(self, video_dim: int = 256, audio_dim: int = 128, num_classes: int = 2):
        super().__init__()

        # Video branch (processes video feature vectors)
        self.video_branch = nn.Sequential(
            nn.Linear(video_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        # Audio branch (processes log-mel spectrogram embeddings)
        self.audio_branch = nn.Sequential(
            nn.Linear(audio_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        # Late-fusion MLP classifier
        self.fusion_mlp = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),  # class 1 = High, class 0 = Low
        )

    def forward(self, video_feat: torch.Tensor, audio_feat: torch.Tensor) -> torch.Tensor:
        v_out = self.video_branch(video_feat)
        a_out = self.audio_branch(audio_feat)
        combined = torch.cat((v_out, a_out), dim=1)  # late fusion by concatenation
        return self.fusion_mlp(combined)
