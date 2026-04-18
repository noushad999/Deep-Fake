"""
Baseline Models for Comparison
================================
CNNDetect  — Wang et al., "CNN-generated images are surprisingly easy to spot", CVPR 2020.
             ResNet-50 fine-tuned for binary real/fake classification.
             Standard baseline in all deepfake detection papers.

UnivFD     — Ojha et al., "Towards Universal Fake Image Detection by Exploiting
             CLIP's Potential", CVPR 2023.
             CLIP ViT-L/14 features + linear layer. No fine-tuning of CLIP.
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models
from typing import Tuple, Optional


# -----------------------------------------------------------------------
# CNNDetect baseline (Wang et al. CVPR 2020)
# -----------------------------------------------------------------------

class CNNDetect(nn.Module):
    """
    ResNet-50 pretrained on ImageNet, last FC replaced with binary classifier.
    Exactly matches Wang et al. 2020 architecture.
    Training: blur augmentation + JPEG augmentation during training.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Replace final FC: 2048 → 1 (binary logit)
        backbone.fc = nn.Linear(2048, 1)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        logits = self.backbone(x)   # [B, 1]
        return logits, None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------------------------------------------------------
# UnivFD baseline (Ojha et al. CVPR 2023)
# -----------------------------------------------------------------------

class UnivFD(nn.Module):
    """
    CLIP ViT-L/14 features (frozen) + single linear layer.
    CLIP is NOT fine-tuned — only the linear probe is trained.
    This is the key insight of UnivFD: CLIP features generalise across generators.
    """

    def __init__(self, clip_model: str = 'ViT-L-14', pretrained: str = 'openai'):
        super().__init__()
        try:
            import open_clip
            self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_model, pretrained=pretrained
            )
        except Exception as e:
            raise ImportError(f"open_clip required: pip install open-clip-torch\n{e}")

        # Freeze all CLIP parameters
        for p in self.clip.parameters():
            p.requires_grad = False

        # Linear probe: 768 (ViT-L/14 dim) → 1
        clip_dim = self.clip.visual.output_dim
        self.linear = nn.Linear(clip_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        with torch.no_grad():
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            feats = self.clip.encode_image(x)           # [B, 768]
            feats = feats / feats.norm(dim=-1, keepdim=True)   # L2 normalise

        logits = self.linear(feats.float())             # [B, 1]
        return logits, None

    def count_parameters(self) -> int:
        # Only the linear probe is trainable
        return sum(p.numel() for p in self.linear.parameters())
