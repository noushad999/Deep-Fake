"""
Semantic Domain Stream - FAT-Lite (Frequency-Aware Transformer Lite)
Lightweight Vision Transformer for global structural inconsistency detection.
Output: EXACTLY 384-dimensional feature vector.
"""
import torch
import torch.nn as nn
import timm
from typing import Tuple


class FATLiteTransformer(nn.Module):
    """
    Semantic stream using ViT-Tiny for catching global structural anomalies.
    Particularly effective against Diffusion model-generated images.
    
    Architecture:
        Input: [B, 3, 256, 256]
        Backbone: ViT-Tiny-Patch16 (pretrained)
        Projection: Linear -> 384-dim
        Output: [B, 384]
    """
    
    def __init__(
        self,
        output_dim: int = 384,
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        super(FATLiteTransformer, self).__init__()
        
        assert output_dim == 384, f"FAT-Lite output must be 384, got {output_dim}"
        
        # ViT-Tiny: 12 layers, 192 embed dim, 3 heads (very lightweight)
        self.backbone = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            global_pool='token'
        )
        
        backbone_feat_dim = self.backbone.num_features  # 192 for vit_tiny
        
        # Project to 384-dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_feat_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 3, 256, 256]
        
        Returns:
            Feature tensor [B, 384]
        """
        # ViT expects 224x224, so we resize via interpolation
        if x.shape[-1] != 224:
            x = nn.functional.interpolate(x, size=224, mode='bilinear', align_corners=False)
        
        # Semantic feature extraction
        features = self.backbone(x)  # [B, 192]
        
        # Project to 384-dim
        output = self.projection(features)  # [B, 384]
        
        return output
    
    def get_feature_dim(self) -> int:
        return 384


if __name__ == "__main__":
    print("="*60)
    print("TESTING: FAT-Lite Transformer (Semantic Stream)")
    print("="*60)
    
    model = FATLiteTransformer(output_dim=384, pretrained=False)
    model.eval()
    
    x = torch.randn(4, 3, 256, 256)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dim:   {output.shape[1]}")
    
    assert output.shape == (4, 384), f"Dimension mismatch! Expected (4, 384), got {output.shape}"
    print("\nFAT-Lite test PASSED!")
