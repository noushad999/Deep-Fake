"""
Multi-Level Adaptive Fusion (MLAF) Module
Attention-weighted fusion of spatial (128), frequency (64), and semantic (384) streams.
Output: Binary classification logit (Real/Fake)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class StreamAttention(nn.Module):
    """
    Cross-stream attention weighting mechanism.
    Learns to weight each stream's contribution adaptively.
    """
    
    def __init__(self, total_dim: int = 576, n_heads: int = 4):
        super(StreamAttention, self).__init__()
        
        # Query, Key, Value projections for multi-head attention
        self.query = nn.Linear(total_dim, total_dim, bias=False)
        self.key = nn.Linear(total_dim, total_dim, bias=False)
        self.value = nn.Linear(total_dim, total_dim, bias=False)
        
        self.n_heads = n_heads
        self.head_dim = total_dim // n_heads
        
        assert total_dim % n_heads == 0, f"total_dim must be divisible by n_heads"
        
        self.out_proj = nn.Linear(total_dim, total_dim)
        self.layer_norm = nn.LayerNorm(total_dim)
        self.dropout = nn.Dropout(0.1)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, total_dim]
        
        Returns:
            Attended features [B, total_dim]
        """
        # Expand to sequence dimension for attention
        x_seq = x.unsqueeze(1)  # [B, 1, total_dim]
        
        B, seq_len, total_dim = x_seq.shape
        
        # Multi-head attention
        Q = self.query(x_seq).view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(x_seq).view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(x_seq).view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, total_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        # Residual connection + layer norm
        output = self.layer_norm(x_seq + output).squeeze(1)  # [B, total_dim]
        
        return output


class MLAFFusion(nn.Module):
    """
    Multi-Level Adaptive Fusion module.
    
    Architecture:
        1. Concatenate: [spatial(128) + freq(64) + semantic(384)] = 576-dim
        2. Attention weighting
        3. Dense layer + Dropout
        4. Binary classification output
    
    Input:  Three feature vectors [B, 128], [B, 64], [B, 384]
    Output: Classification logit [B, 1] + fused features [B, 256]
    """
    
    def __init__(
        self,
        spatial_dim: int = 128,
        freq_dim: int = 64,
        semantic_dim: int = 384,
        hidden_dim: int = 256,
        attention_heads: int = 4,
        dropout_rate: float = 0.4
    ):
        super(MLAFFusion, self).__init__()
        
        self.combined_dim = spatial_dim + freq_dim + semantic_dim  # 576
        
        # Attention weighting
        self.attention = StreamAttention(self.combined_dim, attention_heads)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Feature projection for intermediate representation
        self.feature_proj = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        spatial_feat: torch.Tensor,
        freq_feat: torch.Tensor,
        semantic_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial_feat: [B, 128] from NPR Branch
            freq_feat:    [B, 64]  from FreqBlender
            semantic_feat: [B, 384] from FAT-Lite
        
        Returns:
            logits: [B, 1] binary classification
            fused_features: [B, 256] intermediate fused representation
        """
        # Verify input dimensions
        assert spatial_feat.shape[1] == 128, f"Spatial dim must be 128, got {spatial_feat.shape[1]}"
        assert freq_feat.shape[1] == 64, f"Freq dim must be 64, got {freq_feat.shape[1]}"
        assert semantic_feat.shape[1] == 384, f"Semantic dim must be 384, got {semantic_feat.shape[1]}"
        
        # Concatenate features
        combined = torch.cat([spatial_feat, freq_feat, semantic_feat], dim=1)  # [B, 576]
        
        # Attention weighting
        attended = self.attention(combined)  # [B, 576]
        
        # Project to hidden dim for intermediate features
        fused_features = self.feature_proj(attended)  # [B, 256]
        
        # Classification
        logits = self.classifier(attended)  # [B, 1]
        
        return logits, fused_features


if __name__ == "__main__":
    print("="*60)
    print("TESTING: MLAF Fusion Module")
    print("="*60)
    
    model = MLAFFusion(
        spatial_dim=128,
        freq_dim=64,
        semantic_dim=384,
        hidden_dim=256,
        attention_heads=4
    )
    model.eval()
    
    # Dummy inputs from each stream
    spatial = torch.randn(4, 128)
    freq = torch.randn(4, 64)
    semantic = torch.randn(4, 384)
    
    with torch.no_grad():
        logits, fused = model(spatial, freq, semantic)
    
    print(f"Spatial input:  {spatial.shape}")
    print(f"Freq input:     {freq.shape}")
    print(f"Semantic input: {semantic.shape}")
    print(f"Logits output:  {logits.shape}")
    print(f"Fused features: {fused.shape}")
    
    assert logits.shape == (4, 1), f"Logits shape mismatch! Expected (4, 1), got {logits.shape}"
    assert fused.shape == (4, 256), f"Fused shape mismatch! Expected (4, 256), got {fused.shape}"
    
    print("\nMLAF Fusion test PASSED!")
