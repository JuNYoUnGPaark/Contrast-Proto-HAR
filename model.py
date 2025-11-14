import argparse
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    - Contrast-CrossFormer + CBAM for Sensor-based Human Activity Recognition
    - Author: JunYoungPark and Myung-Kyu Yi
"""

class ChannelAttention1D(nn.Module):
    """
    Channel attention for 1D signals.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.avg_pool(x).squeeze(-1)
        max_out = self.max_pool(x).squeeze(-1)  

        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)

        out = (avg_out + max_out).unsqueeze(-1) 
        scale = self.sigmoid(out)
        return x * scale               

class TemporalAttention1D(nn.Module):
    """
    Temporal attention for 1D signals.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True) 
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        attn_in = torch.cat([avg_out, max_out], dim=1) 
        attn_map = self.conv(attn_in)          
        attn_map = self.sigmoid(attn_map)
        return x * attn_map                       


class CBAM1D(nn.Module):
    """
    CBAM-style attention block for 1D sensor sequences.
    """
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        
        self.channel_att = ChannelAttention1D(channels, reduction=reduction)
        self.temporal_att = TemporalAttention1D(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.temporal_att(x)
        return x

class ContrastCrossFormerBlock(nn.Module):
    """
    Prototype-guided CrossFormer block.
    """
    def __init__(
        self,
        dim: int,
        n_prototypes: int = 6,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        initial_prototypes: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.dim = dim
        self.n_prototypes = n_prototypes
        self.n_heads = n_heads

        self.prototypes = nn.Parameter(torch.randn(n_prototypes, dim))

        if initial_prototypes is not None:
            assert initial_prototypes.shape == self.prototypes.shape, (
                f"Shape mismatch: initial_prototypes {initial_prototypes.shape} "
                f"vs self.prototypes {self.prototypes.shape}"
            )
            self.prototypes.data.copy_(initial_prototypes)
        else:
            nn.init.xavier_uniform_(self.prototypes)

        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm3 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        self.proto_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_proto_features: bool = False,
        skip_cross_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, T, C = x.shape
        attn_weights = None

        if not skip_cross_attention:
            normalized_prototypes = F.normalize(self.prototypes, dim=1, eps=1e-6) 
            prototypes = normalized_prototypes.unsqueeze(0).expand(B, -1, -1)  

            x_norm = self.norm1(x)
            cross_out, attn_weights = self.cross_attn(x_norm, prototypes, prototypes)
            x = x + cross_out  

        x_norm = self.norm2(x)
        self_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self_out

        x = x + self.mlp(self.norm3(x))

        proto_features = None
        if return_proto_features:
            proto_features = x.mean(dim=1) 
            proto_features = self.proto_proj(proto_features)

        return x, proto_features, attn_weights

class ContrastivePrototypeLoss(nn.Module):
    """
    Supervised prototype contrast loss.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,  
        prototypes: torch.Tensor,
        labels: torch.Tensor  
    ) -> torch.Tensor:
        features = F.normalize(features, dim=1, eps=1e-6)
        prototypes = F.normalize(prototypes, dim=1, eps=1e-6)

        logits = torch.matmul(features, prototypes.t()) / self.temperature

        loss = F.cross_entropy(logits, labels)
        return loss

class ContrastCrossFormerCBAM_HAR(nn.Module):
    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        embed_dim: int,
        reduced_dim: int,
        n_classes: int,
        n_prototypes: int,
        n_heads: int,
        kernel_size: int,
        dropout: float,
        temperature: float,
        initial_prototypes: Optional[torch.Tensor],
        use_cbam: bool,
        use_crossformer: bool,
        use_contrast: bool,
        use_dim_reduction: bool,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.reduced_dim = reduced_dim
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes
        self.n_heads = n_heads
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.temperature = temperature

        self.use_cbam = use_cbam
        self.use_crossformer = use_crossformer
        self.use_contrast = use_contrast
        self.use_dim_reduction = use_dim_reduction

        self.embedding = nn.Sequential(
            nn.Conv1d(
                in_channels,
                embed_dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2, 
            ),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if self.use_cbam:
            self.cbam = CBAM1D(
                channels=embed_dim,
                reduction=8,
                kernel_size=kernel_size,
            )

        working_dim = reduced_dim if use_dim_reduction else embed_dim
        if self.use_dim_reduction:
            self.dim_reduce = nn.Linear(embed_dim, reduced_dim)

        if self.use_crossformer:
            self.crossformer = ContrastCrossFormerBlock(
                dim=working_dim,
                n_prototypes=n_prototypes,
                n_heads=n_heads,
                mlp_ratio=2.0,
                dropout=dropout,
                initial_prototypes=initial_prototypes,
            )
        else:
            self.self_attn = nn.TransformerEncoderLayer(
                d_model=working_dim,
                nhead=n_heads,
                dim_feedforward=int(working_dim * 2),
                dropout=dropout,
                batch_first=True,
            )

        if self.use_dim_reduction:
            self.dim_restore = nn.Linear(reduced_dim, embed_dim)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

        if self.use_contrast and self.use_crossformer:
            self.contrast_loss = ContrastivePrototypeLoss(temperature=temperature)

    def forward(
        self,
        x: torch.Tensor,           
        labels: Optional[torch.Tensor] = None,
        return_contrast_loss: bool = False,
    ):

        x = self.embedding(x)        

        if self.use_cbam:
            x = self.cbam(x)         

        x = x.transpose(1, 2).contiguous()   

        if self.use_dim_reduction:
            x = self.dim_reduce(x)   

        proto_features = None
        if self.use_crossformer:
            if return_contrast_loss and self.use_contrast:
                x, proto_features, _ = self.crossformer(
                    x,
                    return_proto_features=True,
                    skip_cross_attention=False,
                )
            else:
                x, _, _ = self.crossformer(
                    x,
                    return_proto_features=False,
                    skip_cross_attention=False,
                )
        else:
            x = self.self_attn(x)      

        if self.use_dim_reduction:
            x = self.dim_restore(x)   

        x = x.transpose(1, 2).contiguous()  
        feat_vec = self.pool(x).squeeze(-1)  

        logits = self.classifier(feat_vec)  

        if (
            return_contrast_loss
            and self.use_contrast
            and self.use_crossformer
            and proto_features is not None
            and labels is not None
        ):
            contrast_loss = self.contrast_loss(
                proto_features,       
                self.crossformer.prototypes,  
                labels,                 
            )
            return logits, contrast_loss

        return logits

def parse_model_args():
    parser = argparse.ArgumentParser(description="Contrast-CrossFormer + CBAM HAR Model")

    parser.add_argument('--in_channels', type=int, default=9,
                        help='Number of input sensor channels.')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Sequence length (timesteps).')
    parser.add_argument('--n_classes', type=int, default=6,
                        help='Number of target classes.')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Embedding dimension after Conv1d.')
    parser.add_argument('--reduced_dim', type=int, default=32,
                        help='Reduced dimension before attention (if enabled).')
    parser.add_argument('--n_prototypes', type=int, default=6,
                        help='Number of class prototypes.')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads.')
    parser.add_argument('--kernel_size', type=int, default=11,
                        help='Kernel size for Conv1d and temporal attention.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate.')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for prototype contrast loss.')
    parser.add_argument('--disable_cbam', action='store_true',
                        help='Disable CBAM attention.')
    parser.add_argument('--disable_crossformer', action='store_true',
                        help='Disable CrossFormer (use TransformerEncoder instead).')
    parser.add_argument('--disable_contrast', action='store_true',
                        help='Disable contrastive prototype loss.')
    parser.add_argument('--use_dim_reduction', action='store_true',
                        help='Enable dimension reduction before attention.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_model_args()

    use_cbam = not args.disable_cbam
    use_crossformer = not args.disable_crossformer
    use_contrast = not args.disable_contrast

    model = ContrastCrossFormerCBAM_HAR(
        in_channels=args.in_channels,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        reduced_dim=args.reduced_dim,
        n_classes=args.n_classes,
        n_prototypes=args.n_prototypes,
        n_heads=args.n_heads,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        temperature=args.temperature,
        initial_prototypes=None,     
        use_cbam=use_cbam,
        use_crossformer=use_crossformer,
        use_contrast=use_contrast,
        use_dim_reduction=args.use_dim_reduction,
    )

    print(model)

    dummy_x = torch.randn(4, args.in_channels, args.seq_len)  
    dummy_labels = torch.randint(0, args.n_classes, (4,))

    model.train()
    logits, con_loss = model(dummy_x, labels=dummy_labels, return_contrast_loss=True)
    ce_loss = F.cross_entropy(logits, dummy_labels)
    total_loss = ce_loss + con_loss

    print(f"\nLogits shape: {logits.shape}")
    print(f"CE loss: {ce_loss.item():.4f}")
    print(f"Contrast loss: {con_loss.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")



