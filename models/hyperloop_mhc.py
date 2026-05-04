"""Hyperloop-mHC Vision Transformer backbone.

This file supports two variants with one shared implementation:

1. Loop-level Hyperloop-mHC
   - Standard begin layers
   - Standard middle layers reused K times
   - Standard end layers
   - Loop-level mHC-style residual stream mixing
   - Scalar/global write gate per stream

2. Spatial-aware Hyperloop-mHC
   - Same as above
   - Replaces scalar/global write gates with token-wise spatial gates
   - Each token/location receives its own adaptive refinement strength

The key design is that hyper-connections are applied at the loop boundary,
not inside every Transformer sub-layer. This is closer to the original
Hyperloop idea than using separate attention/MLP modules for every stream.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out = self.attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=False,
        )[0]
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SpatialLoopGate(nn.Module):
    """Token-wise spatial loop gate."""

    def __init__(self, dim, hidden_ratio=0.25, init_bias=-1.0, dropout=0.0):
        super().__init__()
        hidden_dim = max(16, int(dim * hidden_ratio))
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, x, loop_pos=None):
        if loop_pos is not None:
            x = x + loop_pos
        return torch.sigmoid(self.net(x))


class LoopLevelMHC(nn.Module):
    """Loop-level mHC-style hyper-connection."""

    def __init__(
        self,
        dim,
        n_streams=2,
        sinkhorn_iters=5,
        use_spatial_gate=False,
        gate_hidden_ratio=0.25,
        gate_init_bias=-1.0,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.use_spatial_gate = use_spatial_gate

        self.read_logits = nn.Parameter(torch.zeros(n_streams))

        eye = torch.eye(n_streams)
        self.residual_logits = nn.Parameter(
            eye + 0.01 * torch.randn(n_streams, n_streams)
        )

        if use_spatial_gate:
            self.spatial_gates = nn.ModuleList(
                [
                    SpatialLoopGate(
                        dim=dim,
                        hidden_ratio=gate_hidden_ratio,
                        init_bias=gate_init_bias,
                        dropout=dropout,
                    )
                    for _ in range(n_streams)
                ]
            )
            self.write_logits = None
        else:
            self.write_logits = nn.Parameter(
                torch.full((n_streams,), gate_init_bias)
            )
            self.spatial_gates = None

    def sinkhorn(self, matrix):
        """Project a positive matrix to approximately doubly stochastic form."""
        for _ in range(self.sinkhorn_iters):
            matrix = matrix / (matrix.sum(dim=1, keepdim=True) + 1e-12)
            matrix = matrix / (matrix.sum(dim=0, keepdim=True) + 1e-12)
        return matrix

    def residual_mixing_matrix(self):
        positive = torch.exp(self.residual_logits)
        return self.sinkhorn(positive)

    def read(self, streams):
        weights = torch.softmax(self.read_logits, dim=0)
        x = 0.0
        for i in range(self.n_streams):
            x = x + weights[i] * streams[i]
        return x

    def _get_gate(self, stream_idx, x, loop_pos=None):
        if self.use_spatial_gate:
            return self.spatial_gates[stream_idx](x, loop_pos=loop_pos)
        return torch.sigmoid(self.write_logits[stream_idx])

    def write(self, streams, x, loop_pos=None, return_gates=False):
        M = self.residual_mixing_matrix()
        new_streams = []
        gates = []

        for i in range(self.n_streams):
            mixed_old = 0.0
            for j in range(self.n_streams):
                mixed_old = mixed_old + M[i, j] * streams[j]

            gate = self._get_gate(i, x, loop_pos=loop_pos)
            new_stream = (1.0 - gate) * mixed_old + gate * x
            new_streams.append(new_stream)
            gates.append(gate)

        if return_gates:
            return new_streams, gates
        return new_streams

    def forward(self, streams, middle_block, loop_pos=None, return_gates=False):
        x = self.read(streams)
        if loop_pos is not None:
            x = x + loop_pos

        for block in middle_block:
            x = block(x)

        return self.write(
            streams=streams,
            x=x,
            loop_pos=loop_pos,
            return_gates=return_gates,
        )


class HyperloopBlock(nn.Module):
    """Hyperloop middle block with loop-level mHC."""

    def __init__(
        self,
        num_layers,
        dim,
        num_heads,
        K=3,
        mlp_ratio=4,
        dropout=0.0,
        n_streams=2,
        sinkhorn_iters=5,
        use_spatial_gate=False,
        gate_hidden_ratio=0.25,
        gate_init_bias=-1.0,
    ):
        super().__init__()
        self.K = K
        self.dim = dim
        self.n_streams = n_streams
        self.use_spatial_gate = use_spatial_gate

        self.middle = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.hyper = LoopLevelMHC(
            dim=dim,
            n_streams=n_streams,
            sinkhorn_iters=sinkhorn_iters,
            use_spatial_gate=use_spatial_gate,
            gate_hidden_ratio=gate_hidden_ratio,
            gate_init_bias=gate_init_bias,
            dropout=dropout,
        )

        self.loop_pos_embed = nn.Parameter(torch.zeros(K, 1, dim))
        nn.init.normal_(self.loop_pos_embed, std=0.02)

    def forward(self, streams, return_gates=False):
        all_gates = []

        for k in range(self.K):
            loop_pos = self.loop_pos_embed[k].unsqueeze(0)
            if return_gates:
                streams, gates = self.hyper(
                    streams=streams,
                    middle_block=self.middle,
                    loop_pos=loop_pos,
                    return_gates=True,
                )
                all_gates.append(gates)
            else:
                streams = self.hyper(
                    streams=streams,
                    middle_block=self.middle,
                    loop_pos=loop_pos,
                    return_gates=False,
                )

        if return_gates:
            return streams, all_gates
        return streams


class HyperloopViT(nn.Module):
    """Hyperloop-mHC Vision Transformer backbone."""

    def __init__(
        self,
        in_channels,
        img_size=(32, 64),
        embed_dim=128,
        num_heads=4,
        begin_depth=2,
        middle_depth=4,
        end_depth=2,
        K=3,
        n_streams=2,
        mlp_ratio=4,
        dropout=0.0,
        sinkhorn_iters=5,
        use_spatial_gate=False,
        gate_hidden_ratio=0.25,
        gate_init_bias=-1.0,
    ):
        super().__init__()
        H, W = img_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.n_streams = n_streams
        self.use_spatial_gate = use_spatial_gate

        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=1,
        )
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, H, W) * 0.02)

        self.begin = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(begin_depth)
            ]
        )

        self.middle = HyperloopBlock(
            num_layers=middle_depth,
            dim=embed_dim,
            num_heads=num_heads,
            K=K,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            n_streams=n_streams,
            sinkhorn_iters=sinkhorn_iters,
            use_spatial_gate=use_spatial_gate,
            gate_hidden_ratio=gate_hidden_ratio,
            gate_init_bias=gate_init_bias,
        )

        self.end = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(end_depth)
            ]
        )

        self.norm_out = nn.LayerNorm(embed_dim)
        self.last_gates = None

    def _add_pos_embed(self, x):
        _, _, H, W = x.shape
        if self.pos_embed.shape[-2:] == (H, W):
            pos = self.pos_embed
        else:
            pos = F.interpolate(
                self.pos_embed,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
        return x + pos

    def forward_features(self, x, return_gates=False):
        _, _, H, W = x.shape
        x = self.patch_embed(x)
        x = self._add_pos_embed(x)
        x = x.flatten(2).transpose(1, 2)

        for block in self.begin:
            x = block(x)

        streams = [x.clone() for _ in range(self.n_streams)]
        if return_gates:
            streams, gates = self.middle(streams, return_gates=True)
        else:
            streams = self.middle(streams, return_gates=False)
            gates = None

        x = sum(streams) / len(streams)

        for block in self.end:
            x = block(x)

        x = self.norm_out(x)
        return x, H, W, gates

    def forward(self, x):
        x, H, W, _ = self.forward_features(x, return_gates=False)
        B = x.shape[0]
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        return x

    @torch.no_grad()
    def forward_with_gates(self, x):
        x, H, W, gates = self.forward_features(x, return_gates=True)
        B = x.shape[0]
        output = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)

        gate_maps = []
        for loop_gates in gates:
            stream_maps = []
            for gate in loop_gates:
                if gate.dim() == 0:
                    gate_map = gate.view(1, 1, 1, 1).expand(B, 1, H, W)
                else:
                    gate_map = gate.transpose(1, 2).reshape(B, 1, H, W)
                stream_maps.append(gate_map)
            gate_maps.append(torch.cat(stream_maps, dim=1))

        gate_maps = torch.stack(gate_maps, dim=1)
        self.last_gates = gate_maps
        return output, gate_maps


def hyperloop_vit_tiny(
    in_channels,
    img_size=(32, 64),
    use_spatial_gate=False,
):
    """Tiny Hyperloop-mHC ViT with 8 unique layers and 16 effective passes."""
    return HyperloopViT(
        in_channels=in_channels,
        img_size=img_size,
        embed_dim=128,
        num_heads=4,
        begin_depth=2,
        middle_depth=4,
        end_depth=2,
        K=3,
        n_streams=2,
        mlp_ratio=4,
        dropout=0.0,
        sinkhorn_iters=5,
        use_spatial_gate=use_spatial_gate,
        gate_hidden_ratio=0.25,
        gate_init_bias=-1.0,
    )
