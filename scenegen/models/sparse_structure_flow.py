from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from ..modules.spatial import patchify, unpatchify
from .position_head import PositionHead


class BatchEncoder(nn.Module):
    def __init__(self, max_batch_size=32, embed_dim=1024, init_scale=5.0):
        super().__init__()
        self.batch_embeddings = nn.Parameter(
            torch.randn(max_batch_size, 1, embed_dim) * init_scale
        )
    
    def forward(self, batch_indices):
        batch_indices = batch_indices.clamp(max=self.batch_embeddings.shape[0]-1)
        embeddings = self.batch_embeddings[batch_indices.squeeze(-1)]
        return embeddings


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        use_global: bool = False,
        num_register_tokens: int = 4,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quatR_S",
        num_iteration: int = 4,
        use_batch_encoder: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint  # whether to use checkpointing in transformer blocks
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.use_global = use_global
        self.num_register_tokens = num_register_tokens
        self.trunk_depth = trunk_depth
        self.pose_encoding_type = pose_encoding_type
        self.num_iteration = num_iteration
        self.use_batch_encoder = use_batch_encoder
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution // patch_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)


        if self.use_global:
            if self.num_register_tokens <= 0:
                raise ValueError("num_register_tokens must be greater than 0 when use_global is True")
            self.position_token = nn.Parameter(torch.randn(2, 1, model_channels))
            self.register_token = nn.Parameter(torch.randn(2, self.num_register_tokens, model_channels))
            self.patch_start_index = 1 + self.num_register_tokens
            self.position_head = PositionHead(
                model_channels=self.model_channels,
                trunk_depth=self.trunk_depth,
                pose_encoding_type=self.pose_encoding_type,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                use_checkpoint=self.use_checkpoint,
                use_fp16=use_fp16,
            )
            if self.use_batch_encoder:
                self.batch_encoder = BatchEncoder(
                    max_batch_size=16,
                    embed_dim=model_channels,
                    init_scale=3.0
                )
            nn.init.normal_(self.position_token, std=1)
            nn.init.normal_(self.register_token, std=1)

        
        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                use_global=self.use_global,
                num_register_tokens=self.num_register_tokens,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        # self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)
        
        # position head weights are not converted to fp16
        if self.use_global:
            self.position_head.trunk.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)
        if self.use_global:
            self.position_head.trunk.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"
        if not self.use_global:
            h = patchify(x, self.patch_size)
            h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
            h = self.input_layer(h)
            h = h + self.pos_emb[None]
            t_emb = self.t_embedder(t)
            if self.share_mod:
                t_emb = self.adaLN_modulation(t_emb)
            t_emb = t_emb.type(self.dtype)
            h = h.type(self.dtype)
            cond = cond.type(self.dtype)
            for block in self.blocks:
                h = block(h, t_emb, cond)
            h = h.type(x.dtype)
            h = F.layer_norm(h, h.shape[-1:])
            h = self.out_layer(h)
            h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
            h = unpatchify(h, self.patch_size).contiguous()
            return h
        else:
            h = patchify(x, self.patch_size)
            h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

            B, N, C = h.shape

            h = self.input_layer(h)
            h = h + self.pos_emb[None]

            # Concatenate the camera token and register token to the input
            position_token = self.slice_expand(self.position_token, B)
            register_token = self.slice_expand(self.register_token, B)
            h = torch.cat([position_token, register_token, h], dim=1)

            t_emb = self.t_embedder(t)
            if self.share_mod:
                t_emb = self.adaLN_modulation(t_emb)
            t_emb = t_emb.type(self.dtype)
            h = h.type(self.dtype)
            cond = cond.type(self.dtype)
            for block in self.blocks:
                h = block(h, t_emb, cond)
            h = h.type(x.dtype)
            position_token = h[:, :1, :]
            h = h[:, self.num_register_tokens + 1:, :]

            positions = self.position_head(
                position_token,
                num_iteration=self.num_iteration,
            )

            h = F.layer_norm(h, h.shape[-1:])
            h = self.out_layer(h)

            # Detach the camera token and register token
            h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
            h = unpatchify(h, self.patch_size).contiguous()

            return h, positions

    def slice_expand(self, token_tensor, B):
        query = token_tensor[0:1, ...]
        others = token_tensor[1:, ...].repeat(B - 1, 1, 1)
        # concatenate the query and others along the batch dimension: [B, ..., C]
        # batch_indices = torch.arange(1, B).view(B-1, 1, 1).to(token_tensor.device)
        # if not self.use_batch_encoder:
        #     index_coding = batch_indices * 1
        # else:
        #     index_coding = self.batch_encoder(batch_indices)
        # others = others + index_coding

        combined = torch.cat([query, others], dim=0)
        return combined
