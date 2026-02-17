from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from diffusers.models.transformers.cogvideox_transformer_3d import Transformer2DModelOutput
from diffusers.utils import is_torch_version
from diffusers.loaders import  PeftAdapterMixin
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps, get_3d_sincos_pos_embed
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor2_0
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero, AdaLayerNormZeroSingle
from diffusers.configuration_utils import ConfigMixin, register_to_config




class CogVideoXJointAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        num_anchors: int = 4,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            
            # query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            # if not attn.is_cross_attention:
            #     key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

            single_anchor_length = image_rotary_emb[0].shape[0]
            for i in range(num_anchors):
                query[:, :, text_seq_length + i*single_anchor_length:(text_seq_length + (i+1)*single_anchor_length)] = apply_rotary_emb(query[:, :, text_seq_length + i*single_anchor_length:(text_seq_length + (i+1)*single_anchor_length)], image_rotary_emb)
            if not attn.is_cross_attention:
                for i in range(num_anchors):
                    key[:, :, text_seq_length + i*single_anchor_length:(text_seq_length + (i+1)*single_anchor_length)] = apply_rotary_emb(key[:, :, text_seq_length + i*single_anchor_length:(text_seq_length + (i+1)*single_anchor_length)], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXJointAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_anchors: int = 4,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            num_anchors=num_anchors,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class CogVideoXControlnetPCD(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        use_camera_condition: bool = False,
        attention_head_dim: int = 64,
        vae_channels: int = 16,
        in_channels: int = 3,
        downscale_coef: int = 8,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        num_layers: int = 8,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        out_proj_dim: int = None,
        out_proj_dim_zero_init: bool = False,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )
        
        self.vae_channels = vae_channels
        start_channels = 6 * (downscale_coef ** 2)
        input_channels = [start_channels, start_channels // 2, start_channels // 4]
        self.unshuffle = nn.PixelUnshuffle(downscale_coef)
        self.use_camera_condition = use_camera_condition

        patch_embed_in_channels = vae_channels*2
            
        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=patch_embed_in_channels,
            embed_dim=inner_dim,
            bias=True,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        
        if use_camera_condition:
            # Process camera poses: 49 frames -> 13 frames (first frame unchanged, last 48 -> 12)
            # Input: (batch, 49, 12) -> Output: (batch, 13, 12)
            camera_pose_dim = 12
            hidden_dim = inner_dim // 2
            
            # Time convolution to compress last 48 frames to 12 frames with interaction
            # Use multi-layer conv to allow interaction between groups
            # 48 -> 24 -> 12: each layer allows neighboring frames to interact
            self.camera_time_conv = nn.Sequential(
                # First layer: 48 -> 24, kernel_size=3 allows overlap between groups
                nn.Conv1d(
                    in_channels=camera_pose_dim,
                    out_channels=camera_pose_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1  # padding to maintain proper output size
                ),
                nn.GELU(),
                # Second layer: 24 -> 12, kernel_size=3 allows overlap
                nn.Conv1d(
                    in_channels=camera_pose_dim,
                    out_channels=camera_pose_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.GELU()
            )
            
            # MLP to project 12-dim camera pose to inner_dim for each layer
            self.camera_pose_blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(camera_pose_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, inner_dim)
                    )
                    for _ in range(num_layers)
                ]
            )
            
            self.confidence_net = nn.Sequential(
                nn.Linear(12, inner_dim // 4),  # Input: relative pose (12-dim)
                nn.GELU(),
                nn.Linear(inner_dim // 4, inner_dim // 8),
                nn.GELU(),
                nn.Linear(inner_dim // 8, inner_dim // 16),
                nn.GELU(),
                nn.Linear(inner_dim // 16, 1),  # Output: single confidence score
                nn.Sigmoid()  # Ensure output in [0, 1]
            )
                        

        self.out_projectors = None
        if out_proj_dim is not None:
            self.out_projectors = nn.ModuleList(
                [nn.Linear(inner_dim, out_proj_dim) for _ in range(num_layers)]
            )
            if out_proj_dim_zero_init:
                for out_projector in self.out_projectors:
                    self.zeros_init_linear(out_projector)
        
        # Camera states output projectors (similar to out_projectors)
        self.camera_out_projectors = None
        if use_camera_condition and out_proj_dim is not None:
            self.camera_out_projectors = nn.ModuleList(
                [nn.Linear(inner_dim, out_proj_dim) for _ in range(num_layers)]
            )
            if out_proj_dim_zero_init:
                for camera_out_projector in self.camera_out_projectors:
                    self.zeros_init_linear(camera_out_projector)   
            
        self.gradient_checkpointing = False
    
    def zeros_init_linear(self, linear: nn.Module):
        if isinstance(linear, (nn.Linear, nn.Conv1d)):
            if hasattr(linear, "weight"):
                nn.init.zeros_(linear.weight)
            if hasattr(linear, "bias"):
                nn.init.zeros_(linear.bias)
        
    def _set_gradient_checkpointing(self, enable=False, gradient_checkpointing_func=None):
        self.gradient_checkpointing = enable

    def compress_time(self, x, num_frames):
        x = rearrange(x, '(b f) c h w -> b f c h w', f=num_frames)
        batch_size, frames, channels, height, width = x.shape
        x = rearrange(x, 'b f c h w -> (b h w) c f')
        
        if x.shape[-1] % 2 == 1:
            x_first, x_rest = x[..., 0], x[..., 1:]
            if x_rest.shape[-1] > 0:
                x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)

            x = torch.cat([x_first[..., None], x_rest], dim=-1)
        else:
            x = F.avg_pool1d(x, kernel_size=2, stride=2)
        x = rearrange(x, '(b h w) c f -> (b f) c h w', b=batch_size, h=height, w=width)
        return x
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_states: Tuple[torch.Tensor, torch.Tensor],
        gt_cameras: torch.Tensor,
        context_cameras: torch.Tensor,
        anchor_masks: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        
        # apply anchor masks to controlnet states
        # controlnet_states = controlnet_states * anchor_masks
        
        batch_size, num_anchors, num_frames, channels, height, width = controlnet_states.shape
        
        controlnet_states = rearrange(controlnet_states, 'b n f c h w -> (b n) f c h w')
        hidden_states = hidden_states.repeat(num_anchors, 1, 1, 1, 1)
        
        hidden_states = torch.cat([hidden_states, controlnet_states], dim=2)
        
        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        
        hidden_states = self.patch_embed(encoder_hidden_states.repeat(num_anchors, 1, 1), hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:batch_size, :text_seq_length]
        
        hidden_states = hidden_states[:, text_seq_length:]
        hidden_states = rearrange(hidden_states, '(b n) l c -> b (n l) c', n=num_anchors)
        
        mask_height, mask_width = anchor_masks.shape[-2:]
        
        # gt camera to hidden frames
        gt_cameras = gt_cameras.view(batch_size, (num_frames-1)*4+1, -1).permute(0,2,1)
        gt_cameras_last_frames = self.camera_time_conv(gt_cameras[:,:,1:])
        gt_cameras = torch.cat([gt_cameras[:,:,:1], gt_cameras_last_frames], dim=2)
        gt_cameras = gt_cameras.permute(0,2,1)
        gt_cameras = rearrange(gt_cameras, 'b f c -> b f 1 1 c')
        gt_cameras = gt_cameras.repeat(1, 1, 1, mask_height, mask_width, 1)
        gt_cameras = gt_cameras.view(batch_size, -1, 12)
        
        # compute weighted masks
        context_cameras = context_cameras.view(batch_size, num_anchors, (num_frames-1)*4+1, -1)
        context_cameras_confidence = self.confidence_net(context_cameras)
        
        # pool with 4 frames at a time
        # Reshape to (batch*num_anchors, 1, num_frames) for avg_pool1d
        b, n, f = context_cameras_confidence.shape[:3]
        last_frames = context_cameras_confidence[:,:,1:].reshape(b*n, 1, -1)
        pooled = F.avg_pool1d(last_frames, kernel_size=4, stride=4)
        context_cameras_confidence = torch.cat([
            context_cameras_confidence[:,:,:1], 
            pooled.reshape(b, n, (f-1)//4, -1)
        ], dim=2)
        context_cameras_confidence = rearrange(context_cameras_confidence, 'b n f 1 -> b n f 1 1 1')
        
        context_cameras_confidence = context_cameras_confidence.repeat(1, 1, 1, 1, mask_height, mask_width)
        
        anchor_masks = anchor_masks * context_cameras_confidence
        mask_sum = anchor_masks.sum(1, keepdim=True).clamp(min=1e-8)  # Avoid division by zero
        anchor_masks = anchor_masks / mask_sum # normalize to 0-1
        anchor_masks = rearrange(anchor_masks, 'b n f 1 h w -> b n (f h w) 1', n=num_anchors).to(controlnet_states.dtype) # weights
        
        controlnet_hidden_states = []
        camera_states_list = []
        
        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            # Compute camera states for this layer from original gt_cameras
            if self.use_camera_condition:
                camera_states = self.camera_pose_blocks[i](gt_cameras)
                # Apply camera_out_projector if available
                if self.camera_out_projectors is not None:
                    camera_states = self.camera_out_projectors[i](camera_states)
                camera_states_list.append(camera_states)
            else:
                camera_states_list.append(None)
            
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    num_anchors,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    num_anchors=num_anchors,
                )
                
            # Store hidden_states
            if self.out_projectors is not None:
                if anchor_masks is not None:
                    # controlnet_hidden_states += (self.out_projectors[i](hidden_states) * anchor_masks,)
                    output_hidden_states = rearrange(hidden_states, 'b (n l) c -> b n l c', n=num_anchors)
                    output_hidden_states = output_hidden_states * anchor_masks
                    output_hidden_states = output_hidden_states.sum(dim=1)
                    output_hidden_states = self.out_projectors[i](output_hidden_states)
                    controlnet_hidden_states.append(output_hidden_states)
                else:
                    controlnet_hidden_states.append(self.out_projectors[i](hidden_states))
            else:
                controlnet_hidden_states.append(hidden_states)

        if not return_dict:
            return (controlnet_hidden_states, camera_states_list)
        return Transformer2DModelOutput(sample=(controlnet_hidden_states, camera_states_list))