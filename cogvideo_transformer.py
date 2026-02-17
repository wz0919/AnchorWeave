from typing import Any, Dict, Optional, Tuple, Union

import torch
import numpy as np
from diffusers.utils import is_torch_version
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel, Transformer2DModelOutput


class CustomCogVideoXTransformer3DModel(CogVideoXTransformer3DModel):      
    def set_learnable_parameters(self, unfrozen_layers: int = 16):
        for param in self.patch_embed.parameters():
            param.requires_grad = True
                
        for i in range(unfrozen_layers):
            block = self.transformer_blocks[i]
            attn = block.attn1
            for module in  [block.norm2, block.ff]:
                for param in self.patch_embed.parameters():
                    param.requires_grad = True
                    
            for name in ['to_q', 'to_k', 'to_v', 'norm_q', 'norm_k']:
                module = getattr(attn, name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True
                else:
                    print(f"[Warning] {name} not found in attn1 of block {i}")
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        start_frame = None,
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        controlnet_states: torch.Tensor = None,
        controlnet_weights: Optional[Union[float, int, list, np.ndarray, torch.FloatTensor]] = 1.0,
        camera_states_list: Optional[list] = None,
        camera_weights: Optional[Union[float, int, list, np.ndarray, torch.FloatTensor]] = 1.0,
        return_dict: bool = True,
    ):
        batch_size, num_frames, channels, height, width = hidden_states.shape

        if start_frame is not None:
            hidden_states = torch.cat([start_frame, hidden_states], dim=2)
        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            # Add camera_states to hidden_states BEFORE the block
            if (camera_states_list is not None) and (i < len(camera_states_list)) and (camera_states_list[i] is not None):
                camera_block_weight = camera_weights[i] if isinstance(camera_weights, (list, np.ndarray)) or torch.is_tensor(camera_weights) else camera_weights
                hidden_states = hidden_states + camera_states_list[i] * camera_block_weight
            
            if self.gradient_checkpointing:
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
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

            # Add controlnet_states AFTER the block (unlike camera_states which is added BEFORE)
            if (controlnet_states is not None) and (i < len(controlnet_states)):
                controlnet_states_block = controlnet_states[i]
                controlnet_block_weight = 1.0
                if isinstance(controlnet_weights, (list, np.ndarray)) or torch.is_tensor(controlnet_weights):
                    controlnet_block_weight = controlnet_weights[i]
                elif isinstance(controlnet_weights, (float, int)):
                    controlnet_block_weight = controlnet_weights
                
                hidden_states = hidden_states + controlnet_states_block * controlnet_block_weight

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)