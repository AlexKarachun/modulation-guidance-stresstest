import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from torchvision import transforms


########################################################################################################################
#                     COSMOS TRANSFORMER WITH AN ADDITIONAL EMBEDDING USING POOLED CLIP ONE                            #
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
class FP32SiLU(nn.Module):
    r"""
    SiLU activation function with input upcasted to torch.float32.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs.float(), inplace=False).to(inputs.dtype)


class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class CosmosWithPooledCLIP(nn.Module):
    def __init__(self,
                 transformer: nn.Module,
                 pooled_projection_dim: int = 768,
                 w: int = 3,
                 start_layer: int = 0,
                 ):
        """
        :param transformer: Basic COSMOS transformer model without pooled CLIP embedding
        """
        super().__init__()
        self.transformer = transformer
        self.transformer.requires_grad_(False)  # Double check for no grads

        embedding_dim = self.transformer.num_attention_heads * self.transformer.attention_head_dim
        self.text_embedder_clip = PixArtAlphaTextProjection(in_features=pooled_projection_dim,
                                                            hidden_size=3 * embedding_dim,
                                                            # multiply by 3 acc. to COSMOS pipeline
                                                            act_fn="silu").to(transformer.device)
        self.text_embedder_clip.requires_grad_(True)  # We train only a single layer
        self.scales = nn.Parameter(torch.zeros(len(self.transformer_blocks), 3 * embedding_dim))
        self.w = w
        self.start_layer = start_layer

    def __getattr__(self, name):
        """
        Delegate reads to the inner transformer *only if it exists* and nn.Module
        didn't already resolve it (params/buffers/submodules/properties).
        """
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError as e:
            t = self.transformer
            if t is not None:
                return getattr(t, name)  # may still raise AttributeError (that's fine)
            # transformer not set yet; keep the original error
            raise e

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            pooled_projections: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            fps: Optional[int] = None,
            condition_mask: Optional[torch.Tensor] = None,
            padding_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
            apply_clip_pooled: bool = True,
            pooled_projections_positive: torch.Tensor = None,
            pooled_projections_negative: torch.Tensor = None,
            return_pooled_emb: bool = False,
    ) -> torch.Tensor:
        """
        The code is adopted from Diffusers library
        We've just added an additional layer for pooled CLIP embedding
        :param hidden_states: [B x L_img x dim_img]
        :param timestep: [B], from 1 to 0, as in flow matching
        :param encoder_hidden_states: [B x L_text x dim_text]; dim_text != dim_img in the Cosmos case; L_text - constant with padding
        :param attention_mask: [B x L_text], 1 for meaningful tokens, 0 for pad TODO double check that
        :param fps: None for image case
        :param condition_mask: None
        :param padding_mask: None
        :param return_dict: True
        """

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        # 1. Concatenate padding mask if needed & prepare attention mask
        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

        if self.config.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(hidden_states.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            hidden_states = torch.cat(
                [hidden_states, padding_mask.unsqueeze(2).repeat(batch_size, 1, num_frames, 1, 1)], dim=1
            )

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]

        # 2. Generate positional embeddings
        image_rotary_emb = self.rope(hidden_states, fps=fps)
        extra_pos_emb = self.learnable_pos_embed(hidden_states) if self.config.extra_pos_embed_type else None

        # 3. Patchify input
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.flatten(1, 3)  # [B, T, H, W, C] -> [B, THW, C]

        # 4. Timestep embeddings
        if timestep.ndim == 1:
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
            pooled_emb = self.text_embedder_clip(pooled_projections) if apply_clip_pooled else 0
            if pooled_projections_positive is not None:
                pooled_emb_delta = self.text_embedder_clip(pooled_projections_positive) - self.text_embedder_clip(
                    pooled_projections_negative)
                pooled_emb = pooled_emb + self.w * pooled_emb_delta
            # temb = temb + pooled_emb
        else:
            assert False

        # 5. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            
            if i >= self.start_layer and apply_clip_pooled:
                emb = temb + self.scales[i] * pooled_emb
            elif self.w == 0:
                emb = temb + self.scales[i] * pooled_emb
            else:
                emb = temb
                
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    embedded_timestep,
                    emb, #temb + self.scales[i] * pooled_emb,
                    image_rotary_emb,
                    extra_pos_emb,
                    attention_mask,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    embedded_timestep=embedded_timestep,
                    temb=emb, #temb + self.scales[i] * pooled_emb,
                    image_rotary_emb=image_rotary_emb,
                    extra_pos_emb=extra_pos_emb,
                    attention_mask=attention_mask,
                )

        # 6. Output norm & projection & unpatchify
        hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.unflatten(2, (p_h, p_w, p_t, -1))
        hidden_states = hidden_states.unflatten(1, (post_patch_num_frames, post_patch_height, post_patch_width))
        # NOTE: The permutation order here is not the inverse operation of what happens when patching as usually expected.
        # It might be a source of confusion to the reader, but this is correct
        hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        if not return_pooled_emb:
            return Transformer2DModelOutput(sample=hidden_states)
        else:
            return Transformer2DModelOutput(sample=hidden_states), pooled_emb
# ----------------------------------------------------------------------------------------------------------------------

