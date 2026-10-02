import torch
import numpy as np
import logging
from typing import Any, Dict, Optional, Union
from diffusers.models.modeling_outputs import Transformer2DModelOutput
import torch.nn.functional as F


USE_PEFT_BACKEND = False
logger = logging.getLogger(__name__)

def encode_prompt(pipe, prompt):
    _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                                prompt=prompt,
                                prompt_2=None,
                                prompt_embeds=None,
                                pooled_prompt_embeds=None,
                                device='cuda',
                                num_images_per_prompt=1,
                                max_sequence_length=256,
                                lora_scale=None)
    return pooled_prompt_embeds


def register_gate_scaling(transformer, factor, start_layer=0, n_cond=5, cond=4):
    """
    Multiply the gate outputs of AdaLN modulation by `factor` for batch rows with
    `row % n_cond == cond`, in every block whose global index is >= start_layer.
    Double blocks: gate_msa, gate_mlp of both image and text streams; single blocks: gate.
    Returns hook handles (call .remove() on them to undo).
    """
    def make_hook(n_chunks, gate_ids):
        def hook(module, inputs, output):
            rows = (torch.arange(output.shape[0], device=output.device) % n_cond == cond)[:, None]
            chunks = list(output.chunk(n_chunks, dim=1))
            for g in gate_ids:
                scaled = (chunks[g].float() * factor).to(chunks[g].dtype)
                chunks[g] = torch.where(rows, scaled, chunks[g])
            return torch.cat(chunks, dim=1)
        return hook

    handles = []
    n_double = len(transformer.transformer_blocks)
    for i, block in enumerate(transformer.transformer_blocks):
        if i >= start_layer:
            # chunks: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
            handles.append(block.norm1.linear.register_forward_hook(make_hook(6, (2, 5))))
            handles.append(block.norm1_context.linear.register_forward_hook(make_hook(6, (2, 5))))
    for i, block in enumerate(transformer.single_transformer_blocks):
        if n_double + i >= start_layer:
            # chunks: shift, scale, gate
            handles.append(block.norm.linear.register_forward_hook(make_hook(3, (2,))))
    return handles


def forward_modulation_guidance(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
        pooled_projections_1 = None,
        pooled_projections_0 =  None,
        w=0.5,
        start_layer=0,
        end_layer=1000,
        log=None,
        n_cond=4,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    """
    The [`FluxTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `torch.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )
            
    if log is not None: log.append(dict())
    
    B = hidden_states.shape[0]
    
    hidden_states = self.x_embedder(hidden_states)
    timestep = timestep.to(hidden_states.dtype) * 1000
    
    if log is not None: log[-1]['timestep'] = timestep.cpu()
    
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    
    # Modulation guidance
    if w > 0:
        temb_1 = (
            self.time_text_embed(timestep, pooled_projections_1)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections_1)
        )
        temb_0 = (
            self.time_text_embed(timestep, pooled_projections_0)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections_0)
        )
        temb_delta = temb_1 - temb_0
        temb_new = temb + w * temb_delta
        
        temb_mix = torch.zeros_like(temb_new)
        
        for j in range(B):
            
            if j % n_cond == 0: temb_mix[j] = temb[j]

            if j % n_cond == 1: temb_mix[j] = temb_new[j]

            if j % n_cond == 2: temb_mix[j] = (temb_new[j] * torch.norm(temb[j], dtype=torch.float32) / torch.norm(temb_new[j], dtype=torch.float32)).to(dtype=temb.dtype)

            if j % n_cond == 3: temb_mix[j] = (temb[j] * torch.norm(temb_new[j], dtype=torch.float32) / torch.norm(temb[j], dtype=torch.float32)).to(dtype=temb.dtype)

            # x unchanged here; its gates are scaled by register_gate_scaling
            if j % n_cond == 4: temb_mix[j] = temb[j]


            if j % n_cond == 1:
                if log is not None: log[-1][f'p={j - 1}:||x+-x-||'] = torch.norm(temb_delta[j].float(), dim=-1).cpu()
        
                cos_0 = F.cosine_similarity(temb_mix[j - 1].float(), temb_delta[j - 1].float(), dim=-1)
                cos_1 = F.cosine_similarity(temb_mix[j].float(), temb_delta[j - 1].float(), dim=-1)
                
                cos_rad_0 = torch.acos(cos_0.float().clamp(-1, 1)).cpu()
                cos_rad_1 = torch.acos(cos_1.float().clamp(-1, 1)).cpu()
                
                
                if log is not None: log[-1][f'p={j - 1}:angle(x,x+-x-)'] = cos_rad_0
                if log is not None: log[-1][f'p={j - 1}:angle(x*,x+-x-)'] = cos_rad_1    
                
                
        # if log is not None: log[-1]['temb_mix'] = temb_mix.float().cpu()
        if log is not None: log[-1]['norm(temb_mix,dim=-1)'] = torch.norm(temb_mix, dtype=torch.float32, dim=-1).cpu()
        
        
        

        
        
        
    else:
        temb_mix = temb
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)


    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    global_index_block = 0
    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
            )

        else:
            if global_index_block >= start_layer and global_index_block <= end_layer:
                _ = temb_mix
            else:
                _ = temb
                            
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=_,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        global_index_block += 1

        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            interval_control = int(np.ceil(interval_control))
            # For Xlabs ControlNet.
            if controlnet_blocks_repeat:
                hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                )
            else:
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

    for index_block, block in enumerate(self.single_transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                temb,
                image_rotary_emb,
            )

        else:
            if global_index_block >= start_layer and global_index_block <= end_layer:
                _ = temb_mix
            else:
                _ = temb

            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=_,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        global_index_block += 1
        # controlnet residual
        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
            interval_control = int(np.ceil(interval_control))
            hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
            )

    #hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)