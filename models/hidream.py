import torch
from typing import Any, Dict, Optional, Tuple
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from typing import List
import logging

USE_PEFT_BACKEND = False
logger = logging.getLogger(__name__)


def encode_prompt(pipe, prompt):
    _, _, _, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                                    prompt=prompt,
                                    do_classifier_free_guidance=False,
                                    )
    return pooled_prompt_embeds


def forward_modulation_guidance(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.LongTensor = None,
        encoder_hidden_states_t5: torch.Tensor = None,
        encoder_hidden_states_llama3: torch.Tensor = None,
        pooled_embeds: torch.Tensor = None,
        img_ids: Optional[torch.Tensor] = None,
        img_sizes: Optional[List[Tuple[int, int]]] = None,
        hidden_states_masks: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        pooled_projections_1 = None,
        pooled_projections_0 =  None,
        w=0.5,
        start_layer=0,
        **kwargs,
    ):
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)

        if encoder_hidden_states is not None:
            deprecation_message = "The `encoder_hidden_states` argument is deprecated. Please use `encoder_hidden_states_t5` and `encoder_hidden_states_llama3` instead."
            deprecate("encoder_hidden_states", "0.35.0", deprecation_message)
            encoder_hidden_states_t5 = encoder_hidden_states[0]
            encoder_hidden_states_llama3 = encoder_hidden_states[1]

        if img_ids is not None and img_sizes is not None and hidden_states_masks is None:
            deprecation_message = (
                "Passing `img_ids` and `img_sizes` with unpachified `hidden_states` is deprecated and will be ignored."
            )
            deprecate("img_ids", "0.35.0", deprecation_message)

        if hidden_states_masks is not None and (img_ids is None or img_sizes is None):
            raise ValueError("if `hidden_states_masks` is passed, `img_ids` and `img_sizes` must also be passed.")
        elif hidden_states_masks is not None and hidden_states.ndim != 3:
            raise ValueError(
                "if `hidden_states_masks` is passed, `hidden_states` must be a 3D tensors with shape (batch_size, patch_height * patch_width, patch_size * patch_size * channels)"
            )

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # spatial forward
        batch_size = hidden_states.shape[0]
        hidden_states_type = hidden_states.dtype

        # Patchify the input
        if hidden_states_masks is None:
            hidden_states, hidden_states_masks, img_sizes, img_ids = self.patchify(hidden_states)

        # Embed the hidden states
        hidden_states = self.x_embedder(hidden_states)

        # 0. time
        timesteps = self.t_embedder(timesteps, hidden_states_type)
        p_embedder = self.p_embedder(pooled_embeds)
        temb = timesteps + p_embedder
        
        p_embedder_1, p_embedder_0 = self.p_embedder(pooled_projections_1), self.p_embedder(pooled_projections_0)
        if len(pooled_embeds) == 2:
            p_embedder[1] = p_embedder[1] + w * (p_embedder_1 - p_embedder_0)
            p_embedder[0] = p_embedder[0] - w * (p_embedder_1 - p_embedder_0)
        else:
            p_embedder = p_embedder + w * (p_embedder_1 - p_embedder_0)
        temb_new = timesteps + p_embedder

        encoder_hidden_states = [encoder_hidden_states_llama3[k] for k in self.config.llama_layers]

        if self.caption_projection is not None:
            new_encoder_hidden_states = []
            for i, enc_hidden_state in enumerate(encoder_hidden_states):
                enc_hidden_state = self.caption_projection[i](enc_hidden_state)
                enc_hidden_state = enc_hidden_state.view(batch_size, -1, hidden_states.shape[-1])
                new_encoder_hidden_states.append(enc_hidden_state)
            encoder_hidden_states = new_encoder_hidden_states
            encoder_hidden_states_t5 = self.caption_projection[-1](encoder_hidden_states_t5)
            encoder_hidden_states_t5 = encoder_hidden_states_t5.view(batch_size, -1, hidden_states.shape[-1])
            encoder_hidden_states.append(encoder_hidden_states_t5)

        txt_ids = torch.zeros(
            batch_size,
            encoder_hidden_states[-1].shape[1]
            + encoder_hidden_states[-2].shape[1]
            + encoder_hidden_states[0].shape[1],
            3,
            device=img_ids.device,
            dtype=img_ids.dtype,
        )
        ids = torch.cat((img_ids, txt_ids), dim=1)
        image_rotary_emb = self.pe_embedder(ids)

        # 2. Blocks
        block_id = 0
        initial_encoder_hidden_states = torch.cat([encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1)
        initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]
        for bid, block in enumerate(self.double_stream_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            cur_encoder_hidden_states = torch.cat(
                [initial_encoder_hidden_states, cur_llama31_encoder_hidden_states], dim=1
            )
            if block_id >= start_layer:
                _ = temb_new
            else:
                _ = temb
            
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, initial_encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    hidden_states_masks,
                    cur_encoder_hidden_states,
                    _,
                    image_rotary_emb,
                )
            else:
                hidden_states, initial_encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    hidden_states_masks=hidden_states_masks,
                    encoder_hidden_states=cur_encoder_hidden_states,
                    temb=_,
                    image_rotary_emb=image_rotary_emb,
                )
            initial_encoder_hidden_states = initial_encoder_hidden_states[:, :initial_encoder_hidden_states_seq_len]
            block_id += 1

        image_tokens_seq_len = hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
        hidden_states_seq_len = hidden_states.shape[1]
        if hidden_states_masks is not None:
            encoder_attention_mask_ones = torch.ones(
                (batch_size, initial_encoder_hidden_states.shape[1] + cur_llama31_encoder_hidden_states.shape[1]),
                device=hidden_states_masks.device,
                dtype=hidden_states_masks.dtype,
            )
            hidden_states_masks = torch.cat([hidden_states_masks, encoder_attention_mask_ones], dim=1)

        for bid, block in enumerate(self.single_stream_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            hidden_states = torch.cat([hidden_states, cur_llama31_encoder_hidden_states], dim=1)
            if block_id >= start_layer:
                _ = temb_new
            else:
                _ = temb
                
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    hidden_states_masks,
                    None,
                    _,
                    image_rotary_emb,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    hidden_states_masks=hidden_states_masks,
                    encoder_hidden_states=None,
                    temb=_,
                    image_rotary_emb=image_rotary_emb,
                )
            hidden_states = hidden_states[:, :hidden_states_seq_len]
            block_id += 1

        hidden_states = hidden_states[:, :image_tokens_seq_len, ...]
        output = self.final_layer(hidden_states, temb_new)
        output = self.unpatchify(output, img_sizes, self.training)
        if hidden_states_masks is not None:
            hidden_states_masks = hidden_states_masks[:, :image_tokens_seq_len]

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)