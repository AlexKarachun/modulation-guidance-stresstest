########################################################################################################################
#                              SAMPLER FUNCTION ACCORDING TO FLOW MATCHING REGIME                                      #
########################################################################################################################

import pandas as pd
import torch
import numpy as np
import os
import random
import torch.distributed as dist

from typing import List, Optional, Union
from accelerate.logging import get_logger
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm
from transformers import Siglip2TextModel

logger = get_logger(__name__)


                                ########################################
                                ########### SAMPLING PART ##############
                                ########################################


VALIDATION_PROMPTS = [
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    'A girl with pale blue hair and a cami tank top',
    'cute girl, Kyoto animation, 4k, high resolution',
    "Four cows in a pen on a sunny day",
    "Three dogs sleeping together on an unmade bed",
    "a deer with bird feathers, highly detailed, full body",
    "A cruise ship parked in a bathtub",
    "A monkey juggles tiny elephants",
    "kana arima solving a rubik cube",
]


# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def log_validation(
    args,
    transformer, vae,
    text_encoder_T5, tokenizer_T5,
    text_encoder_clip, tokenizer_clip,
    solver,
    noise_scheduler,
    accelerator,
    output_dir,
    global_step,
    num_images_per_prompt=4,
    apply_pooled_shifting=False,
):

    # Set validation prompts and seed
    device = 'cuda' if accelerator is None else accelerator.device
    validation_prompts = VALIDATION_PROMPTS
    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    weight_dtype = text_encoder_T5.dtype
    vae_scale_factor_spatial = 2 ** len(vae.temperal_downsample)
    video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
    assert weight_dtype == torch.bfloat16
    
    if args.shift_type == 'complexity':
        p1 = "Extremely complex, the highest quality"
        p2 = "very simple, no details at all"
    elif args.shift_type == 'realism':
        p1 = "Ultra-detailed, photorealistic, cinematic"
        p2 = "Low-res, flat, cartoonish"    
        
    output_dir = f'{output_dir}_w{args.w}_start{args.start_layer}_{args.shift_type}'
    os.makedirs(output_dir, exist_ok=True)
    
    ## Running part
    ## -------------------------------------------------------------------------------------------
    image_logs = []
    j = 0
    for _, prompt in enumerate(validation_prompts):

        # Get embeddings
        pooled_prompt_embeds_clip = get_clip_pooled_prompt_embeds(
            text_encoder_clip, tokenizer_clip, prompt,
            device=transformer.device)
        T5_prompt_embeds = get_T5_prompt_embeds(
            text_encoder_T5, tokenizer_T5, prompt,
            device=transformer.device)

        pooled_prompt_embeds_clip_positive = get_clip_pooled_prompt_embeds(
                            text_encoder_clip, tokenizer_clip, p1,
                            device=transformer.device) if apply_pooled_shifting else None
        pooled_prompt_embeds_clip_negative = get_clip_pooled_prompt_embeds(
                            text_encoder_clip, tokenizer_clip, p2,
                            device=transformer.device) if apply_pooled_shifting else None

        pooled_prompt_embeds_clip_uncond = get_clip_pooled_prompt_embeds(
                text_encoder_clip, tokenizer_clip, "",
                device=transformer.device) if args.cfg_scale > 1.0 else None
        T5_prompt_embeds_uncond = get_T5_prompt_embeds(
                text_encoder_T5, tokenizer_T5, "",
                device=transformer.device) if args.cfg_scale > 1.0 else None

        # Get sampling stuff
        sigmas = noise_scheduler.sigmas
        timesteps = noise_scheduler.timesteps

        images = []
        # Run
        for _ in tqdm(range(num_images_per_prompt)):
            latent = torch.randn(
                (1, 16, 1, args.height // 8, args.width // 8),
                generator=generator,
                device=device
            )
            latent = noise_scheduler.config.sigma_max * latent

            latents = solver.flow_matching_sampling(
                transformer, latent,
                T5_prompt_embeds, pooled_prompt_embeds_clip,
                T5_prompt_embeds_uncond, pooled_prompt_embeds_clip_uncond,
                cfg_scale=args.cfg_scale, sigmas=sigmas, timesteps=timesteps, apply_clip_pooled=args.apply_clip_pooled,
                pooled_prompt_embeds_positive=pooled_prompt_embeds_clip_positive, pooled_prompt_embeds_negative=pooled_prompt_embeds_clip_negative
            ).to(weight_dtype)

            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std / noise_scheduler.config.sigma_data + latents_mean
            video = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
            video = video_processor.postprocess_video(video, output_type="pil")
            image = [batch[0] for batch in video]
            images.append(image)
            image[0].save(f'{output_dir}/{global_step}_{j}.jpg')
            j += 1

        image_logs.append({"validation_prompt": prompt, "images": images})

    torch.cuda.empty_cache()
    ## -------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------



                                ########################################
                                ######### TEXT ENCODERS PART ###########
                                ########################################
            
            
def prepare_val_prompts(path, bs=20, max_cnt=5000):
    df = pd.read_csv(path)
    all_text = list(df['captions'])
    all_text = all_text[:max_cnt]

    num_batches = ((len(all_text) - 1) // (bs * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = np.array_split(np.array(all_text), num_batches)
    rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]

    index_list = np.arange(len(all_text))
    all_batches_index = np.array_split(index_list, num_batches)
    rank_batches_index = all_batches_index[dist.get_rank():: dist.get_world_size()]
    return rank_batches, rank_batches_index, all_text            


## CLIP Pooled embedding
## ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def get_clip_pooled_prompt_embeds(
    text_encoder, tokenizer,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
):
    device = device or 'cuda'
    tokenizer_max_length = 64 if isinstance(text_encoder, Siglip2TextModel) else 77

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1: -1])
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer_max_length} tokens: {removed_text}"
        )
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds
## ---------------------------------------------------------------------------------------------------------------------


## T5 embedding
## ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def get_T5_prompt_embeds(
    text_encoder, tokenizer,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    device = device or 'cuda'

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
        return_length=True,
        return_offsets_mapping=False,
    )
    text_input_ids = text_inputs.input_ids
    prompt_attention_mask = text_inputs.attention_mask.bool().to(device)
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1: -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    prompt_embeds = text_encoder(
        text_input_ids.to(device), attention_mask=prompt_attention_mask
    ).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    lengths = prompt_attention_mask.sum(dim=1).cpu()
    for i, length in enumerate(lengths):
        prompt_embeds[i, length:] = 0

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds
## ---------------------------------------------------------------------------------------------------------------------


                                ########################################
                                ######## DIFFUSION UTILS PART ##########
                                ########################################

# Sample sigma and weighting
## ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def process_and_save_latent(accelerator, VideoProcessor, vae, x0, noise_scheduler, name):
    video_processor = VideoProcessor(vae_scale_factor=2 ** len(vae.temperal_downsample))
    latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(x0.device, x0.dtype)
            )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                x0.device, x0.dtype
            )
    latents = x0 / latents_std / noise_scheduler.config.sigma_data + latents_mean
    video = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
    video = video_processor.postprocess_video(video, output_type="pil")
    image = [batch[0] for batch in video]
    if accelerator.is_main_process:
        t = 0
        for im in image:
            im.save(f'{t}_{name}.jpg')
            t += 1

@torch.no_grad()
def encode_image(vae, x0, noise_scheduler):
    latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(x0.device, x0.dtype)
            )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                x0.device, x0.dtype
            )
    encoded_image = vae.encode(x0.unsqueeze(2)).latent_dist.mode()
    latents = (encoded_image - latents_mean) * latents_std * noise_scheduler.config.sigma_data
    return latents

def drop_with_prob_inplace(prompts, p=0.1, seed=None):
    rnd = random.Random(seed) if seed is not None else random
    for i, s in enumerate(prompts):
        if rnd.random() < p:
            prompts[i] = ""
    return prompts

def replace_prompt_to_uncond(
    prompts_embeds: torch.Tensor,
    uncond_prompt_embeds: torch.Tensor,
    p: float = 0.05,
    seed = None
) -> torch.Tensor:
    """
    prompts_embeds: [B, L, D]
    uncond_prompt_embeds: [L, D]
    """
    uncond_prompt_embeds = uncond_prompt_embeds.squeeze()
    B, L, D = prompts_embeds.shape
    assert uncond_prompt_embeds.shape == (L, D)
    
    prompts_embeds = prompts_embeds.clone()

    replaced = (torch.rand(B, device=prompts_embeds.device) < p)
    idx = torch.nonzero(replaced, as_tuple=False).squeeze(1)
    
    if idx.numel():
        # Broadcasted assignment (no .expand); writes uncond into each selected row
        prompts_embeds[idx] = uncond_prompt_embeds
        
    return prompts_embeds
## ---------------------------------------------------------------------------------------------------------------------

