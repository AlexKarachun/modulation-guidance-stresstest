########################################################################################################################
#                              SAMPLER FUNCTION ACCORDING TO FLOW MATCHING REGIME                                      #
########################################################################################################################

import torch
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------------------------
class FlowMatchingSolver:

    def __init__(
            self,
            noise_scheduler,
    ):
        """
        :param noise_scheduler: scheduler from the diffusers
        """
        self.noise_scheduler = noise_scheduler

    def flow_matching_single_step(self, sample, model_output, sigma, sigma_next):
        prev_sample = sample + (sigma_next - sigma) * model_output
        return prev_sample

    @torch.no_grad()
    def flow_matching_sampling(
        self,
        model, latent,
        prompt_embeds, pooled_prompt_embeds,
        uncond_prompt_embeds, uncond_pooled_prompt_embeds,
        cfg_scale, sigmas=None, timesteps=None, apply_clip_pooled=True,
        pooled_prompt_embeds_positive=None, pooled_prompt_embeds_negative=None,
    ):
        try:
            dtype = model.dtype
        except AttributeError:
            dtype = model.module.dtype

        sigmas = self.noise_scheduler.sigmas if sigmas is None else sigmas
        timesteps = self.noise_scheduler.timesteps if timesteps is None else timesteps

        batch_size, num_channels, num_frames, height, width = latent.shape
        padding_mask = latent.new_zeros(1, 1, int(height * 8), int(width * 8), dtype=dtype)

        for j in range(len(timesteps)):
            sigma = sigmas[j].to(device=model.device)
            sigma_next = sigmas[j + 1].to(device=model.device)

            current_t = sigma / (sigma + 1)
            c_in = 1 - current_t
            c_skip = 1 - current_t
            c_out = -current_t
            timestep = current_t.expand(latent.shape[0]).to(dtype)

            latent_model_input = latent * c_in
            latent_model_input = latent_model_input.to(dtype)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                F_pred = model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    padding_mask=padding_mask,
                    pooled_projections=pooled_prompt_embeds,
                    apply_clip_pooled=apply_clip_pooled,
                    pooled_projections_positive=pooled_prompt_embeds_positive,
                    pooled_projections_negative=pooled_prompt_embeds_negative
                )[0]
                x0_pred = (c_skip * latent + c_out * F_pred.float()).to(dtype)

                if cfg_scale > 1.0:
                    F_pred_uncond = model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            padding_mask=padding_mask,
                            encoder_hidden_states=uncond_prompt_embeds,
                            pooled_projections=uncond_pooled_prompt_embeds,
                            apply_clip_pooled=apply_clip_pooled,
                            pooled_projections_positive=None,
                            pooled_projections_negative=None
                        )[0]
                    x0_pred_uncond = (c_skip * latent + c_out * F_pred_uncond.float()).to(dtype)
                    x0_pred = x0_pred + cfg_scale * (x0_pred - x0_pred_uncond)

            noise_pred = (latent - x0_pred) / sigma
            latent = self.flow_matching_single_step(latent, noise_pred,
                                                    sigma,
                                                    sigma_next)
        return latent
# ----------------------------------------------------------------------------------------------------------------------