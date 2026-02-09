########################################################################################################################
#                                         MAIN TRAINING CODE                                                           #
########################################################################################################################
import warnings
warnings.filterwarnings("ignore")

import torch
import logging
import os
import datasets
import transformers
import diffusers
import copy
import torch.nn.functional as F

from transformers import (
    Siglip2TextModel,
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel, T5TokenizerFast
)
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    CosmosTransformer3DModel,
    AutoencoderKLWan
)
from diffusers.training_utils import cast_training_params
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from accelerate import Accelerator
from accelerate.logging import get_logger
from pathlib import Path

from cosmos_model import CosmosWithPooledCLIP
from sampler import FlowMatchingSolver
from utils import log_validation

logger = get_logger(__name__)
logging.getLogger('utils').setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)


                                    ########################################
                                    ######### TRAINING FUNCTION ############
                                    ########################################


# ----------------------------------------------------------------------------------------------------------------------
def generate(args):

    ## Prepare accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = prepare_accelertor(args, logging_dir, True)
    
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers("cosmos", config=tracker_config)
    
    ## Prepare models
    transformer, vae, text_encoder_clip, tokenizer_clip, text_encoder_T5, tokenizer_T5, noise_scheduler, weight_dtype \
        = prepare_models(args, accelerator)
    
    global_step = loading(args, accelerator, transformer)
    global_step = 0
    transformer = accelerator.prepare(transformer)

    ## Set up schedulers: diffusion and distilled models
    sigmas = torch.linspace(0, 1, 35, dtype=torch.float32) # HARDCODED
    noise_scheduler.set_timesteps(sigmas=sigmas, device='cuda')
    fm_solver = FlowMatchingSolver(noise_scheduler)
    
    # Generate
    args.apply_clip_pooled = True
    log_validation(
                   args,
                   transformer, vae,
                   text_encoder_T5, tokenizer_T5,
                   text_encoder_clip, tokenizer_clip,
                   fm_solver, noise_scheduler, accelerator,
                   global_step=global_step,
                   output_dir=logging_dir, num_images_per_prompt=1, apply_pooled_shifting=True
                    )
# ----------------------------------------------------------------------------------------------------------------------



                                    ########################################
                                    ############ PREPARATION ###############
                                    ########################################


# MODELS PREPARATION AND OTHER STUFF
# ----------------------------------------------------------------------------------------------------------------------
def prepare_models(args, accelerator):
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    if "siglip" in args.pretrained_model_name_or_path_clip:
        text_encoder_clip = Siglip2TextModel.from_pretrained(
            args.pretrained_model_name_or_path_clip,
        )
        tokenizer_clip = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path_clip
        )
    else:
        text_encoder_clip = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path_clip, subfolder="text_encoder",
        )
        tokenizer_clip = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path_clip, subfolder="tokenizer",
        )
    text_encoder_T5 = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
    )
    tokenizer_T5 = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer",
    )
    transformer = CosmosTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
    )
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
    )
    device = 'cuda' if accelerator is None else accelerator.device

    # We only train the additional layer for CLIP
    transformer.requires_grad_(False)
    text_encoder_clip.requires_grad_(False)
    text_encoder_T5.requires_grad_(False)
    vae.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer)
    # to half-precision as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.bfloat16

    # Move transformer, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    transformer.to(device, dtype=weight_dtype, memory_format=torch.channels_last)
    text_encoder_clip.to(device, dtype=weight_dtype)
    text_encoder_T5.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Update the model
    transformer = CosmosWithPooledCLIP(
        transformer,
        pooled_projection_dim=text_encoder_clip.text_model.final_layer_norm.weight.shape[0],
        w=args.w, start_layer=args.start_layer
    )
    
    # Make sure the trainable params are in float32.
    models = [transformer]
    cast_training_params(models, dtype=torch.float32)

    return (transformer, vae,
            text_encoder_clip, tokenizer_clip, text_encoder_T5, tokenizer_T5,
            noise_scheduler, weight_dtype)
# ----------------------------------------------------------------------------------------------------------------------


# ACCELERATOR PREPARATION AND OTHER STUFF
# ----------------------------------------------------------------------------------------------------------------------
def prepare_accelertor(args, logging_dir, find_unused_parameters=False):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(logging_dir, exist_ok=True)

    return accelerator
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def loading(args, accelerator, transformer):
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint") if d != "logs"]
            dirs = sorted(dirs, key=lambda x: int(x.split(".")[0].split("_")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            print(path)

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            state_dict = torch.load(os.path.join(args.output_dir, path), map_location="cpu")
            transformer.load_state_dict(state_dict, strict=False)
            transformer = transformer.to(accelerator.device)
            global_step = int(path.split(".")[0].split("_")[1])

            initial_global_step = global_step
    else:
        initial_global_step = 0

    return initial_global_step
# ----------------------------------------------------------------------------------------------------------------------
