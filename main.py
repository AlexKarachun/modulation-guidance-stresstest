from IPython.display import display
import types
import torch
from functools import partial
from diffusers import FluxPipeline
from models.flux_schnell import encode_prompt, forward_modulation_guidance
from pathlib import Path

# Import a model
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to('cuda')




# Define the hyperparametrs: 
# 1. Prompts: Generation prompt, positive and negative prompts
# 2. Modulation guidance strength (w)
# 3. Guidance start layer
prompt = 'A wolf on a plain background'
prompt_positive = "Ultra-detailed, photorealistic, cinematic"
prompt_negative = "Low-res, flat, cartoonish"

w = 3
start_layer = 5

# Get pooled CLIP embeddings
clip_positive = encode_prompt(pipe=pipe, prompt=prompt_positive)
clip_negative = encode_prompt(pipe=pipe, prompt=prompt_negative)

forward_logger = []

# Change forward of the pipe using the prompts and guidance weight
forward_modulation_guidance_partial = partial(forward_modulation_guidance, 
                                      pooled_projections_1=clip_positive, 
                                      pooled_projections_0=clip_negative,
                                      w=w, start_layer=start_layer, 
                                      log=forward_logger)
pipe.transformer.forward = types.MethodType(forward_modulation_guidance_partial, pipe.transformer)



# Run generation
seed = 0
images = pipe([prompt] * 4,
              guidance_scale=0.0,
              num_inference_steps=4,
              max_sequence_length=256,
              generator=[torch.Generator("cpu").manual_seed(seed) for _ in range(4)],
              output_type='pil').images





# main_dir = Path('generations')
# test_dir = main_dir / 'test'
# test_dir.mkdir(parents=True, exist_ok=True)

# for i, img in enumerate(images):
#     img.save(test_dir / f"image_{i}.png")